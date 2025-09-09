import faiss
import numpy as np
import os
import pickle
import logging
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class SimpleRAGStore:
    def __init__(self, session_id: str, embedding_model, index_dir="rag_data"):
        self.session_id = session_id
        self.embedding_model = embedding_model
        self.index_dir = index_dir
        self.text_chunks = []
        self.embeddings = None
        self.index = None
        self._load_index()

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.array([])
        try:
            embeddings = self.embedding_model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False  # Disable progress bar for better UX
            )
            return embeddings
        except Exception as e:
            logger.error(f"[ERROR] Embedding failed: {e}")
            return np.array([])

    def add_documents(self, chunks: List[str], source: str):
        """
        Adds new text chunks and their embeddings to the RAG store.
        Handles de-duplication and updates the FAISS index.
        """
        # Handle None or non-string values in chunks
        if chunks is None:
            logger.warning("add_documents called with None chunks, skipping.")
            return
            
        # Filter and clean chunks
        cleaned_chunks = []
        for c in chunks:
            # Handle different data types
            if c is None:
                continue
            if not isinstance(c, str):
                c = str(c)
            if c.strip():  # Only add non-empty chunks
                cleaned_chunks.append(c)
        
        new_entries = [{"text": c, "source": source} for c in cleaned_chunks]

        # Prevent adding duplicate text chunks
        existing_texts = {chunk["text"] for chunk in self.text_chunks}
        filtered_entries = [e for e in new_entries if e["text"] not in existing_texts]

        if not filtered_entries:
            logger.warning("No new unique chunks to add.")
            return

        # Generate embeddings for the new, unique chunks
        vectors = self.embed([e["text"] for e in filtered_entries])
        if vectors.size == 0:
            logger.warning("Embedding returned no vectors for new chunks.")
            return

        # Extend the list of text chunks
        self.text_chunks.extend(filtered_entries)

        # Initialize or update the FAISS index and embeddings array
        if self.index is None:
            # If no index exists, create a new one and set initial embeddings
            self.index = faiss.IndexFlatIP(vectors.shape[1]) # Use Inner Product for normalized embeddings (cosine similarity)
            self.embeddings = vectors
            logger.info(f"Created FAISS index with dimension {vectors.shape[1]}")
        else:
            # If index exists, vertically stack new embeddings with existing ones
            # This is where the ValueError occurred if self.embeddings was malformed
            if self.embeddings is None or self.embeddings.shape[1] != vectors.shape[1]:
                # This case should ideally be handled by _load_index, but as a safeguard
                logger.warning("Rebuilding self.embeddings due to shape mismatch or None state.")
                # Re-embed all text chunks to ensure self.embeddings is consistent
                self.embeddings = self.embed([entry["text"] for entry in self.text_chunks])
                if self.embeddings.size == 0:
                    logger.error("Failed to rebuild self.embeddings. Aborting add_documents.")
                    return
            else:
                self.embeddings = np.vstack((self.embeddings, vectors))

            logger.info(f"FAISS index now has {self.index.ntotal + len(vectors)} total vectors (before add)")

        # Add vectors to the FAISS index
        if vectors.size > 0:
            self.index.add(vectors)
            logger.info(f"Added {len(vectors)} new vectors to FAISS index.")
        self._save_index() # Save the updated index and metadata

    def add_chat_message(self, text: str, role: str):
        """
        Stores a chat message (user or assistant) as a vector for long-term recall.
        This is similar to add_documents but specifically for chat history.
        """
        entry = {
            "text": text,
            "source": "chat",
            "role": role,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Prevent adding duplicate chat messages
        if any(e["text"] == text and e.get("role") == role for e in self.text_chunks):
            logger.warning("Duplicate chat message detected; skipping.")
            return

        embedding = self.embed([text])
        if embedding.size == 0:
            logger.warning("Failed to embed chat message.")
            return

        if self.index is None:
            self.index = faiss.IndexFlatIP(embedding.shape[1])
            self.embeddings = embedding
            self.index.add(embedding)
            logger.info(f"Created FAISS index with dimension {embedding.shape[1]} from chat message.")
        else:
            if self.embeddings is None or self.embeddings.shape[1] != embedding.shape[1]:
                logger.warning("Rebuilding self.embeddings for chat message due to shape mismatch or None state.")
                self.embeddings = self.embed([entry["text"] for entry in self.text_chunks])
                if self.embeddings.size == 0:
                    logger.error("Failed to rebuild self.embeddings for chat message. Aborting.")
                    return
            self.embeddings = np.vstack((self.embeddings, embedding))
            self.index.add(embedding)
            logger.info("Added 1 chat message to FAISS index.")

        self.text_chunks.append(entry)
        self._save_index()

    def search(self, query: str, top_k=5, similarity_threshold=0.2) -> List[Tuple[str, str, float]]:
        """
        Performs a similarity search against the stored embeddings.
        Returns a list of (text_chunk, source, similarity_score) tuples.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("FAISS index is empty; no search performed.")
            return []

        query_vec = self.embed([query])
        if query_vec.size == 0:
            logger.warning("Query embedding failed; no search performed.")
            return []

        # Perform the search
        # min(top_k, self.index.ntotal) ensures we don't ask for more results than available
        scores, indices = self.index.search(query_vec, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if score < similarity_threshold:
                continue
            if 0 <= idx < len(self.text_chunks): # Ensure index is within bounds
                chunk = self.text_chunks[idx]
                results.append((chunk["text"], chunk.get("source", "unknown"), float(score)))
        return results

    def evaluate_retrieval(self, query: str, ground_truth_doc_id: str) -> dict:
        """
        Evaluates the retrieval performance for a given query.
        """
        results = self.search(query)
        retrieved_ids = [res[1] for res in results]
        
        if ground_truth_doc_id in retrieved_ids:
            return {"hit": True, "rank": retrieved_ids.index(ground_truth_doc_id) + 1}
        else:
            return {"hit": False, "rank": -1}

    def clear(self):
        """Clears all stored chunks and deletes the FAISS index and metadata files."""
        self.text_chunks = []
        self.embeddings = None
        self.index = None

        # Delete the physical files
        if os.path.exists(self._index_path()):
            os.remove(self._index_path())
            logger.info(f"Deleted index file: {self._index_path()}")

        if os.path.exists(self._meta_path()):
            os.remove(self._meta_path())
            logger.info(f"Deleted metadata file: {self._meta_path()}")
        
        # Also delete the embeddings file
        if os.path.exists(self._embeddings_path()):
            os.remove(self._embeddings_path())
            logger.info(f"Deleted embeddings file: {self._embeddings_path()}")

        logger.info(f"Cleared FAISS index and data for session '{self.session_id}'")

    def _index_path(self):
        """Returns the full path to the FAISS index file for the current session."""
        return os.path.join(self.index_dir, f"{self.session_id}.faiss")

    def _meta_path(self):
        """Returns the full path to the metadata (text chunks) file for the current session."""
        return os.path.join(self.index_dir, f"{self.session_id}_meta.pkl")

    def _embeddings_path(self):
        """Returns the full path to the embeddings NumPy file for the current session."""
        return os.path.join(self.index_dir, f"{self.session_id}_embeddings.npy")

    def _save_index(self):
        """Saves the FAISS index, text chunks (metadata), and embeddings to disk."""
        os.makedirs(self.index_dir, exist_ok=True) # Ensure the directory exists

        if self.index and self.index.ntotal > 0: # Only save if index has vectors
            faiss.write_index(self.index, self._index_path())
            logger.info(f"FAISS index saved to {self._index_path()}")
        else:
            # If index is empty, ensure old files are removed if they exist
            if os.path.exists(self._index_path()):
                os.remove(self._index_path())
                logger.info(f"Removed empty FAISS index file: {self._index_path()}")

        # Always save text_chunks, even if empty, to reflect current state
        with open(self._meta_path(), "wb") as f:
            pickle.dump(self.text_chunks, f)
        logger.info(f"Metadata saved to {self._meta_path()}")

        # Save embeddings array if it exists and is not empty
        if self.embeddings is not None and self.embeddings.size > 0:
            np.save(self._embeddings_path(), self.embeddings)
            logger.info(f"Embeddings saved to {self._embeddings_path()}")
        else:
            # If embeddings are empty, ensure old file is removed if it exists
            if os.path.exists(self._embeddings_path()):
                os.remove(self._embeddings_path())
                logger.info(f"Removed empty embeddings file: {self._embeddings_path()}")


    def _load_index(self):
        """
        Attempts to load the FAISS index, text chunks, and embeddings from disk.
        Handles cases where files might not exist or are corrupted.
        """
        try:
            os.makedirs(self.index_dir, exist_ok=True) # Ensure index_dir exists

            index_path = self._index_path()
            meta_path = self._meta_path()
            embeddings_path = self._embeddings_path()

            loaded_index_exists = os.path.exists(index_path)
            loaded_meta_exists = os.path.exists(meta_path)
            loaded_embeddings_exists = os.path.exists(embeddings_path)

            if loaded_index_exists:
                self.index = faiss.read_index(index_path)
                logger.info(f"FAISS index loaded from {index_path}")
            else:
                logger.info(f"No FAISS index found at {index_path}")

            if loaded_meta_exists:
                with open(meta_path, "rb") as f:
                    self.text_chunks = pickle.load(f)
                logger.info(f"Loaded {len(self.text_chunks)} chunks for session '{self.session_id}'")
            else:
                logger.info(f"No metadata found at {meta_path}")
            
            if loaded_embeddings_exists:
                self.embeddings = np.load(embeddings_path)
                logger.info(f"Embeddings loaded from {embeddings_path} with shape {self.embeddings.shape}")
            else:
                logger.info(f"No embeddings file found at {embeddings_path}")

            # Consistency check and reconstruction if necessary
            # If text chunks are loaded but embeddings are missing/inconsistent, re-embed
            if self.text_chunks and (self.embeddings is None or self.embeddings.shape[0] != len(self.text_chunks)):
                logger.warning("Embeddings inconsistent with text chunks or missing. Re-embedding all chunks.")
                re_embedded_vectors = self.embed([entry["text"] for entry in self.text_chunks])

                if re_embedded_vectors.size > 0:
                    self.embeddings = re_embedded_vectors
                    logger.info(f"Reconstructed self.embeddings with shape {self.embeddings.shape}")
                    # If index is missing or inconsistent, rebuild it
                    if self.index is None or self.index.ntotal != self.embeddings.shape[0]:
                        logger.warning("FAISS index inconsistent or missing. Rebuilding index from reconstructed embeddings.")
                        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
                        self.index.add(self.embeddings)
                        logger.info(f"Rebuilt FAISS index from {self.embeddings.shape[0]} chunks.")
                else:
                    logger.warning("Re-embedding loaded chunks resulted in empty vectors. Clearing index and chunks.")
                    self.index = None
                    self.text_chunks = []
                    self.embeddings = None
            elif not self.text_chunks:
                # If text_chunks is empty, ensure index and embeddings are also cleared
                self.index = None
                self.embeddings = None
                logger.info("No text chunks loaded, ensuring FAISS index and embeddings are cleared.")


        except Exception as e:
            logger.error(f"Error loading FAISS index or metadata for session '{self.session_id}': {e}")
            # Ensure a clean state if loading fails
            self.index = None
            self.text_chunks = []
            self.embeddings = None
            # Clean up potentially corrupted files to prevent future errors
            if os.path.exists(index_path):
                os.remove(index_path)
            if os.path.exists(meta_path):
                os.remove(meta_path)
            if os.path.exists(embeddings_path):
                os.remove(embeddings_path)
            logger.info(f"Cleaned up potentially corrupted index files for session '{self.session_id}'.")

