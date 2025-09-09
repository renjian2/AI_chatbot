import streamlit as st
from file_utils import load_and_process_file, calculate_tokens
from config import SUPPORTED_FILE_TYPES # Import SUPPORTED_FILE_TYPES

def handle_file_upload(llm):
    """Handles file uploads and processing in the sidebar."""
    st.sidebar.subheader("✿ File Upload")
    # Dynamically generate the list of supported file extensions
    supported_extensions = list(SUPPORTED_FILE_TYPES.keys())
    uploaded_file = st.sidebar.file_uploader("Upload a file", type=supported_extensions)
    if uploaded_file:
        # Check file size (in bytes)
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > 200:  # Limit to 200MB
            st.sidebar.error(f"❌ File '{uploaded_file.name}' is too large ({file_size_mb:.1f}MB). Maximum file size is 200MB.")
            return
            
        if uploaded_file.name not in [f["name"] for f in st.session_state.files]:
            try:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    file_bytes = uploaded_file.read()
                    # Reset file pointer to beginning
                    uploaded_file.seek(0)
                    
                    processed_content = load_and_process_file(file_bytes, uploaded_file.name, _llm=llm)

                    st.session_state.files.append({
                        "name": uploaded_file.name,
                        "content": processed_content,
                        "use_for_context": True
                    })
                    st.sidebar.success(f"Uploaded {uploaded_file.name}")

                    # Only add to RAG store if we have content
                    if processed_content:
                        # Add the full content to RAG store for comprehensive search
                        # This ensures that even if only chunks are used for context,
                        # the full content is available for RAG retrieval
                        full_content = ""
                        if isinstance(processed_content, list):
                            # Join all chunks to get the full content
                            flattened_content = []
                            for item in processed_content:
                                if isinstance(item, list):
                                    flattened_content.extend(item)
                                else:
                                    flattened_content.append(item)
                            # Filter out empty or very short chunks
                            filtered_content = [chunk for chunk in flattened_content if chunk and len(chunk.strip()) > 10]
                            full_content = "\n\n".join(filtered_content)
                        elif isinstance(processed_content, str):
                            full_content = processed_content
                        
                        # Add the full content to RAG store with optimized chunking for large files
                        if full_content.strip():
                            # Split into paragraphs but keep the full content available
                            paragraphs = [p for p in full_content.split("\n\n") if p and len(p.strip()) > 10]
                            
                            # For large files, implement hierarchical chunking strategy:
                            # 1. Store the full content for comprehensive search
                            # 2. Store individual paragraphs for granular retrieval  
                            # 3. Create overlapping chunks for better context coverage
                            st.session_state.rag_store.add_documents([full_content], source=f"{uploaded_file.name} (full)")
                            st.session_state.rag_store.add_documents(paragraphs, source=f"{uploaded_file.name} (paragraphs)")
                            
                            # For very large files, create hierarchical chunks using our new utilities
                            from ui.chunking_utils import adaptive_chunk_sizing
                            if len(paragraphs) > 30:  # Only for larger files
                                # Create medium-sized chunks using adaptive sizing
                                total_chars = sum(len(p) for p in paragraphs)
                                chunk_size, num_chunks = adaptive_chunk_sizing(total_chars, 8192, min_chunks=5, max_chunks=20)
                                
                                # Create overlapping chunks
                                medium_chunks = []
                                step_size = max(1, len(paragraphs) // num_chunks)
                                overlap = max(1, step_size // 3)
                                
                                for i in range(0, len(paragraphs), step_size):
                                    start_idx = max(0, i - overlap if i > 0 else 0)
                                    end_idx = min(len(paragraphs), i + step_size + overlap)
                                    chunk = "\n\n".join(paragraphs[start_idx:end_idx])
                                    if chunk.strip():
                                        medium_chunks.append(chunk)
                                
                                st.session_state.rag_store.add_documents(medium_chunks, source=f"{uploaded_file.name} (chunks)")
                    st.rerun()
            except MemoryError as e:
                st.sidebar.error(f"❌ Failed to process file '{uploaded_file.name}': File is too large for available memory. Try uploading a smaller file or splitting it into parts.")
                # Log the error for debugging
                import logging
                logging.error(f"Memory error processing file {uploaded_file.name}: {str(e)}")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to process file '{uploaded_file.name}': {type(e).__name__}: {str(e)}")
                # Only show traceback in development mode
                if st.secrets.get("ENVIRONMENT", "development") == "development":
                    import traceback
                    st.sidebar.error(traceback.format_exc())
        else:
            st.sidebar.info(f"File '{uploaded_file.name}' already uploaded for this session.")