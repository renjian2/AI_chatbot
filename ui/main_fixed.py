import streamlit as st
import json
import os
import time
import re
from fpdf import FPDF
from io import BytesIO
import base64
import markdown
try:
    from xhtml2pdf import pisa
except ImportError:
    pisa = None
from functools import lru_cache
import logging

# Configure logging if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("chatbot_debug.log"),
            logging.StreamHandler()
        ]
    )

from file_utils import calculate_tokens, trim_context_to_fit
from llm_utils import get_llm_callable
from config import BUFFER_TOKENS, MODEL_CONFIGS, RAG_SEARCH_TOP_K, RAG_SIMILARITY_THRESHOLD, MAX_FILE_CHUNK_TOKENS
from db_utils import init_db, clear_chat_history, delete_session, save_chat_summary, update_last_chat_message, load_chat_summary
from ui.utils import get_embedding_model, summarize_chat_history_sync, post_process_response, remove_duplicate_sentences, store_chat_message, clean_model_response, remove_qwen_think_block
from ui.pdf_export import export_chat_to_pdf
from ui.dynamic_response_control import enforce_response_limits
from ui.session_manager import initialize_session_state, manage_sessions
from ui.file_handler import handle_file_upload
from chat_utils import build_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_and_process_response(llm, prompt, user_input, temperature, top_p, repeat_penalty, min_response_tokens):
    """Generates, streams, and processes the LLM response."""
    response_container = st.empty()
    full_response = ""
    response_text = ""

    try:
        model_n_ctx = MODEL_CONFIGS.get(st.session_state.model_name, {}).get("n_ctx", 4096)
        prompt_tokens = calculate_tokens(prompt, llm)
        dynamic_max_tokens = model_n_ctx - prompt_tokens - BUFFER_TOKENS
        
        if dynamic_max_tokens < 0:
            st.error(f"Prompt tokens ({prompt_tokens}) exceed the context window. Please shorten the input or clear history.")
            return "", ""

        response_max_tokens = dynamic_max_tokens

        if response_max_tokens < min_response_tokens:
            st.warning(f"Available space for response is only {response_max_tokens} tokens, less than desired {min_response_tokens}. Response may be short.")

        response_max_tokens = min(response_max_tokens, 4096)

        response_stream = llm(
            prompt,
            max_tokens=response_max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            repeat_penalty=repeat_penalty,
        )

        for chunk in response_stream:
            if isinstance(chunk, dict) and "choices" in chunk and chunk["choices"]:
                text = chunk["choices"][0].get("text", "")
                if text:
                    full_response += text
                    response_container.markdown(full_response + "▌")
        
        response_container.markdown(full_response)
        response_text = full_response

    except ValueError as e:
        response_text = f"❌ {type(e).__name__}: {str(e)}"
        st.error(response_text)

    # Post-processing
    if "qwen" in st.session_state.model_name.lower() and st.session_state.get("remove_think_step", True):
        response_text = remove_qwen_think_block(response_text)
    response_text = clean_model_response(response_text, st.session_state.model_name)
    response_text = post_process_response(response_text, prompt)
    response_text = enforce_response_limits(response_text, user_input)
    response_text = response_text.replace("▌", "").strip()

    response_container.markdown(response_text)
    return response_text, full_response

def run_ui():
    """Main function to run the Streamlit application UI."""
    st.set_page_config(page_title="✿ AArete Chat", layout="wide")

    # Load and apply custom CSS
    try:
        css_file_path = os.path.join(os.path.dirname(__file__), "styles.css")
        if os.path.exists(css_file_path):
            with open(css_file_path, "r") as f:
                css_content = f.read()
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    except Exception as e:
        print(f"Warning: Failed to apply custom CSS: {e}")


    st.title("✿ AArete Chat")
    st.divider()

    init_db()
    embedding_model = get_embedding_model()

    image_path = "static/logo.png"
    try:
        if os.path.exists(image_path):
            st.sidebar.image(image_path, width=130)
    except Exception as e:
        st.sidebar.markdown(f'''<div style="text-align: center;">**Ollama Chat**</div>''')
        if st.session_state.get('debug_mode', False):
            st.sidebar.warning(f"Failed to load logo: {e}")

    initialize_session_state(embedding_model)
    manage_sessions(embedding_model)

    st.sidebar.markdown(f"**Active Session:** `{st.session_state.session_id}`")

    st.sidebar.header("✿ Model Selection")
    model_name = st.sidebar.selectbox("Choose a model", list(MODEL_CONFIGS.keys()), index=0, key="model_selection")
    st.session_state.model_name = model_name

    with st.spinner("Loading model..."):
        try:
            model_config = MODEL_CONFIGS.get(model_name, {})
            llm = get_llm_callable(model_config)
        except Exception as e:
            st.error(f"Failed to load model: {model_name}. Error: {str(e)}")
            st.stop()

    if llm is None:
        st.error(f"Failed to load model: {model_name}. Please check your configuration and model path.")
        st.stop()

    model_n_ctx = MODEL_CONFIGS.get(st.session_state.model_name, {}).get("n_ctx", 4096)
    
    # Calculate tokens used by the prompt template
    dummy_prompt = build_prompt(user_input="", model_name=st.session_state.model_name, chat_history=[], rag_context="", file_context="")
    template_tokens = calculate_tokens(dummy_prompt, llm)
    
    available_input_tokens = model_n_ctx - BUFFER_TOKENS - template_tokens

    handle_file_upload(llm)

    with st.sidebar.expander("✿ Settings"):
        st.subheader("Model Settings")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
        top_p = st.slider("Top P", 0.1, 1.0, 0.95, 0.05)
        repeat_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.1, 0.05)
        min_response_tokens = st.slider("Min Response Tokens", 128, 2048, 1024, 128)
        if 'qwen' in st.session_state.model_name.lower():
            st.session_state.remove_think_step = st.toggle(
                "Remove <think> step", 
                value=st.session_state.get("remove_think_step", True),
                help="If enabled, the <think>...</think> block will be removed from the Qwen model's output."
            )
        
        st.subheader("Session Management")
        if st.button("✿ Clear RAG Index for this session"):
            st.session_state.rag_store.clear()
            st.sidebar.success("Cleared RAG index and metadata.")
            st.rerun()
        if st.button("✿ Clear Chat for this session"):
            st.session_state.chat_history = []
            st.session_state.files = []
            clear_chat_history(st.session_state.session_id)
            st.rerun()
        if st.button("✿ Delete this session"):
            delete_session(st.session_state.session_id)
            st.session_state.chat_history = []
            st.session_state.files = []
            st.session_state.session_id = "guest"
            st.rerun()

    # Display chat messages from history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if user_input := st.chat_input("Ask something..."):
        # Add user message to chat history and database
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        store_chat_message(st.session_state.session_id, "user", user_input, st.session_state.rag_store)
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # 1. Initialize RAG context
                rag_context = ""
                
                # 2. Get context from uploaded files
                # Optimized file context handling for large files
                BASE_FILE_CHUNK_TOKENS = 512  # Smaller base chunks for better granularity
                MAX_CONTEXT_CHUNKS_PER_FILE = 15  # Allow more chunks for better coverage
                
                # Dynamic adjustment based on available context space
                # Reserve 35% of available input tokens for file context
                max_file_context_tokens = int(available_input_tokens * 0.35)
                
                # Import chunking utilities
                from ui.chunking_utils import select_diverse_chunks, rank_chunks_by_importance, adaptive_chunk_sizing
                
                file_context_str = ""
                used_files = []
                total_file_tokens = 0
                
                # Calculate total tokens for all files first to determine distribution
                file_token_counts = []
                for f in st.session_state.get('files', []):
                    if f.get('use_for_context'):
                        content = f.get('content')
                        if isinstance(content, list):
                            content = "\n\n".join(content)
                        elif not isinstance(content, str):
                            content = str(content)
                        file_tokens = calculate_tokens(content, llm)
                        file_token_counts.append((f, file_tokens))
                
                # Distribute available tokens among files
                if file_token_counts:
                    # Calculate proportional tokens per file
                    total_tokens = sum(tokens for _, tokens in file_token_counts)
                    for f, file_tokens in file_token_counts:
                        # Allocate tokens proportionally, but ensure a minimum
                        allocated_tokens = max(
                            MAX_FILE_CHUNK_TOKENS,  # Minimum allocation
                            min(
                                int((file_tokens / total_tokens) * max_file_context_tokens),  # Proportional allocation
                                file_tokens,  # Don't allocate more than the file has
                                max_file_context_tokens // len(file_token_counts) * 3  # Increased cap to allow more tokens per file
                            )
                        )
                        
                        content = f.get('content')
                        if isinstance(content, list):
                            content = "\n\n".join(content)
                        elif not isinstance(content, str):
                            content = str(content)
                            
                        # Dynamically adjust chunk size and count based on allocated tokens
                        # This allows larger files to contribute more context while preventing
                        # any single file from overwhelming the context window
                        # Use adaptive chunk sizing for better optimization
                        content_length = len(content) if isinstance(content, str) else sum(len(str(c)) for c in content if c)
                        dynamic_chunk_tokens, max_chunks_for_file = adaptive_chunk_sizing(
                            content_length, 
                            allocated_tokens,
                            min_chunks=3,
                            max_chunks=MAX_CONTEXT_CHUNKS_PER_FILE
                        )
                        
                        # Chunk and truncate file content using adaptive sizing
                        from ui.utils import chunk_text_by_tokens
                        file_chunks = chunk_text_by_tokens(content, llm, max_tokens=dynamic_chunk_tokens)
                        
                        # Intelligent chunk selection for large files
                        # Instead of just taking the first N chunks, select diverse chunks
                        # from different parts of the document to provide better coverage
                        if file_chunks:
                            # Select diverse chunks that cover different parts of the document
                            chunks_to_add = select_diverse_chunks(file_chunks, max_chunks_for_file)
                            
                            # If we have a user query, rank chunks by relevance
                            if user_input.strip():
                                ranked_chunks = rank_chunks_by_importance(chunks_to_add, user_input)
                                # Take top ranked chunks but maintain diversity
                                chunks_to_add = ranked_chunks[:max(3, max_chunks_for_file // 2)]
                            
                            file_context_str += "\n\n" + "\n\n---\n\n".join(chunks_to_add)
                            total_file_tokens += sum(calculate_tokens(chunk, llm) for chunk in chunks_to_add)
                        used_files.append(f['name'])

                if file_context_str:
                    rag_context += f"Context from uploaded files ({', '.join(used_files)}):\n{file_context_str}\n\n"

                # 3. Get context from RAG search (chat history and other documents)
                rag_results = st.session_state.rag_store.search(
                    user_input, 
                    top_k=RAG_SEARCH_TOP_K, 
                    similarity_threshold=RAG_SIMILARITY_THRESHOLD
                )
                
                # Filter out results that are identical to the user input
                rag_results = [(text, src, score) for text, src, score in rag_results if text.strip() != user_input.strip()]
                
                if rag_results:
                    rag_context += "Relevant information from knowledge base:\n"
                    rag_context += "\n\n".join([f"(source: {src})\n{text}" for text, src, score in rag_results])

                # 4. Prepare chat history for the prompt
                prompt_chat_history = st.session_state.chat_history[:-1]

                # Handle both string and list content types in chat history
                chat_lines = []
                for m in prompt_chat_history:
                    content = m["content"]
                    if isinstance(content, list):
                        content_str = "\n".join(str(item) for item in content)
                    else:
                        content_str = str(content)
                    chat_lines.append(f'{m["role"]}: {content_str}')
                
                chat_text_for_token_count = "\n".join(chat_lines)
                token_count_history = calculate_tokens(chat_text_for_token_count, llm)

                # 5. Summarize chat history if it's too long
                MAX_HISTORY_TOKENS = int(model_n_ctx * 0.6)
                if token_count_history > MAX_HISTORY_TOKENS:
                    st.info("✿ Chat too long — summarizing to retain memory...")
                    summary = summarize_chat_history_sync(prompt_chat_history, llm, repeat_penalty=repeat_penalty, model_name=st.session_state.model_name)
                    save_chat_summary(st.session_state.session_id, summary)
                    st.session_state.rag_store.add_documents([summary], source="chat_summary")
                    
                    # Prepend summary to the RAG context
                    rag_context = f"Summary of earlier conversation:\n{summary}\n\n{rag_context}"
                    
                    # After summarization, the chat history is replaced by the summary
                    prompt_chat_history = [{"role": "system", "content": "Summary of previous conversation: " + summary}]
                    chat_text_for_token_count = summary
                    token_count_history = calculate_tokens(chat_text_for_token_count, llm)

                # 6. Build prompt and manage token limits
                def build_and_trim_prompt():
                    prompt = build_prompt(
                        user_input=user_input,
                        model_name=st.session_state.model_name,
                        chat_history=prompt_chat_history,
                        rag_context=rag_context,
                        file_context=""
                    )
                    return prompt

                max_prompt_tokens = model_n_ctx - BUFFER_TOKENS - min_response_tokens # Reserve space for response
                prompt = build_and_trim_prompt()
                total_tokens = calculate_tokens(prompt, llm)

                # Trim RAG context if prompt is too long
                while total_tokens > max_prompt_tokens and len(rag_context) > 100:
                    st.warning(f"Token limit exceeded ({total_tokens}/{max_prompt_tokens}). Trimming RAG context...")
                    trim_len = int(len(rag_context) * 0.2)
                    rag_context = rag_context[:-trim_len]
                    prompt = build_and_trim_prompt()
                    total_tokens = calculate_tokens(prompt, llm)

                # If still too long, clear RAG context completely
                if total_tokens > max_prompt_tokens and rag_context:
                    st.warning("Prompt still too long after trimming RAG context. Removing RAG context entirely.")
                    rag_context = ""
                    prompt = build_and_trim_prompt()
                    total_tokens = calculate_tokens(prompt, llm)

                # Trim chat history if prompt is still too long
                while total_tokens > max_prompt_tokens and len(prompt_chat_history) > 1:
                    st.warning(f"Token limit still exceeded ({total_tokens}/{max_prompt_tokens}). Trimming chat history...")
                    prompt_chat_history = prompt_chat_history[1:] # Remove the oldest message
                    prompt = build_and_trim_prompt()
                    total_tokens = calculate_tokens(prompt, llm)

                # If still too long, clear chat history completely
                if total_tokens > max_prompt_tokens and prompt_chat_history:
                    st.warning("Prompt still too long after trimming chat history. Removing chat history entirely.")
                    prompt_chat_history = []
                    prompt = build_and_trim_prompt()
                    total_tokens = calculate_tokens(prompt, llm)

                # Final check
                if total_tokens > max_prompt_tokens:
                    st.error(f"Cannot fit prompt within context window of {model_n_ctx} tokens. Please clear history or reduce file sizes.")
                    st.stop()

                # 7. Display RAG debugger if in debug mode
                if st.session_state.get('debug_mode', False):
                    with st.expander("RAG Debugger: Retrieved Context"):
                        st.text(rag_context if rag_context else "No context retrieved.")

                # 8. Final prompt assignment
                st.session_state.last_prompt = prompt

                if st.session_state.get('debug_mode', False):
                    with st.expander("🧪 Prompt Debugger"):
                        st.text(prompt)

                # Calculate token information for debug display
                prompt_tokens = calculate_tokens(prompt, llm)
                model_n_ctx = MODEL_CONFIGS.get(st.session_state.model_name, {}).get("n_ctx", 4096)
                dynamic_max_tokens = model_n_ctx - prompt_tokens - BUFFER_TOKENS
                response_max_tokens = min(dynamic_max_tokens, 4096)
                
                # Add prompt info when debug mode is active
                if st.session_state.get('debug_mode', False):
                    with st.expander("✿ Prompt Info"):
                        st.markdown(f"**Prompt Token count:** {prompt_tokens}")
                        st.markdown(f"**Model Max Context (n_ctx):** {model_n_ctx}")
                        st.markdown(f"**Buffer Tokens:** {BUFFER_TOKENS}")
                        st.markdown(f"**Dynamic Max Response Tokens:** {dynamic_max_tokens}")
                        st.markdown(f"**Actual Max Response Tokens (after slider cap):** {response_max_tokens}")

                start_time = time.time()
                response_text, full_response = generate_and_process_response(llm, prompt, user_input, temperature, top_p, repeat_penalty, min_response_tokens)
                end_time = time.time()
                response_time = end_time - start_time
                st.caption(f"Response time: {response_time:.2f} seconds")

                st.session_state.chat_history.append({"role": "assistant", "content": response_text, "original_content": full_response, "time": response_time})
                
                metadata = {
                    "response_time": response_time,
                    "prompt_tokens": calculate_tokens(prompt, llm),
                    "temperature": temperature,
                    "top_p": top_p,
                    "repeat_penalty": repeat_penalty
                }
                store_chat_message(st.session_state.session_id, "assistant", response_text, st.session_state.rag_store, full_response, metadata)


    st.sidebar.subheader("✿ Uploaded Files")
    if st.session_state.files:
        # Calculate total tokens used by files when used for context
        total_file_tokens = 0
        file_token_info = []
        
        # First pass: calculate token information for each file
        for f in st.session_state.files:
            if f.get('use_for_context'):
                content = f.get('content')
                if isinstance(content, list):
                    content = "\n\n".join(content)
                elif not isinstance(content, str):
                    content = str(content)
                file_tokens = calculate_tokens(content, llm)
                file_token_info.append((f['name'], file_tokens))
                total_file_tokens += file_tokens
        
        # Show total file context usage
        if total_file_tokens > 0:
            st.sidebar.caption(f"Total file context: ~{total_file_tokens} tokens")
            
        for i, f in enumerate(st.session_state.files):
            token_count = calculate_tokens(f['content'], llm) if isinstance(f['content'], str) else 0
            with st.sidebar.expander(f"{f['name']} ({token_count} tokens)", expanded=False):
                st.text_area("Preview", str(f["content"])[:1000], height=200, key=f"preview_{i}")
                st.session_state.files[i]["use_for_context"] = st.checkbox(
                    "Use for context", value=f["use_for_context"], key=f"use_context_{f['name']}_{i}"
                )
                
                # Show file-specific token information
                if f.get('use_for_context'):
                    # Find the token count for this file
                    file_tokens = next((tokens for name, tokens in file_token_info if name == f['name']), 0)
                    if file_tokens > 0:
                        st.caption(f"Context tokens: ~{file_tokens}")
                        
                        # Show dynamic allocation info
                        if 'available_input_tokens' in locals():
                            # This is an approximation - in a real implementation we'd pass the actual values
                            dynamic_chunk_tokens = min(512, file_tokens // 2)  # Example dynamic sizing
                            max_chunks_for_file = max(1, min(file_tokens // dynamic_chunk_tokens, 10))
                            st.caption(f"Using ~{min(file_tokens, dynamic_chunk_tokens * max_chunks_for_file)} tokens from this file")
    else:
        st.sidebar.info("No files uploaded for this session.")

    if st.button("✿ Summarize this chat", key="summarize_btn"):
        summary = summarize_chat_history_sync(st.session_state.chat_history, llm, repeat_penalty=repeat_penalty, model_name=st.session_state.model_name)
        save_chat_summary(st.session_state.session_id, summary)
        st.session_state.rag_store.add_documents([summary], source="chat_summary")
        st.info(summary)

    if st.button("✿ Regenerate response", key="regenerate_btn") and st.session_state.last_prompt:
        with st.chat_message("assistant"):
            with st.spinner("Regenerating..."):
                last_user_message = ""
                if st.session_state.chat_history and len(st.session_state.chat_history) >= 2:
                    last_user_message = st.session_state.chat_history[-2]["content"]
                
                start_time = time.time()
                response_text, full_response = generate_and_process_response(llm, st.session_state.last_prompt, last_user_message, temperature, top_p, repeat_penalty, min_response_tokens)
                end_time = time.time()
                response_time = end_time - start_time
                st.caption(f"Response time: {response_time:.2f} seconds")

                if st.session_state.chat_history and st.session_state.chat_history[-1]["role"] == "assistant":
                    st.session_state.chat_history[-1]["content"] = response_text
                    st.session_state.chat_history[-1]["original_content"] = full_response
                    st.session_state.chat_history[-1]["time"] = response_time
                    update_last_chat_message(st.session_state.session_id, response_text, full_response)
                else:
                    st.session_state.chat_history.append({"role": "assistant", "content": response_text, "original_content": full_response, "time": response_time})


    if st.session_state.chat_history:
        pdf_data = export_chat_to_pdf(st.session_state.chat_history)
        st.sidebar.download_button(
            "📄 Export Chat as PDF", 
            pdf_data, 
            file_name="chat_history.pdf", 
            mime="application/pdf"
        )
        chat_history_json = json.dumps(st.session_state.chat_history, indent=2)
        st.sidebar.download_button(
            "✿ Download Chat History (JSON)",
            chat_history_json,
            file_name="chat_history.json",
            mime="application/json"
        )
        from ui.csv_exporter import export_chat_to_detailed_csv
        csv_data = export_chat_to_detailed_csv(st.session_state.chat_history)
        st.sidebar.download_button(
            "📊 Export Chat as CSV",
            csv_data,
            file_name="chat_history.csv",
            mime="text/csv"
        )

    st.sidebar.divider()
    debug_mode = st.sidebar.checkbox("🐛 Debug Mode", value=st.session_state.get('debug_mode', False))
    st.session_state.debug_mode = debug_mode


if __name__ == "__main__":
    run_ui()
