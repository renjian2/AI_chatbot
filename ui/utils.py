"""
Simplified UI Utilities Module
"""

import streamlit as st
import base64
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from file_utils import calculate_tokens

# Cache for expensive operations
@st.cache_resource
def get_embedding_model():
    """Loads and caches the SentenceTransformer model."""
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def cached_get_base64_image(image_path):
    """Cached version of get_base64_encoded_image to avoid repeated file reads."""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return encoded_string
    except FileNotFoundError:
        return None
    except Exception as e:
        # Log the error for debugging
        print(f"Error reading image file {image_path}: {e}")
        return None

def clean_model_output(text):
    """Removes any extra newlines and whitespace from the model output and ensures proper code formatting."""
    # Handle None or empty text
    if not text:
        return ""
        
    # Remove leading and trailing whitespace
    text = text.strip()
    # Replace multiple newlines with a single newline
    import re
    text = re.sub(r'\n+', '\n', text)
    
    return text

def clean_model_response(text: str, model_name: str) -> str:
    """Cleans model response output."""
    # Simple cleaning
    return text.strip()

def remove_duplicate_sentences(context: str, user_input: str) -> str:
    """Removes sentences from the context that are duplicates of the user input."""
    # Simple implementation - in a real app, you might want more sophisticated deduplication
    return context

def store_chat_message(session_id, role, content, rag_store, original_content=None, metadata=None):
    """Stores a chat message in the database and RAG store."""
    try:
        # Save to database
        from db_utils import save_chat_message
        save_chat_message(session_id, role, content, original_content, metadata)
        
        # Save to RAG store for long-term recall
        if rag_store:
            rag_store.add_chat_message(content, role)
    except Exception as e:
        print(f"Error storing chat message: {e}")

def store_chat_messages_batch(session_id, messages, rag_store):
    """Stores multiple chat messages in batch."""
    try:
        # Save to database
        from db_utils import save_chat_messages_batch
        save_chat_messages_batch(session_id, messages)
        
        # Save to RAG store for long-term recall
        if rag_store:
            for msg in messages:
                if isinstance(msg, dict):
                    rag_store.add_chat_message(msg.get("content", ""), msg.get("role", ""))
                elif isinstance(msg, tuple) and len(msg) >= 3:
                    # Assuming tuple format: (session_id, role, content, ...)
                    rag_store.add_chat_message(msg[2], msg[1])
    except Exception as e:
        print(f"Error storing chat messages batch: {e}")

def build_summary_prompt(chat_history_text: str) -> str:
    """Builds a prompt for summarizing chat history."""
    return f"Please summarize the following conversation:\n\n{chat_history_text}"

def summarize_chat_history_sync(chat_history, llm, repeat_penalty=1.1, model_name="default"):
    """Synchronizes chat history summarization."""
    # Placeholder implementation
    return "Chat history summary"

def post_process_response(response_text: str, prompt: str) -> str:
    """Post-processes the response to clean up any duplication."""
    # Simple cleaning
    if not response_text:
        return ""
    
    # Remove leading and trailing whitespace
    response_text = response_text.strip()
    
    # Aggressively remove consecutive duplicate lines
    lines = response_text.split('\n')
    if len(lines) > 1:
        # Check if we have repetitive lines
        unique_lines = []
        for line in lines:
            stripped_line = line.strip()
            # Skip empty lines
            if not stripped_line:
                # Only add empty lines if we don't already have too many in a row
                if len(unique_lines) == 0 or unique_lines[-1].strip() != "":
                    unique_lines.append(line)
            # If this line is not a duplicate of the previous non-empty line, add it
            elif not unique_lines or stripped_line != unique_lines[-1].strip():
                unique_lines.append(line)
            # If this is a duplicate, check if we've seen too many duplicates
            elif len(unique_lines) > 2 and stripped_line == unique_lines[-2].strip():
                # This is a repeated pattern, stop adding lines
                # Add a note that content was truncated
                if not unique_lines[-1].endswith("[further repetitions truncated]"):
                    unique_lines.append("[further repetitions truncated]")
                break
        
        # Reconstruct the response with unique lines only
        response_text = '\n'.join(unique_lines)
    
    # Additional aggressive cleaning for repetitive statements
    # Split by punctuation that typically ends statements
    import re
    statements = re.split(r'[.;]', response_text)
    if len(statements) > 5:  # More than 5 statements might be repetitive
        # Check if statements are identical
        cleaned_statements = []
        for stmt in statements:
            stripped_stmt = stmt.strip()
            if stripped_stmt and stripped_stmt not in cleaned_statements:
                cleaned_statements.append(stripped_stmt)
        
        # If we reduced the number of statements significantly, use the cleaned version
        if len(cleaned_statements) < len(statements) and len(cleaned_statements) > 0:
            response_text = ';\n'.join(cleaned_statements) + ';'
            if len(statements) - len(cleaned_statements) > 2:
                response_text += "\n\n[further identical statements truncated]"
    
    return response_text.strip()

def enforce_single_response(response_text: str, content_type: str = "generic") -> str:
    """Ensures only a single, clean response is returned."""
    # Simple implementation
    return response_text

def extract_clean_code_block(response_text: str, language: str = "python") -> str:
    """Extracts clean code block for programming languages."""
    # Simple implementation
    return response_text

def extract_json_block(response_text: str) -> str:
    """Extracts clean JSON block."""
    # Simple implementation
    return response_text

def is_technical_content_efficient(user_input: str) -> bool:
    """Efficiently determines if content is technical."""
    technical_keywords = [
        "code", "python", "java", "javascript", "c++", "c#", "html", "css",
        "sql", "database", "query", "select", "insert", "update", "delete", 
        "mysql", "postgresql", "oracle", "mssql",
        "function", "class", "method", "variable", "array", "list", "object",
        "algorithm", "data structure", "api", "json", "xml", "http", "rest",
        "procedure", "trigger", "index", "view", "table", "column", "schema",
        "email", "validation", "test", "strategy", "explain", "technical",
        "steps", "modeling", "error", "exception", "optimization", "performance",
        "deployment", "debug", "refactor", "design pattern", "architecture",
        "tcp", "udp", "protocol", "network", "server", "client", "api", "sdk"
    ]
    
    user_input_lower = user_input.lower()
    return any(keyword in user_input_lower for keyword in technical_keywords)

def cached_load_and_process_file(file_obj, progress_callback=None):
    """Caches and processes file content."""
    # Placeholder implementation
    return file_obj.get("content", "")



def validate_response_completeness(response_text, original_length=None):
    """Validates if a response is complete."""
    if not response_text:
        return False, "Response is empty", {"issues": ["Empty response"], "suggestions": ["Try regenerating the response"]}
    
    # Basic checks for completeness
    issues = []
    suggestions = []
    
    if len(response_text.strip()) < 10:
        issues.append("Response is very short")
        suggestions.append("Try regenerating or rephrasing your query")
    
    if not response_text.strip().endswith(('.', '!', '?', '"', "'", '`')):
        issues.append("Response may be cut off")
        suggestions.append("Response doesn't end with proper punctuation")
    
    is_complete = len(issues) == 0
    message = "Response appears complete" if is_complete else "Response may be incomplete"
    
    return is_complete, message, {"issues": issues, "suggestions": suggestions}

def remove_qwen_think_block(text: str) -> str:
    """
    Removes the <think>...</think> block from Qwen model responses.
    """
    if "<think>" in text and "</think>" in text:
        import re
        return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()
    return text

def chunk_text_by_tokens(text, llm, max_tokens=1024):
    """
    Splits text into chunks, each <= max_tokens, while respecting paragraph and sentence boundaries.
    """
    if not text:
        return []

    # First, split the text into paragraphs
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        # If a paragraph itself is larger than max_tokens, split it by sentences
        if calculate_tokens(para, llm) > max_tokens:
            sentences = para.split('. ')
            for sent in sentences:
                # If a sentence is larger than max_tokens, we have to split it by words
                if calculate_tokens(sent, llm) > max_tokens:
                    words = sent.split(' ')
                    for word in words:
                        if calculate_tokens(current_chunk + word, llm) > max_tokens:
                            chunks.append(current_chunk.strip())
                            current_chunk = ""
                        current_chunk += word + " "
                # otherwise, add sentence to chunk
                else:
                    if calculate_tokens(current_chunk + sent, llm) > max_tokens:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    current_chunk += sent + ". "
        # otherwise, add paragraph to chunk
        else:
            if calculate_tokens(current_chunk + para, llm) > max_tokens:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            current_chunk += para + "\n\n"

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Final check to ensure no chunk is over the limit.
    # This can happen with the last chunk.
    final_chunks = []
    for chunk in chunks:
        if calculate_tokens(chunk, llm) > max_tokens:
            # If a chunk is still too large, split it by words as a last resort.
            words = chunk.split(' ')
            new_chunk = ""
            for word in words:
                if calculate_tokens(new_chunk + word, llm) > max_tokens:
                    final_chunks.append(new_chunk.strip())
                    new_chunk = ""
                new_chunk += word + " "
            if new_chunk:
                final_chunks.append(new_chunk.strip())
        else:
            final_chunks.append(chunk)
            
    return final_chunks


# Export all functions for backward compatibility
__all__ = [
    'get_embedding_model',
    'cached_get_base64_image',
    'clean_model_output',
    'clean_model_response',
    'remove_duplicate_sentences',
    'store_chat_message',
    'store_chat_messages_batch',
    'build_summary_prompt',
    'summarize_chat_history_sync',
    'post_process_response',
    'enforce_single_response',
    'extract_clean_code_block',
    'extract_json_block',
    'is_technical_content_efficient',
    'cached_load_and_process_file',
    'validate_response_completeness',
    'remove_qwen_think_block',
    'chunk_text_by_tokens'
]