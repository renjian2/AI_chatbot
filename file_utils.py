import streamlit as st
from PyPDF2 import PdfReader
import docx
import io
import os
import pandas as pd
from bs4 import BeautifulSoup

from config import SUPPORTED_FILE_TYPES

def _parse_text_file(file_bytes: bytes) -> str:
    # Handle None or empty file bytes
    if not file_bytes:
        return ""
    try:
        # Check file size to prevent memory issues
        file_size_mb = len(file_bytes) / (1024 * 1024)
        if file_size_mb > 100:  # Warn for files larger than 100MB
            import streamlit as st
            st.warning(f"Large file detected: {file_size_mb:.1f}MB. Processing may take some time.")
        
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        # Fallback for binary files
        return str(file_bytes)

def _parse_document_file(file_bytes: bytes, filename: str) -> str:
    if filename.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif filename.endswith(".docx"):
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError(f"Unsupported document file type: {filename}")

def _parse_code_file(file_bytes: bytes, filename: str) -> list:
    """
    Parse code files and preserve logical blocks (functions, classes, etc.)
    Returns a list of code blocks instead of a single string.
    """
    # For code files, we decode as text first
    content = _parse_text_file(file_bytes)
    
    # For Python files, we can use ast to parse and preserve logical blocks
    if filename.endswith('.py'):
        try:
            import ast
            import textwrap
            
            # Parse the code into an AST
            tree = ast.parse(content)
            
            # Extract top-level nodes (functions, classes, etc.)
            blocks = []
            lines = content.split('\
')
            
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                    # Get the start and end line numbers
                    start_line = node.lineno - 1  # AST line numbers are 1-indexed
                    end_line = getattr(node, 'end_lineno', start_line)  # Python 3.8+
                    
                    if end_line is None:
                        # Fallback for older Python versions
                        end_line = start_line
                        # Try to find the end by looking for the next node or end of file
                        for next_node in tree.body:
                            if next_node.lineno > node.lineno:
                                end_line = next_node.lineno - 2
                                break
                        else:
                            end_line = len(lines) - 1
                    
                    # Extract the block with some context
                    start_context = max(0, start_line - 1)
                    block_lines = lines[start_context:end_line]
                    block_content = '\
'.join(block_lines)
                    blocks.append(block_content)
                elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    # Group imports together
                    if blocks and blocks[-1].startswith(('import ', 'from ')):
                        blocks[-1] += '\
' + ast.get_source_segment(content, node)
                    else:
                        blocks.append(ast.get_source_segment(content, node))
                else:
                    # For other top-level statements, get their source
                    try:
                        source = ast.get_source_segment(content, node)
                        if source:
                            blocks.append(source)
                    except Exception:
                        # Fallback if we can't get the source segment
                        pass
            
            # If we couldn't parse into blocks, return the whole content
            if not blocks:
                return content
                
            return blocks
        except Exception as e:
            # If AST parsing fails, fall back to regular text parsing
            import logging
            logging.warning(f"AST parsing failed for {filename}: {e}")
            return content
    
    # For non-Python code files, return as plain text
    return content

def _parse_data_file(file_bytes: bytes, filename: str) -> list:
    """
    Parse data files (CSV, Excel) and chunk by rows with sheet information.
    Returns a list of data chunks with sheet information.
    """
    try:
        chunks = []
        
        if filename.endswith(".csv"):
            # For CSV files, read and chunk by rows
            import io
            import csv
            
            # Decode bytes to string
            content = file_bytes.decode("utf-8", errors="ignore")
            lines = content.splitlines()
            
            if len(lines) <= 100:
                # If less than 100 rows, keep as single chunk
                chunks.append("CSV File: " + filename + "\n" + content)
            else:
                # Chunk by 100 rows
                header = lines[0] if lines else ""
                for i in range(1, len(lines), 100):
                    chunk_lines = lines[i:i+100]
                    chunk_content = "\n".join([header] + chunk_lines)
                    row_info = f" (rows {i+1}-{min(i+100, len(lines))})"
                    chunks.append("CSV File: " + filename + row_info + "\n" + chunk_content)
        
        elif filename.endswith((".xls", ".xlsx")):
            # For Excel files, read and chunk by rows with sheet information
            try:
                import pandas as pd
                import io
                excel_file = io.BytesIO(file_bytes)
                xl = pd.ExcelFile(excel_file)
                
                for sheet_name in xl.sheet_names:
                    df = xl.parse(sheet_name)
                    
                    if len(df) <= 100:
                        # If less than 100 rows, keep as single chunk
                        chunk_content = "Excel File: " + filename + "\nSheet: " + sheet_name + "\n"
                        chunk_content += df.to_csv(index=False)
                        chunks.append(chunk_content)
                    else:
                        # Chunk by 100 rows
                        for i in range(0, len(df), 100):
                            chunk_df = df.iloc[i:i+100]
                            row_info = f" (rows {i+1}-{min(i+100, len(df))})"
                            chunk_content = "Excel File: " + filename + "\nSheet: " + sheet_name + row_info + "\n"
                            chunk_content += chunk_df.to_csv(index=False)
                            chunks.append(chunk_content)
            except ImportError:
                # If pandas is not available, fall back to regular text parsing
                import logging
                logging.warning("Pandas not available for Excel parsing. Falling back to text parsing.")
                return _parse_text_file(file_bytes)
        
        # If we have chunks, return them; otherwise return the content as text
        if chunks:
            return chunks
        else:
            # Fallback to regular text parsing
            return _parse_text_file(file_bytes)
            
    except Exception as e:
        # If parsing fails, fall back to regular text parsing
        import logging
        logging.warning(f"Data file parsing failed for {filename}: {e}")
        return _parse_text_file(file_bytes)

@st.cache_data(show_spinner=False)
def _parse_file_with_cache(file_bytes: bytes, filename: str) -> str:
    """Cached version of file parsing."""
    file_ext = os.path.splitext(filename)[1].lower()[1:]  # Remove the dot
    file_type = SUPPORTED_FILE_TYPES.get(file_ext, "text")
    
    if file_type == "text":
        return _parse_text_file(file_bytes)
    elif file_type == "document":
        return _parse_document_file(file_bytes, filename)
    elif file_type == "code":
        result = _parse_code_file(file_bytes, filename)
        # If result is a list of blocks, join them with double newlines
        if isinstance(result, list):
            return "\n\n".join(str(block) for block in result)
        return result
    elif file_type == "data":
        result = _parse_data_file(file_bytes, filename)
        # If result is a list of chunks, join them with double newlines
        if isinstance(result, list):
            return "\n\n".join(str(chunk) for chunk in result)
        return result
    else:
        # Default to text parsing
        return _parse_text_file(file_bytes)

@st.cache_data(show_spinner=False)
def load_and_process_file(file_bytes: bytes, filename: str, _llm, max_chunk_tokens=1500):
    """Load and process a file, returning chunks of text."""
    import logging
    import streamlit as st
    
    try:
        logging.info(f"Processing file: {filename} (size: {len(file_bytes)} bytes)")
        content = _parse_file_with_cache(file_bytes, filename)
        if not content:
            logging.warning(f"File {filename} has no content after parsing")
            return []
        
        # Log content size
        content_size = len(content)
        logging.info(f"Parsed content size for {filename}: {content_size} characters")
        
        # Determine if this is a code file for special handling
        file_ext = os.path.splitext(filename)[1].lower()[1:]  # Remove the dot
        file_type = SUPPORTED_FILE_TYPES.get(file_ext, "text")
        is_code_file = file_type == "code"
        
        # Chunk the content if it's too long
        chunks = chunk_text(content, _llm, max_chunk_tokens, is_code_file=is_code_file)
        logging.info(f"File {filename} split into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        st.error(f"Error processing file {filename}: {str(e)}")
        # Log the full traceback for debugging
        import traceback
        logging.error(f"Error processing file {filename}: {traceback.format_exc()}")
        st.sidebar.error(f"Full traceback: {traceback.format_exc()}")
        return []

def truncate_text_by_tokens(text: str, llm, max_in_tokens=2000):
    """Truncate text to a maximum number of tokens."""
    if not text:
        return ""
    
    # Estimate token count (rough approximation)
    estimated_tokens = len(text.split())  # Simple word-based estimation
    
    if estimated_tokens <= max_in_tokens:
        return text
    
    # If we have access to the actual tokenizer, use it
    if hasattr(llm, 'tokenizer'):
        try:
            tokens = llm.tokenizer(text.encode('utf-8'))
            if len(tokens) <= max_in_tokens:
                return text
            # Truncate to max_in_tokens
            truncated_tokens = tokens[:max_in_tokens]
            if hasattr(llm, 'detokenize'):
                return llm.detokenize(truncated_tokens).decode('utf-8', errors='ignore')
            else:
                # Fallback: join the tokens as strings
                return " ".join(str(token) for token in truncated_tokens)
        except Exception:
            # Fallback to word-based truncation
            words = text.split()
            return " ".join(words[:max_in_tokens])
    else:
        # Fallback to word-based truncation
        words = text.split()
        return " ".join(words[:max_in_tokens])

def trim_context_to_fit(context: str, llm, available_tokens: int):
    """Trim context to fit within available tokens."""
    return truncate_text_by_tokens(context, llm, available_tokens)

def chunk_text(text: str, llm, max_tokens: int, overlap: int = 200, is_code_file=False):
    """
    Split text into chunks of approximately max_tokens, with a specified overlap.
    Handles token-based chunking if a tokenizer is available, otherwise falls back
    to character-based chunking with word boundary awareness.
    For code files, preserves logical blocks when possible.
    """
    if not text:
        return []

    # Ensure text is a string
    if not isinstance(text, str):
        text = " ".join(str(item) for item in text) if isinstance(text, list) else str(text)

    # For code files, try to preserve logical blocks first
    if is_code_file:
        code_chunks = _chunk_code_file_preserve_blocks(text, max_tokens)
        if code_chunks:
            return code_chunks

    # 1. Token-based chunking (preferred method)
    if hasattr(llm, 'tokenizer') and hasattr(llm, 'detokenize'):
        try:
            tokens = llm.tokenizer(text.encode('utf-8', errors='ignore'))
            chunks = []
            for i in range(0, len(tokens), max_tokens - overlap):
                chunk_tokens = tokens[i:i + max_tokens]
                if not chunk_tokens:
                    continue
                chunk_text = llm.detokenize(chunk_tokens).decode('utf-8', errors='ignore')
                chunks.append(chunk_text)
            return chunks
        except Exception as e:
            import logging
            logging.warning(f"Token-based chunking failed: {e}. Falling back to character-based chunking.")

    # 2. Fallback to character-based chunking
    # This method is more robust for texts without clear tokenizers or with very long words/lines.
    # We approximate token count by assuming 1 token ~ 4 characters.
    max_chars = max_tokens * 4
    char_overlap = overlap * 4
    
    chunks = []
    start_pos = 0
    while start_pos < len(text):
        end_pos = start_pos + max_chars
        
        # If we are near the end, just take the rest of the text
        if end_pos >= len(text):
            chunk = text[start_pos:]
            if chunk.strip():
                chunks.append(chunk)
            break

        # For code files, try to preserve logical blocks
        # Look for natural break points that coincide with code structure
        break_pos = -1
        
        # Try to find a good break point that preserves code blocks
        # Look for double newlines (common between functions/classes)
        if is_code_file:
            break_pos = text.rfind('\n\n', start_pos, end_pos)
        
        # If no double newline found (or not code), look for single newlines
        if break_pos == -1:
            break_pos = text.rfind('\n', start_pos, end_pos)
        
        # If no newline found, look for spaces
        if break_pos == -1:
            break_pos = text.rfind(' ', start_pos, end_pos)
        
        # If no natural break is found, take the hard cut at max_chars
        if break_pos == -1 or break_pos <= start_pos:
            break_pos = end_pos

        chunk = text[start_pos:break_pos].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start_pos for the next chunk, considering the overlap
        # The next chunk should start `char_overlap` characters before the end of the current chunk.
        start_pos = break_pos - char_overlap
        # Ensure start_pos doesn't go backward or beyond the current break_pos
        if start_pos < 0:
            start_pos = 0
        # If the overlap makes start_pos equal or greater than break_pos,
        # it means the chunk was too small to have a meaningful overlap,
        # so we just move to the end of the current chunk to avoid infinite loops.
        if start_pos >= break_pos:
            start_pos = break_pos + 1


    return chunks


def _chunk_code_file_preserve_blocks(text: str, max_tokens: int) -> list:
    """
    Try to preserve logical blocks (functions, classes) when chunking code files.
    """
    try:
        # Split text into lines
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        # Simple token estimation for code
        def estimate_tokens(line):
            return len(line.split())  # Rough estimate
        
        i = 0
        while i < len(lines):
            line = lines[i]
            line_tokens = estimate_tokens(line)
            
            # If adding this line would exceed the token limit
            if current_tokens + line_tokens > max_tokens and current_chunk:
                # Save current chunk
                chunks.append('\n'.join(current_chunk))
                current_chunk = []
                current_tokens = 0
                # Continue with the same line in the next iteration
                continue
            else:
                # Add line to current chunk
                current_chunk.append(line)
                current_tokens += line_tokens
                i += 1
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            
        return chunks
    except Exception as e:
        import logging
        logging.warning(f"Code block preservation failed: {e}. Falling back to regular chunking.")
        return []


def calculate_tokens(text, llm):
    """
    Calculates the number of tokens in a given text using the model's tokenizer.
    """
    # Handle different data types
    if not isinstance(text, str):
        if isinstance(text, list):
            # If it's a list, join the elements
            text = " ".join(str(item) for item in text)
        else:
            # Convert to string
            text = str(text)
    
    if hasattr(llm, 'tokenizer'):
        try:
            # The tokenizer returns a list of token IDs
            return len(llm.tokenizer(text.encode('utf-8')))
        except Exception as e:
            # Fallback if tokenizer fails
            return len(text.split())
    else:
        # Fallback for models without a direct tokenizer method
        return len(text.split())

# Placeholder for summarization, implement using your llm callable
def summarize_text(text: str, llm, max_tokens=256):
    prompt = f"Summarize the following text:\n\n{text}"
    response = llm(prompt, max_tokens=max_tokens, temperature=0.3, repeat_penalty=1.1)
    # Handle both streaming and non-streaming responses
    if isinstance(response, dict) and "choices" in response:
        return response["choices"][0]["text"].strip()
    else:
        # For streaming responses, we need to collect all chunks
        summary = ""
        for chunk in response:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                text = chunk["choices"][0].get("text", "")
                summary += text
        return summary.strip()