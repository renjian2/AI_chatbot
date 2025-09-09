# db_utils.py
import sqlite3
import os
import streamlit as st
import json

DB_FILE = "chat_history.db"

@st.cache_resource
def get_db_connection():
    """Creates and returns a cached database connection."""
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    # Enable WAL mode for better concurrency
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_db():
    """Initializes the database tables."""
    conn = get_db_connection()
    c = conn.cursor()
    # Create chat_history table with additional fields for original content
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            session_id TEXT,
            role TEXT,
            content TEXT,
            original_content TEXT,
            metadata TEXT DEFAULT '{}'
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_summary (
            session_id TEXT PRIMARY KEY,
            summary TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Check if the original_content column exists, if not add it
    try:
        c.execute("ALTER TABLE chat_history ADD COLUMN original_content TEXT")
    except sqlite3.OperationalError:
        # Column already exists
        pass
    # Check if the metadata column exists, if not add it
    try:
        c.execute("ALTER TABLE chat_history ADD COLUMN metadata TEXT DEFAULT '{}'")
    except sqlite3.OperationalError:
        # Column already exists
        pass
    conn.commit()

def save_chat_summary(session_id, summary):
    """Saves or updates a summary for a given session ID."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO chat_summary (session_id, summary, updated_at)
        VALUES (?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(session_id) DO UPDATE SET
            summary=excluded.summary,
            updated_at=CURRENT_TIMESTAMP
    """, (session_id, summary))
    conn.commit()

def load_chat_summary(session_id):
    """Loads the summary for a given session ID."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT summary FROM chat_summary WHERE session_id = ?", (session_id,))
    row = c.fetchone()
    return row[0] if row else ""

def save_chat_message(session_id, role, content, original_content=None, metadata=None):
    """Saves a single chat message with optional original content and metadata."""
    conn = get_db_connection()
    c = conn.cursor()
    metadata_json = json.dumps(metadata) if metadata else "{}"
    c.execute(
        "INSERT INTO chat_history (session_id, role, content, original_content, metadata) VALUES (?, ?, ?, ?, ?)",
        (session_id, role, content, original_content, metadata_json)
    )
    conn.commit()

def save_chat_messages_batch(session_id, messages):
    """Saves multiple chat messages in a batch."""
    if not messages:
        return
    conn = get_db_connection()
    c = conn.cursor()
    # Handle both old format and new format messages
    formatted_messages = []
    for msg in messages:
        if isinstance(msg, tuple) and len(msg) == 5:
            # Already in new format (session_id, role, content, original_content, metadata)
            formatted_messages.append(msg)
        else:
            # Old format, convert to new format
            metadata_json = json.dumps(msg.get("metadata", {})) if isinstance(msg, dict) else "{}"
            formatted_messages.append((
                session_id, 
                msg["role"] if isinstance(msg, dict) else msg[1], 
                msg["content"] if isinstance(msg, dict) else msg[2], 
                msg.get("original_content") if isinstance(msg, dict) else None,
                metadata_json
            ))
    
    c.executemany(
        "INSERT INTO chat_history (session_id, role, content, original_content, metadata) VALUES (?, ?, ?, ?, ?)",
        formatted_messages
    )
    conn.commit()

def load_chat_history(session_id):
    """Loads all chat messages for a given session ID."""
    conn = get_db_connection()
    c = conn.cursor()
    # Load both content and original_content if available
    c.execute("SELECT role, content, original_content, metadata FROM chat_history WHERE session_id = ?", (session_id,))
    rows = c.fetchall()
    history = []
    for row in rows:
        msg = {"role": row[0], "content": row[1]}
        if row[2] is not None:  # original_content
            msg["original_content"] = row[2]
        try:
            if row[3] and row[3] != "{}":  # metadata
                msg["metadata"] = json.loads(row[3])
        except:
            pass
        history.append(msg)
    return history

def clear_chat_history(session_id):
    """Clears all chat messages for a given session ID."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
    conn.commit()

def list_sessions():
    """Lists all unique session IDs."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT DISTINCT session_id FROM chat_history")
    return [row[0] for row in c.fetchall()]

def delete_session(session_id):
    """Deletes all chat history and RAG index files for a given session."""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
    c.execute("DELETE FROM chat_summary WHERE session_id = ?", (session_id,))
    conn.commit()

    # Delete RAG index files
    rag_base = f"rag_data/{session_id}"
    for ext in [".faiss", "_meta.pkl", "_embeddings.npy"]:
        try:
            os.remove(rag_base + ext)
        except FileNotFoundError:
            pass

def update_last_chat_message(session_id, new_content, original_content=None):
    """Updates the content of the last assistant message for a given session ID."""
    conn = get_db_connection()
    c = conn.cursor()
    # Find the rowid of the last assistant message for the given session
    c.execute("""
        SELECT rowid FROM chat_history
        WHERE session_id = ? AND role = 'assistant'
        ORDER BY rowid DESC
        LIMIT 1
    """, (session_id,))
    last_message = c.fetchone()
    if last_message:
        last_message_rowid = last_message[0]
        if original_content is not None:
            c.execute("""
                UPDATE chat_history
                SET content = ?, original_content = ?
                WHERE rowid = ?
            """, (new_content, original_content, last_message_rowid))
        else:
            c.execute("""
                UPDATE chat_history
                SET content = ?
                WHERE rowid = ?
            """, (new_content, last_message_rowid))
        conn.commit()


