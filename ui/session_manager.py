import streamlit as st
from db_utils import (
    init_db,
    save_chat_summary,
    load_chat_summary,
    save_chat_message,
    load_chat_history,
    clear_chat_history,
    list_sessions,
    delete_session,
)
from rag_utils import SimpleRAGStore
from ui.utils import get_embedding_model # Assuming get_embedding_model is in ui.utils

def initialize_session_state(embedding_model):
    """Initializes session state variables if they don't exist."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = "guest"
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "files" not in st.session_state:
        st.session_state.files = []
    if "last_prompt" not in st.session_state:
        st.session_state.last_prompt = None
    if "rag_store" not in st.session_state:
        st.session_state.rag_store = SimpleRAGStore(session_id=st.session_state.session_id, embedding_model=embedding_model)
    if "rag_session_id" not in st.session_state:
        st.session_state.rag_session_id = st.session_state.session_id
    if "current_session" not in st.session_state:
        st.session_state.current_session = st.session_state.session_id
    if "remove_think_step" not in st.session_state:
        st.session_state.remove_think_step = True

def manage_sessions(embedding_model):
    """Handles session selection and creation in the sidebar."""
    st.sidebar.header(" Session Management")
    sessions = sorted(set(list_sessions()))
    
    # Add a "New Session" option
    session_options = [""] + sessions
    
    # Find the index of the current session
    try:
        current_session_index = session_options.index(st.session_state.get("session_id", ""))
    except ValueError:
        current_session_index = 0

    selected_session_ui = st.sidebar.selectbox(
        "Select existing session",
        options=session_options,
        index=current_session_index,
        key="selected_session_ui"
    )

    new_session_name_ui = st.sidebar.text_input("Or create new session", value="", key="new_session_name_ui").strip().lower()

    # Determine the target session ID
    target_session_id = new_session_name_ui if new_session_name_ui else selected_session_ui if selected_session_ui else "guest"

    # Only reinitialize if the session actually changes
    if target_session_id != st.session_state.get("current_session"):
        st.session_state.session_id = target_session_id
        st.session_state.current_session = target_session_id
        st.session_state.chat_history = load_chat_history(st.session_state.session_id)
        # Don't clear files when switching sessions - they should persist per session
        # st.session_state.files = []
        st.session_state.last_prompt = None
        st.session_state.regenerate_request = None
        # Update RAG session ID before reinitializing RAG store
        st.session_state.rag_session_id = st.session_state.session_id
        # Reinitialize RAG store with correct session ID
        st.session_state.rag_store = SimpleRAGStore(session_id=st.session_state.session_id, embedding_model=embedding_model)
        st.rerun()
