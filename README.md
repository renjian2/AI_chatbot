# ✿ AArete Chatbot

This is an advanced chatbot application built with Streamlit, designed to provide intelligent responses leveraging Retrieval Augmented Generation (RAG) and support various file types for contextual understanding.

## Modular Prompt Design Framework

This application now includes a flexible and reusable prompt engineering framework based on modular design principles. See [README_PROMPT_FRAMEWORK.md](README_PROMPT_FRAMEWORK.md) for detailed documentation.

Key features:
- Reusable base prompt template that can be specialized for any use case
- Parameter-driven specialization through role, domain, task, tone, and format
- Industry-specific prompt libraries (healthcare, legal, finance, technology, education)
- Programming support (Python, Java, JavaScript, CSS, HTML, M Query)
- QA engineering (test planning, test cases, bug reporting)
- Database expertise (SQL queries, performance, schema design)
- Advanced Excel (formulas, dashboards, VBA)
- Business intelligence (Power BI, Power Query, DAX)
- Special handling for common greetings to improve user experience
- Easy to extend and customize for new domains and use cases



## ✨ Features

*   **Interactive Chat:** Engage in natural language conversations with the AI.
*   **Session Management:** Create, select, clear, and delete chat sessions to organize your conversations.
*   **Enhanced File Upload:** Upload diverse file types for contextual understanding:
    *   **Text Files:** .txt
    *   **Document Files:** .pdf, .docx
    *   **Programming Files:** .py, .java, .js, .html (and other common source code formats)
    *   **Data Files:** .csv, .xls, .xlsx
    Files are intelligently chunked based on their logical structure (e.g., code blocks, functions, rows for data files) and stored in a vector database for RAG.
*   **Retrieval Augmented Generation (RAG):** The chatbot uses uploaded file content and chat history to provide more accurate and contextually relevant answers.
*   **Chat History Persistence:** Your chat messages are saved to a SQLite database (chat_history.db).
*   **Export Options:**
    *   **PDF Export:** Export your chat history as a well-formatted PDF document.
    *   **JSON Export:** Download your chat history as a JSON file for further analysis or backup.
*   **Response Regeneration:** Regenerate the last AI response if you need an alternative answer.
*   **Model Selection:** Choose from various pre-configured LLM models.

## 🚀 Setup and Installation

To get this application up and running on your local machine, follow these steps:

### Prerequisites

*   Python 3.9+
*   pip (Python package installer)

### 1. Clone the Repository (if applicable)

If you received this project as a zip file, extract it. Otherwise, if it's in a Git repository, clone it:

```bash
git clone <repository_url>
cd chatbotv3.1c-lp
```

### 2. Install Dependencies

Navigate to the project's root directory and install the required Python packages:

```bash
pip install -r requirements.txt
```

### 3. Download and Configure LLM Models

This application uses local LLM models in .gguf format. You need to:

1.  **Download Models:** Obtain your desired .gguf models (e.g., from Hugging Face or other sources).
2.  **Place Models:** Create a directory (e.g., `m` or `models`) and place your .gguf files inside it. The current configuration expects models in E:\XPO_zoho\script\m\ as defined in config.py.
3.  **Configure `config.py`:** Open `config.py` and update the `MODEL_CONFIGS` dictionary with the correct `model_path` for each model you wish to use. Ensure the `n_ctx` (context window size) is correctly set for your chosen model. You can also configure repetition control parameters for each model to control repetition.\n\n    Example `config.py` snippet:
    ```python
    MODEL_CONFIGS = {
        "Llama-3.1-8B": {
            "model_path": "E:\\XPO_zoho\\script\\m\\Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            "n_ctx": 4096,
            "size_category": "medium",
            "repetition_safeguards": True  # Enable additional repetition safeguards
        },
        # Add more models as needed
    }
    
    # RAG Search Parameters
    RAG_SEARCH_TOP_K = 5  # Number of results to retrieve from RAG
    RAG_SIMILARITY_THRESHOLD = 0.3  # Minimum similarity score for RAG results
    ```

## ▶️ How to Run

Once you have installed the dependencies and configured your LLM models, you can run the Streamlit application from your terminal:

```bash
streamlit run app.py
```

This command will open the chatbot UI in your default web browser.

## 💡 Usage

### Chat Interaction

*   Type your message in the input box at the bottom of the screen and press Enter.
*   The AI will generate a response based on its knowledge and any provided context.

### File Upload

*   In the sidebar, locate the "File Upload" section.
*   Click "Browse files" to select one or more supported files (.txt, .pdf, .docx, .py, .java, .js, .html, .csv, .xls, .xlsx).
*   Uploaded files will be processed, chunked, and their content will be used by the RAG system to enhance AI responses.

### Session Management

*   The sidebar provides options to "Select existing session" or "Or create new session".
*   You can clear the RAG index, clear chat history for the current session, or delete an entire session using the buttons in the "Settings" expander.

### Export Chat History

*   Once you have some chat history, the "Export Chat as PDF" and "Download Chat History" (JSON) buttons will appear in the sidebar.
*   Click the respective buttons to download your conversation in the desired format.

### Regenerate Response

*   After the AI has provided a response, a "Regenerate response" button will appear in the main chat area.
*   Click this button to prompt the AI to generate an alternative response to your last query.

## 📁 Project Structure

```
chatbotv3.1c-lp/
├── app.py                  # Main Streamlit application entry point
├── config.py               # Configuration settings (LLM models, file types, etc.)
├── chat_utils.py           # Utilities for chat-related operations
├── db_utils.py             # Database utilities for chat history
├── file_utils.py           # File parsing, chunking, and processing utilities
├── llm_utils.py            # Utilities for interacting with LLM models
├── prompt_builder.py       # Modular prompt design framework
├── prompt_library.py       # Industry-specific prompt libraries
├── rag_utils.py            # RAG (Retrieval Augmented Generation) utilities
├── requirements.txt        # Python dependencies
├── ui/                     # Contains modularized Streamlit UI components
│   ├── __init__.py
│   ├── main.py             # Main UI layout and flow
│   ├── session_manager.py  # Handles Streamlit session state and user sessions
│   ├── file_handler.py     # Manages file uploads and processing
│   ├── chat_display.py     # Displays chat history
│   ├── chat_input.py       # Handles chat input and response generation
│   └── utils.py            # General UI utility functions
├── rag_data/               # Directory for FAISS index and other RAG data
├── chat_history.db         # SQLite database for chat history
├── static/                 # Static assets (e.g., logo.png)
└── ... (other project files)
```

## ⚙️ Technical Section

This section provides a detailed technical overview of the application's architecture, data flow, and core components. It is designed to be easily understood by both human developers and AI assistants.

### High-Level Architecture

The application follows a simple, yet powerful, architecture:

*   **Frontend:** A web-based user interface built with **Streamlit**. All UI components are located in the `ui/` directory.
*   **Backend Logic:** The core application logic is written in **Python**. It handles everything from user input to LLM interaction.
*   **Language Model:** The application uses a local Large Language Model (LLM) via the **`llama-cpp-python`** library. This allows for offline and secure text generation.
*   **Database:** Chat history and session information are stored in a **SQLite** database file named `chat_history.db`.
*   **RAG Pipeline:** The Retrieval-Augmented Generation (RAG) pipeline uses the **`sentence-transformers`** library for creating embeddings and **`faiss`** for efficient similarity search.

### Core Components and Their Responsibilities

*   **`app.py`**: This is the main entry point of the application. It simply calls the `run_ui()` function from `ui/main.py`.
*   **`ui/main.py`**: This is the core of the user interface. It initializes the application, manages the layout, and calls all the other UI components.
*   **`ui/chat_input.py`**: This file contains the logic for handling user input. It takes the user's message, builds the prompt, calls the LLM, and displays the response.
*   **`ui/file_handler.py`**: This file manages the file upload functionality. It processes uploaded files and adds their content to the RAG store.
*   **`llm_utils.py`**: This file is responsible for loading and interacting with the LLM. It contains the `get_llm_callable` function, which returns a callable object for the selected model.
*   **`db_utils.py`**: This file contains all the functions for interacting with the SQLite database. It handles saving and loading chat history, managing sessions, and storing summaries.
*   **`rag_utils.py`**: This file contains the implementation of the RAG pipeline. It includes the `SimpleRAGStore` class, which manages the FAISS index and the document store.
*   **`chat_utils.py`**: This file contains the `build_prompt` function, which is responsible for constructing the final prompt that is sent to the LLM.
*   **`config.py`**: This file contains the configuration for the application, including the paths to the LLM models and other settings.

### Data Flow for a Single Chat Turn

Here is the step-by-step data flow for a typical user interaction:

1.  The user types a message in the chat input box in the Streamlit UI.
2.  The `handle_chat_input` function in `ui/chat_input.py` is called.
3.  The user's message is saved to the chat history in the SQLite database via `db_utils.py`.
4.  The RAG pipeline in `rag_utils.py` is used to search for relevant context from the uploaded files and the chat history.
5.  The `build_prompt` function in `chat_utils.py` is called to construct the final prompt, which includes the system instructions, the RAG context, and the user's message.
6.  The `llm` callable from `llm_utils.py` is invoked with the prompt.
7.  The LLM generates a response, which is streamed back to the UI.
8.  The `st.write_stream` function in `ui/chat_input.py` displays the response to the user in real-time.
9.  The assistant's response is saved to the chat history in the SQLite database.

### RAG Pipeline Explained

The RAG (Retrieval-Augmented Generation) pipeline is what allows the chatbot to answer questions based on your uploaded files. Here's how it works:

1.  **File Processing:** When you upload a file, the `handle_file_upload` function in `ui/file_handler.py` calls `load_and_process_file` from `file_utils.py`.
2.  **Chunking:** The content of the file is split into smaller chunks. This is important for creating effective embeddings.
3.  **Embedding:** Each chunk is converted into a numerical representation (an embedding) using the `all-MiniLM-L6-v2` model from the `sentence-transformers` library.
4.  **Indexing:** The embeddings are stored in a **FAISS** index, which is a library for efficient similarity search. The index is saved to a `.faiss` file in the `rag_data/` directory.
5.  **Retrieval:** When you ask a question, your question is also converted into an embedding. The RAG pipeline then uses the FAISS index to find the chunks from your documents that are most similar to your question.
6.  **Augmentation:** These retrieved chunks are then added to the context of the prompt that is sent to the LLM. This gives the LLM the information it needs to answer your question.

### Database Schema

The application uses a SQLite database named `chat_history.db` to store chat history and summaries. The schema is as follows:

*   **`chat_history` table:**
    *   `session_id` (TEXT): The ID of the chat session.
    *   `role` (TEXT): The role of the message sender (either "user" or "assistant").
    *   `content` (TEXT): The content of the chat message.
*   **`chat_summary` table:**
    *   `session_id` (TEXT, PRIMARY KEY): The ID of the chat session.
    *   `summary` (TEXT): The summary of the chat session.
    *   `updated_at` (TIMESTAMP): The timestamp of when the summary was last updated.

## ⚠️ Troubleshooting

*   **Model Not Loading:** Ensure the `model_path` in `config.py` is correct and the `.gguf` file exists at that location. Check the console for any error messages from `llama-cpp-python`.
*   **Performance Issues:** Processing large files and generating embeddings can be time-consuming. Ensure your system meets the recommended specifications for running LLMs. Consider using smaller models or optimizing chunking parameters if performance is critical.
*   **Buttons Not Visible:** The export buttons and regenerate button are conditionally displayed. Ensure you have chat history for export buttons, and have received an AI response for the regenerate button.

## 💡 Future Enhancements

*   **Advanced RAG and Vector Store Support:**
    *   Integrate with other vector stores like ChromaDB or Pinecone for more flexible and scalable RAG setups.
    *   Implement more advanced chunking and embedding strategies to improve retrieval accuracy.
    *   Add a RAG context visualizer to help debug and understand what information the model is using.

*   **Expanded Model and Provider Support:**
    *   Integrate with other LLM providers and libraries like Hugging Face Transformers, Ollama, and the Gemini API.
    *   Add a model management interface to download and manage models directly from the UI.
    *   Implement a model comparison tool to easily evaluate the outputs of different models side-by-side.

*   **Enhanced User Interface and Experience:**
    *   Implement a more advanced chat interface with support for message editing, deleting, and quoting.
    *   Add a "dark mode" toggle for user comfort.
    *   Implement background processing for large file uploads to prevent the UI from freezing.

*   **Agentic Capabilities and Tool Integration:**
    *   Integrate with external tools and APIs (e.g., web search, code execution) to give the chatbot more powerful capabilities.
    *   Evolve the chatbot into a ReAct (Reasoning and Acting) agent that can perform tasks on the user's behalf.

*   **Evaluation and Testing Framework:**
    *   Integrate evaluation frameworks like RAGAS or BLEU to systematically assess the quality of the chatbot's responses.
    *   Expand the test suite with more comprehensive unit and integration tests.

*   **Security and Access Control:**
    *   Implement user authentication and multi-user support to protect chat histories and manage access.
