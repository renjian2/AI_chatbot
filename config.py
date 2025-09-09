import os

LLM_ENGINE = "llama.cpp"  # or "ollama"

# Base directory for models
MODEL_BASE_DIR = "/home/administrator/chatbot8212025v1/model"

# Comprehensive model configurations
MODEL_CONFIGS = {
    "Meta-Llama-3.1-8B-Instruct": {
        "model_path": os.path.join(MODEL_BASE_DIR, "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
        "family": "llama3",
        "n_ctx": 4096,
        "repeat_penalty": 1.7, # Restored high value to prevent repetition
        "quantization": "Q4_K_M",
        "n_gpu_layers": -1,
        "top_k": 40,
        "mirostat_mode": 0,
    },
    "Meta-Llama-3.2-3B-Instruct": {
        "model_path": os.path.join(MODEL_BASE_DIR, "Llama-3.2-3B-Instruct-Q4_K_M.gguf"),
        "family": "llama3",
        "n_ctx": 4096,
        "repeat_penalty": 1.7, # Restored high value to prevent repetition
        "quantization": "Q4_K_M",
        "n_gpu_layers": -1,
        "top_k": 40,
        "mirostat_mode": 0,
    },
    "Mistral-7B-Instruct-v0.3": {
        "model_path": os.path.join(MODEL_BASE_DIR, "Mistral-7B-v0.3.Q4_K_M.gguf"),
        "family": "mistral",
        "n_ctx": 4096,
        "repeat_penalty": 1.7,
        "quantization": "Q4_K_M",
        "n_gpu_layers": -1,
        "top_k": 40,
        "mirostat_mode": 0,
    },
    "CodeLlama-7b-Instruct-hf": {
        "model_path": os.path.join(MODEL_BASE_DIR, "codellama-7b.Q4_K_M.gguf"),
        "family": "codellama",
        "n_ctx": 4096,
        "repeat_penalty": 1.7,
        "quantization": "Q4_K_M",
        "n_gpu_layers": -1,
        "top_k": 40,
        "mirostat_mode": 0,
    },
    "CodeLlama-13b-Instruct-hf": {
        "model_path": os.path.join(MODEL_BASE_DIR, "codellama-13b.Q4_K_M.gguf"),
        "family": "codellama",
        "n_ctx": 4096,
        "repeat_penalty": 1.7,
        "quantization": "Q4_K_M",
        "n_gpu_layers": -1,
        "top_k": 40,
        "mirostat_mode": 0,
    },
    "gemma-3-1b-it": {
        "model_path": os.path.join(MODEL_BASE_DIR, "gemma-3-1b-it-BF16.gguf"),
        "family": "gemma",
        "n_ctx": 4096,
        "repeat_penalty": 1.7,
        "quantization": "BF16 (Note: Not a standard GGUF quant)",
        "n_gpu_layers": -1,
        "top_k": 40,
        "mirostat_mode": 0,
    },
    "Qwen2-7B-Instruct": { # Standardized name
        "model_path": os.path.join(MODEL_BASE_DIR, "Qwen3-8B-Q5_K_M.gguf"), # Note: Path says Qwen3-8B
        "family": "qwen",
        "n_ctx": 4096,
        "repeat_penalty": 1.7,
        "quantization": "Q5_K_M",
        "n_gpu_layers": -1,
        "top_k": 40,
        "mirostat_mode": 0,
    },
    "DeepSeek-Coder-V2-Lite-Instruct": {
        "model_path": os.path.join(MODEL_BASE_DIR, "DeepSeek-Coder-V2-Lite-Base-Q4_K_M.gguf"),
        "family": "deepseek",
        "n_ctx": 4096,
        "repeat_penalty": 1.7,
        "quantization": "Q4_K_M",
        "n_gpu_layers": -1,
        "top_k": 40,
        "mirostat_mode": 0,
    },
    "Qwen2-1.5B-Instruct": { # Standardized name
        "model_path": os.path.join(MODEL_BASE_DIR, "Qwen3-1.7B-Q8_0.gguf"), # Note: Path says Qwen3-1.7B
        "family": "qwen",
        "n_ctx": 4096,
        "repeat_penalty": 1.7,
        "quantization": "Q8_0",
        "n_gpu_layers": -1,
        "top_k": 40,
        "mirostat_mode": 0,
    },"Qwen3-4B-Instruct": { # Standardized name
        "model_path": os.path.join(MODEL_BASE_DIR, "Qwen3-4B-Q4_K_M.gguf"), # Note: Path says Qwen3-1.7B
        "family": "qwen",
        "n_ctx": 4096,
        "repeat_penalty": 1.7,
        "quantization": "Q8_0",
        "n_gpu_layers": -1,
        "top_k": 40,
        "mirostat_mode": 0,
    },
}

MAX_TOTAL_TOKENS = 4096
BUFFER_TOKENS = 512

# RAG Search Parameters
RAG_SEARCH_TOP_K = 5
RAG_SIMILARITY_THRESHOLD = 0.3

MAX_FILE_CHUNK_TOKENS = 256


SUPPORTED_FILE_TYPES = {
    "txt": "text",
    "pdf": "document",
    "docx": "document",
    "py": "code",
    "java": "code",
    "js": "code",
    "html": "code",
    "csv": "data",
    "xls": "data",
    "xlsx": "data",
}