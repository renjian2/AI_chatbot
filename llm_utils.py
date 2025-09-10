import streamlit as st
from typing import Dict, Any, Callable, Union, Generator
from llama_cpp import Llama
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource(show_spinner=True)
def get_llama_model(model_config: Dict[str, Any]) -> Llama:
    """Initialize and cache the Llama model with performance optimizations."""
    try:
        import os
        cpu_count = os.cpu_count() or 4
        optimal_threads = min(cpu_count, 8)
        
        model = Llama(
            model_path=model_config.get("model_path"),
            n_ctx=model_config.get("n_ctx", 4096),
            n_threads=model_config.get("n_threads", optimal_threads),
            n_gpu_layers=model_config.get("n_gpu_layers", -1), # Use -1 for max offloading
            verbose=False,
            use_mlock=False,
            n_batch=2048,
            last_n_tokens_size=64,
            logits_all=False,
            embedding=False,
            use_mmap=True,
            flash_attn=True,
            offload_kqv=True,
        )
        logger.info(f"✅ Llama model loaded successfully with context size: {model.n_ctx()}")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to load Llama model: {str(e)}")
        raise

def get_llm_callable(model_config: Dict[str, Any]) -> Callable:
    """Create a callable wrapper for the LLM with comprehensive model support."""
    model_path = model_config.get("model_path")
    if not model_path:
        raise ValueError("llama_cpp mode requires 'model_path' in model_config.")

    try:
        llama_model = get_llama_model(model_config)
    except Exception as e:
        raise ValueError(f"Failed to initialize LLM: {str(e)}")

    def llm_caller(
        prompt: str, 
        max_tokens: int = 256,  
        temperature: float = 0.7, 
        top_p: float = 0.95, 
        repeat_penalty: float = 1.1,
        stream: bool = True,
        stop: list = None
    ) -> Union[Dict, Generator]:
        """
        Call the LLM with comprehensive support for different model architectures.
        """
        # Determine stop tokens based on model family from config
        if stop is None:
            model_family = model_config.get("family", "default")
            stop_token_map = {
                "llama3": ["<|eot_id|>", "<|end_of_text|>", "\nUser:", "\nAssistant:"],
                "qwen": ["<|im_end|>", "<|endoftext|>", "\nUser:", "\nAssistant:"],
                "mistral": ["</s>", "[/INST]", "\nUser:", "\nAssistant:"],
                "codellama": ["</s>", "<EOT>", "\nUser:", "\nAssistant:"],
                "deepseek": ["<|EOT|>", "<|end_of_sentence|>", "\nUser:", "\nAssistant:"],
                "gemma": ["<end_of_turn>", "<eos>", "\nUser:", "\nAssistant:"],
                "default": ["</s>", "\nUser:", "\nAssistant:"]
            }
            stop_tokens = stop_token_map.get(model_family, stop_token_map["default"])
        else:
            stop_tokens = stop
        
        # Remove duplicate stop tokens
        stop_tokens = list(set(stop_tokens))
        
        try:
            logger.info(f"LLM call - Family: {model_config.get('family')}, Tokens: {max_tokens}, Temp: {temperature}")
            
            result = llama_model(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                stream=stream,
                stop=stop_tokens,
                # Get sampling parameters from config, with sensible defaults
                top_k=model_config.get("top_k", 40),
                mirostat_mode=model_config.get("mirostat_mode", 0),
                tfs_z=1.0,
                mirostat_tau=5.0,
                mirostat_eta=0.1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
            )
            
            if stream:
                return result
            else:
                if isinstance(result, dict) and "choices" in result and result["choices"]:
                    return {"choices": [{"text": result["choices"][0].get("text", "")}]}
                else:
                    logger.warning("LLM returned unexpected response format")
                    return {"choices": [{"text": str(result)}]}
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            raise

    llm_caller.context_window_size = llama_model.n_ctx()
    llm_caller.tokenize = llama_model.tokenize
    llm_caller.detokenize = llama_model.detokenize
    return llm_caller

def get_llm_engine():
    """Get the configured LLM engine."""
    return LLM_ENGINE
