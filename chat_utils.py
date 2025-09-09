import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_prompt(
    user_input: str, 
    model_name: str,
    chat_history: list[dict[str, str]] = None, 
    rag_context: str = None,
    file_context: str = None
) -> str:
    """
    Build a structured prompt for the LLM, adapting to the model's specific format.
    """
    try:
        # Common system prompt content
        system_content = (
            "You are a helpful and concise AI assistant. "
            "Answer the user's question based on the information provided in the 'Reference Information' section and the conversation history. "
            "Your answer should be direct and to the point. Do not repeat the question or the context. "
            "Always use Markdown code blocks (```) for any code snippets. "
            "If the answer is not available in the provided information, state that you do not have enough information to answer."
            "Do not include reasoning or 'thinking steps' in your reply—only provide the final answer."
        )

        # Combine RAG and file context into a single reference block
        reference_info = ""
        if rag_context and rag_context.strip():
            reference_info += f"--- RAG Context ---\n{rag_context.strip()}\n\n"
        if file_context and file_context.strip():
            reference_info += f"--- File Context ---\n{file_context.strip()}\n\n"
        
        if reference_info:
            # Prepend reference info to the system prompt content for better context adherence
            system_content += f"\n\n## Reference Information\n{reference_info}"

        # Model-specific prompt construction
        if "qwen" in model_name.lower():
            # Qwen ChatML format
            prompt = f"<|im_start|>system\n{system_content}<|im_end|>\n"
            if chat_history:
                for msg in chat_history:
                    role = msg["role"]
                    content = msg["content"]
                    prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
            prompt += f"<|im_start|>user\n{user_input.strip()}<|im_end|>\n<|im_start|>assistant\n"
        else:
            # Generic format for other models
            prompt = f"System: {system_content}\n\n"
            prompt += "## Conversation History\n"
            if chat_history:
                for msg in chat_history:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    prompt += f"{role}: {msg['content']}\n"
            prompt += "\n"
            prompt += f"User: {user_input.strip()}\nAssistant:"

        logger.info(f"Built prompt with {len(prompt)} characters for model {model_name}")
        return prompt
        
    except Exception as e:
        logger.error(f"Error building prompt: {e}")
        # Fallback for any error
        return f"User: {user_input or 'Hello'}\nAssistant:"
