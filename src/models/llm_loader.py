"""
LLM Loader

Load language models for inference:
- HuggingFace Transformers (local GPU)
- Ollama (local, easy setup)
- Cloud providers (fallback)
"""

from typing import Optional, Union
from dataclasses import dataclass
import os

# These imports will work after installing requirements
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch


@dataclass
class LLMConfig:
    """Configuration for LLM loading."""
    provider: str = "huggingface"  # "huggingface", "ollama", "openai"
    model_name: str = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4"

    # Generation settings
    max_new_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"

    # Device settings
    device_map: str = "auto"
    torch_dtype: str = "float16"

    @classmethod
    def from_dict(cls, config: dict) -> "LLMConfig":
        """Create LLMConfig from a dictionary (e.g., YAML config)."""
        model_config = config.get("model", {})

        return cls(
            provider=model_config.get("provider", "huggingface"),
            model_name=model_config.get("name", cls.model_name),
            max_new_tokens=model_config.get("max_new_tokens", 2048),
            temperature=model_config.get("temperature", 0.7),
            top_p=model_config.get("top_p", 0.9),
            repetition_penalty=model_config.get("repetition_penalty", 1.1),
            ollama_base_url=model_config.get("ollama_base_url", "http://localhost:11434"),
            device_map=model_config.get("device_map", "auto"),
            torch_dtype=model_config.get("torch_dtype", "float16"),
        )


def load_llm(config: Optional[LLMConfig] = None):
    """
    Load LLM based on configuration.
    
    Args:
        config: LLM configuration. Uses defaults if None.
        
    Returns:
        Tuple of (model, tokenizer) for HuggingFace
        Or client object for Ollama/cloud
        
    Example:
        # Default: Qwen 32B quantized
        model, tokenizer = load_llm()
        
        # Custom config
        config = LLMConfig(
            provider="ollama",
            model_name="qwen2.5:32b"
        )
        client = load_llm(config)
    """
    if config is None:
        config = LLMConfig()
    
    if config.provider == "huggingface":
        return _load_huggingface(config)
    elif config.provider == "ollama":
        return _load_ollama(config)
    elif config.provider == "openai":
        return _load_openai(config)
    else:
        raise ValueError(f"Unknown provider: {config.provider}")


def _load_huggingface(config: LLMConfig):
    """
    Load model from HuggingFace.
    
    For 32GB VRAM, recommended models:
    - Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4 (~18GB)
    - mistralai/Mistral-Small-Instruct-2409 (~14GB quantized)
    - Qwen/Qwen2.5-14B-Instruct (~28GB full precision)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    print(f"Loading {config.model_name}...")
    
    # Determine torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map.get(config.torch_dtype, torch.float16)
    
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map=config.device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    
    print(f"Model loaded on {next(model.parameters()).device}")
    
    return model, tokenizer


def _load_ollama(config: LLMConfig):
    """
    Load model via Ollama.
    
    Make sure Ollama is running: ollama serve
    And model is pulled: ollama pull qwen2.5:32b-instruct-q4_K_M
    """
    import ollama
    
    # Test connection
    try:
        ollama.list()
        print(f"Connected to Ollama at {config.ollama_base_url}")
    except Exception as e:
        raise ConnectionError(
            f"Could not connect to Ollama at {config.ollama_base_url}. "
            f"Make sure Ollama is running: ollama serve\n"
            f"Error: {e}"
        )
    
    # Return a simple wrapper
    class OllamaWrapper:
        def __init__(self, model_name: str, config: LLMConfig):
            self.model_name = model_name
            self.config = config
        
        def generate(self, prompt: str) -> str:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "num_predict": self.config.max_new_tokens,
                }
            )
            return response["response"]
        
        def chat(self, messages: list) -> str:
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "num_predict": self.config.max_new_tokens,
                }
            )
            return response["message"]["content"]
    
    return OllamaWrapper(config.model_name, config)


def _load_openai(config: LLMConfig):
    """Load OpenAI client (for cloud fallback)."""
    from openai import OpenAI
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    return OpenAI(api_key=api_key)


def check_gpu():
    """Check GPU availability and memory."""
    import torch
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - will use CPU (slow!)")
        return None
    
    device_count = torch.cuda.device_count()
    print(f"✓ Found {device_count} GPU(s)")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / 1024**3
        print(f"  GPU {i}: {props.name} ({total_gb:.1f} GB)")
    
    return device_count


if __name__ == "__main__":
    # Quick test
    print("Checking GPU...")
    check_gpu()
    
    print("\nTo load a model, run:")
    print("  from src.models import load_llm")
    print("  model, tokenizer = load_llm()")
