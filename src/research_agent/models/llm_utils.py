"""
LLM Utilities for Research Agent

Provides LLM integration with multiple backends:
- Ollama (local, recommended)
- HuggingFace Transformers (fallback)
"""

import logging
import requests
import json

logger = logging.getLogger(__name__)


class VRAMConstraintError(Exception):
    """Custom exception for VRAM limitations"""
    pass


class OllamaUnavailableError(Exception):
    """Exception for when Ollama is not available"""
    pass


def get_vram_info():
    """Get detailed VRAM information"""
    try:
        import torch
    except ImportError:
        return {"available": 0, "total": 0, "device": None}

    if not torch.cuda.is_available():
        return {"available": 0, "total": 0, "device": None}

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    allocated_gb = torch.cuda.memory_allocated() / 1024**3
    reserved_gb = torch.cuda.memory_reserved() / 1024**3
    available_gb = total_gb - reserved_gb

    return {
        "total": total_gb,
        "allocated": allocated_gb,
        "reserved": reserved_gb,
        "available": available_gb,
        "device": torch.cuda.get_device_name(0)
    }


def check_vram(threshold_gb=28):  # Leave 4GB headroom
    """Check VRAM and provide detailed diagnostics"""
    try:
        import torch
    except ImportError:
        return

    vram_info = get_vram_info()

    if not torch.cuda.is_available():
        logger.warning("CUDA not available - will use CPU")
        return

    logger.info(f"GPU: {vram_info['device']}")
    logger.info(f"Total VRAM: {vram_info['total']:.2f}GB")
    logger.info(f"Available VRAM: {vram_info['available']:.2f}GB")
    logger.info(f"Allocated VRAM: {vram_info['allocated']:.2f}GB")

    if vram_info['allocated'] > threshold_gb:
        raise VRAMConstraintError(
            f"VRAM usage {vram_info['allocated']:.2f}GB exceeds threshold {threshold_gb}GB"
        )


def get_qlora_pipeline():
    """Return quantized Qwen2.5 LLM with memory optimizations"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    vram_info = get_vram_info()

    logger.info(f"Attempting to load Qwen2.5-32B model...")
    logger.info(f"GPU: {vram_info.get('device', 'None')}, Total VRAM: {vram_info.get('total', 0):.2f}GB")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
            device_map="auto",
            dtype=torch.float16,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4")
        logger.info("Successfully loaded Qwen2.5-32B model")
        return model, tokenizer
    except (OSError, RuntimeError) as e:
        logger.warning(f"Primary model failed ({type(e).__name__}): {str(e)}")

        # First fallback: 8-bit quantization
        try:
            logger.info("Attempting 8-bit quantization fallback...")
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
                device_map="auto",
                load_in_8bit=True,
                dtype=torch.float16,
                trust_remote_code=True
            )
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4")
            logger.info("Successfully loaded Qwen2.5-32B with 8-bit quantization")
            return model, tokenizer
        except (OSError, RuntimeError) as e:
            logger.warning(f"8-bit quantization failed ({type(e).__name__}): {str(e)}")

            # Second fallback: CPU offloading with 4-bit
            try:
                logger.info("Attempting 4-bit CPU offloading fallback...")
                model = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
                    device_map="cpu",
                    load_in_4bit=True,
                    dtype=torch.float16,
                    trust_remote_code=True
                )
                tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4")
                logger.info("Successfully loaded Qwen2.5-32B with 4-bit + CPU offload")
                return model, tokenizer
            except (OSError, RuntimeError) as e:
                logger.warning(f"4-bit CPU offload failed ({type(e).__name__}): {str(e)}")

                # Third fallback: Use Mistral 7B (public, good quality)
                try:
                    logger.info("Attempting Mistral 7B-Instruct fallback...")
                    model = AutoModelForCausalLM.from_pretrained(
                        "mistralai/Mistral-7B-Instruct-v0.1",
                        device_map="auto",
                        dtype=torch.float16,
                        trust_remote_code=True
                    )
                    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
                    logger.info("Successfully loaded Mistral 7B model")
                    return model, tokenizer
                except (OSError, RuntimeError) as e:
                    logger.warning(f"Mistral 7B failed ({type(e).__name__}): {str(e)}")

                    # Fourth fallback: Use lightweight model
                    try:
                        logger.info("Loading lightweight fallback model (TinyLlama)...")
                        model = AutoModelForCausalLM.from_pretrained(
                            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                            device_map="auto",
                            dtype=torch.float32,
                            trust_remote_code=True
                        )
                        tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
                        logger.info("Successfully loaded TinyLlama fallback model")
                        return model, tokenizer
                    except (OSError, RuntimeError) as e:
                        logger.error(f"Lightweight fallback also failed ({type(e).__name__}): {str(e)}")
                        raise VRAMConstraintError(
                            f"Cannot load model - all attempts failed. "
                            f"Available VRAM: {vram_info.get('available', 0):.2f}GB. "
                            f"Error: {str(e)}"
                        ) from e


class OllamaModel:
    """Wrapper for Ollama models"""

    # Model tiers for smart defaults
    PREFERRED_MODELS = [
        "qwen3:32b",
        "qwen2.5-coder:32b",
        "mistral-small3.2:latest",
        "deepseek-coder-v2:16b-lite-instruct-q4_K_M",
    ]

    def __init__(self, model_name: str = "mistral", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama model wrapper.

        Args:
            model_name: Ollama model name (e.g., "mistral", "llama2", "neural-chat")
            base_url: Ollama server URL
        """
        self.model_name = model_name
        self.base_url = base_url
        self.generate_url = f"{base_url}/api/generate"

        # Check if Ollama is available
        self._check_availability()

    def switch_model(self, model_name: str):
        """Switch to a different Ollama model."""
        self.model_name = model_name
        logger.info(f"Switched to model: {model_name}")

    def list_available_models(self) -> list:
        """List available Ollama models."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m.get("name") for m in models]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
        return []

    @classmethod
    def get_best_available_model(cls, base_url: str = "http://localhost:11434") -> str:
        """Get the best available model from the preferred list."""
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                available = [m.get("name") for m in response.json().get("models", [])]
                for preferred in cls.PREFERRED_MODELS:
                    if preferred in available:
                        return preferred
                # Fallback to first available
                if available:
                    return available[0]
        except Exception:
            pass
        return "mistral"  # Default fallback

    def _check_availability(self):
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            if response.status_code == 200:
                logger.info(f"Ollama server available at {self.base_url}")
                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]
                logger.info(f"Available models: {model_names}")
            else:
                raise OllamaUnavailableError(f"Ollama returned status {response.status_code}")
        except (requests.ConnectionError, requests.Timeout, requests.RequestException) as e:
            raise OllamaUnavailableError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running with: ollama serve"
            ) from e

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """
        Generate text using Ollama.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        try:
            # For qwen3 models with thinking, we need more tokens
            effective_max_tokens = max_tokens
            if "qwen3" in self.model_name.lower():
                effective_max_tokens = max(max_tokens * 3, 1024)  # Allow more for thinking

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": effective_max_tokens
                }
            }

            response = requests.post(self.generate_url, json=payload, timeout=120)
            if response.status_code == 200:
                result = response.json()
                # Handle qwen3 thinking mode - response may be in 'thinking' field
                text = result.get("response", "")
                if not text and "thinking" in result:
                    # If response is empty but thinking exists, return thinking content
                    # This happens when model is still reasoning
                    thinking = result.get("thinking", "")
                    if thinking:
                        logger.info("Model used thinking mode, extracting from thinking field")
                        text = f"[Thinking]: {thinking}"
                return text
            else:
                raise OllamaUnavailableError(f"Ollama returned status {response.status_code}")
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise


class OpenAICompatibleModel:
    """Wrapper for OpenAI or OpenAI-compatible API endpoints."""

    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: str | None,
        fallback_models: list[str] | None = None,
        timeout: int = 120,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._fallback_models = fallback_models or ([] if model_name is None else [model_name])

    def switch_model(self, model_name: str):
        self.model_name = model_name
        logger.info(f"Switched to model: {model_name}")

    def list_available_models(self) -> list:
        if not self.api_key:
            logger.warning("OpenAI API key not set; using fallback model list")
            return self._fallback_models

        try:
            response = requests.get(
                f"{self.base_url}/models",
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json()
                models = [m.get("id") for m in data.get("data", []) if m.get("id")]
                return models or self._fallback_models
            logger.warning("Model listing failed: %s", response.status_code)
        except Exception as e:
            logger.warning("Model listing failed: %s", e)

        return self._fallback_models

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        if not self.api_key:
            return "Error: OpenAI API key is not set."

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout,
            )
            if response.status_code == 200:
                data = response.json()
                choices = data.get("choices") or []
                message = choices[0].get("message", {}) if choices else {}
                return message.get("content", "")
            logger.error("OpenAI API error: %s %s", response.status_code, response.text)
            return f"Error: OpenAI API returned {response.status_code}"
        except Exception as e:
            logger.error("OpenAI generation failed: %s", e)
            return f"Error: OpenAI request failed: {e}"


class AnthropicNativeModel:
    """Claude model using the native Anthropic SDK with tool-use support.

    Unlike the OpenAI-compatible shim, this uses Claude's native tool calling
    to let the model decide which tools to invoke and in what order.
    """

    MAX_TOOL_ROUNDS = 5  # Safety cap on tool-use iterations

    def __init__(self, model_name: str, api_key: str):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "anthropic package required: pip install anthropic"
            )
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name
        self.api_key = api_key

    def switch_model(self, model_name: str):
        self.model_name = model_name
        logger.info("Switched Anthropic model to: %s", model_name)

    def run_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        system: str = "",
        tool_executor=None,
        max_tokens: int = 4096,
    ) -> dict:
        """Run a tool-use conversation loop.

        Args:
            messages: Conversation messages [{"role": "user", "content": "..."}]
            tools: Tool definitions in Anthropic format
            system: System prompt
            tool_executor: Callable(tool_name, tool_input) -> str
            max_tokens: Max tokens per response

        Returns:
            {"answer": str, "tool_calls": [{"name", "input", "result"}]}
        """
        all_tool_calls = []

        for _round in range(self.MAX_TOOL_ROUNDS):
            kwargs = {
                "model": self.model_name,
                "max_tokens": max_tokens,
                "messages": messages,
            }
            if system:
                kwargs["system"] = system
            if tools:
                kwargs["tools"] = tools

            try:
                response = self.client.messages.create(**kwargs)
            except Exception as e:
                logger.error("Anthropic API error: %s", e)
                return {"answer": f"Error: Anthropic API error: {e}", "tool_calls": all_tool_calls}

            # Extract text blocks and tool_use blocks
            text_parts = []
            tool_uses = []
            for block in response.content:
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_uses.append(block)

            # If no tool calls, we're done — return the text answer
            if not tool_uses or response.stop_reason != "tool_use":
                return {
                    "answer": "\n".join(text_parts),
                    "tool_calls": all_tool_calls,
                }

            # Execute each tool call and build tool_result messages
            # First, append the assistant's response (with tool_use blocks) to messages
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for tool_use in tool_uses:
                tool_name = tool_use.name
                tool_input = tool_use.input

                if tool_executor:
                    try:
                        result_str = tool_executor(tool_name, tool_input)
                    except Exception as e:
                        logger.error("Tool %s failed: %s", tool_name, e)
                        result_str = json.dumps({"error": str(e)})
                else:
                    result_str = json.dumps({"error": "No tool executor provided"})

                all_tool_calls.append({
                    "name": tool_name,
                    "input": tool_input,
                    "result": result_str,
                })
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": result_str,
                })

            messages.append({"role": "user", "content": tool_results})

        # Exhausted rounds — return whatever text we have
        logger.warning("Claude tool-use loop hit max rounds (%d)", self.MAX_TOOL_ROUNDS)
        return {
            "answer": "\n".join(text_parts) if text_parts else "I wasn't able to complete the research within the allowed steps.",
            "tool_calls": all_tool_calls,
        }


def get_ollama_pipeline(model_name: str = "mistral", base_url: str = "http://localhost:11434") -> OllamaModel:
    """
    Get an Ollama model pipeline.

    Args:
        model_name: Ollama model to use
        base_url: Ollama server URL

    Returns:
        OllamaModel instance

    Raises:
        OllamaUnavailableError: If Ollama is not available
    """
    logger.info(f"Attempting to connect to Ollama model '{model_name}'...")
    return OllamaModel(model_name=model_name, base_url=base_url)
