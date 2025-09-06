import transformers

from .base import BaseCausalLmPipe

class Qwen3ForCausalLmPipe(BaseCausalLmPipe, transformers.Qwen3ForCausalLM):
    CONFIG_CLASS = transformers.Qwen3Config
    TRANSFORMERS_CLASS = transformers.Qwen3ForCausalLM