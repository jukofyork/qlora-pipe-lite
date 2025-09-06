import transformers

from .base import BaseCausalLmPipe

class Qwen2ForCausalLmPipe(BaseCausalLmPipe, transformers.Qwen2ForCausalLM):
    CONFIG_CLASS = transformers.Qwen2Config
    TRANSFORMERS_CLASS = transformers.Qwen2ForCausalLM