import transformers

from .base import BaseCausalLmPipe

class LlamaForCausalLmPipe(BaseCausalLmPipe, transformers.LlamaForCausalLM):
    CONFIG_CLASS = transformers.LlamaConfig
    TRANSFORMERS_CLASS = transformers.LlamaForCausalLM