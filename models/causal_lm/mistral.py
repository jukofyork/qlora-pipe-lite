import transformers

from .base import BaseCausalLmPipe

class MistralForCausalLmPipe(BaseCausalLmPipe, transformers.MistralForCausalLM):
    CONFIG_CLASS = transformers.MistralConfig
    TRANSFORMERS_CLASS = transformers.MistralForCausalLM