import transformers

from .base import BaseCausalLmPipe

class Phi3ForCausalLmPipe(BaseCausalLmPipe, transformers.Phi3ForCausalLM):
    CONFIG_CLASS = transformers.Phi3Config
    TRANSFORMERS_CLASS = transformers.Phi3ForCausalLM