import transformers

from .base import BaseCausalLmPipe

class MixtralForCausalLmPipe(BaseCausalLmPipe, transformers.MixtralForCausalLM):
    CONFIG_CLASS = transformers.MixtralConfig
    TRANSFORMERS_CLASS = transformers.MixtralForCausalLM