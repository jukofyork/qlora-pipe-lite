import transformers

from .base import BaseCausalLmPipe

class Cohere2ForCausalLmPipe(BaseCausalLmPipe, transformers.Cohere2ForCausalLM):
    CONFIG_CLASS = transformers.Cohere2Config
    TRANSFORMERS_CLASS = transformers.Cohere2ForCausalLM

    def _get_tie_weights(self):
        return 'model.embed_tokens.weight'

    def _get_lm_head_kwargs(self):
        return {'logit_scale': getattr(self.config, 'logit_scale', None)}