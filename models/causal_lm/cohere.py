import transformers

from .base import BaseCausalLmPipe

class CohereForCausalLmPipe(BaseCausalLmPipe, transformers.CohereForCausalLM):
    CONFIG_CLASS = transformers.CohereConfig
    TRANSFORMERS_CLASS = transformers.CohereForCausalLM

    def _get_tie_weights(self):
        return 'model.embed_tokens.weight'

    def _get_lm_head_kwargs(self):
        return {'logit_scale': getattr(self.config, 'logit_scale', None)}