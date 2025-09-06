import transformers

from .base import BaseCausalLmPipe

class Gemma2ForCausalLmPipe(BaseCausalLmPipe, transformers.Gemma2ForCausalLM):
    CONFIG_CLASS = transformers.Gemma2Config
    TRANSFORMERS_CLASS = transformers.Gemma2ForCausalLM

    def _get_attention_implementation(self):
        return 'eager'

    def _get_tie_weights(self):
        return 'model.embed_tokens.weight'

    def _get_lm_head_kwargs(self):
        return {'final_logit_softcapping': getattr(self.config, 'final_logit_softcapping', None)}

    def _normalize_embedding_sqrt(self):
        return True