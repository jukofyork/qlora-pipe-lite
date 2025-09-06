import transformers

from .base import BaseCausalLmPipe

class Gemma3ForCausalLmPipe(BaseCausalLmPipe, transformers.Gemma3ForCausalLM):
    CONFIG_CLASS = transformers.Gemma3Config
    TRANSFORMERS_CLASS = transformers.Gemma3ForCausalLM

    def _get_attention_implementation(self):
        return 'eager'

    def _get_tie_weights(self):
        return 'model.embed_tokens.weight'

    def _get_lm_head_kwargs(self):
        return {'final_logit_softcapping': getattr(self.config, 'final_logit_softcapping', None)}

    def _normalize_embedding_sqrt(self):
        return True

    def _require_local_rotary(self):
        return True