from torch import nn
import accelerate
import torch

import transformers

from models.pipeline_layers import (
    LayerSpec,
    PrepareInputsPipe,
    EmbeddingPipe,
    LlamaDecoderLayerPipe,
    LlamaRMSNormPipe,
    LmHeadPipe,
    ComputeMetrics
)
from models.pipeline_model import PipelineModel

from utils.utils import DTYPE_MAP

class BaseCausalLMPipe(PipelineModel):
    """Base class for all CausalLM pipeline models."""

    CONFIG_CLASS = None  # To be overridden by subclasses
    TRANSFORMERS_CLASS = None  # To be overridden by subclasses

    def __init__(self, config, quantization_config, trust_remote_code=False):
        if self.CONFIG_CLASS is None or self.TRANSFORMERS_CLASS is None:
            raise NotImplementedError("Subclasses must define CONFIG_CLASS and TRANSFORMERS_CLASS")

        self.training_config = config

        model_config = self.CONFIG_CLASS.from_pretrained(config['model_dir'], trust_remote_code=trust_remote_code)
        model_config._attn_implementation = self._get_attention_implementation()

        torch.set_default_dtype(torch.bfloat16)  # Always use bfloat16 for full fine-tuning regardless
        with accelerate.init_empty_weights():
            self.TRANSFORMERS_CLASS.__init__(self, model_config)
            PipelineModel.__init__(self, config, quantization_config, model_config)
        torch.set_default_dtype(torch.float32)

    def to_layer_specs(self):
        embedding_on_cpu = not self.full_fine_tune
        embedding_size = 1 if embedding_on_cpu else self._get_embedding_size_hint()

        result = []

        result.append(LayerSpec(PrepareInputsPipe, _estimated_size=0))

        result.append(LayerSpec(
            EmbeddingPipe,
            self.module_loader,
            self.model.embed_tokens,
            self.model,
            embedding_on_cpu=embedding_on_cpu,
            _estimated_size=embedding_size
        ))

        for layer_idx, block in enumerate(self.model.layers):
            result.append(LayerSpec(LlamaDecoderLayerPipe, self.module_loader, block, layer_idx))

        result.append(LayerSpec(LlamaRMSNormPipe, self.module_loader, self.model.norm, _estimated_size=0))

        result.append(LayerSpec(
            LmHeadPipe,
            self.module_loader,
            self.lm_head,
            tie_weights=self._get_tie_weights(),
            _estimated_size=self._get_lm_head_size_hint(),
            **self._get_lm_head_kwargs()
        ))

        result.append(LayerSpec(ComputeMetrics))

        return result

    def _get_attention_implementation(self):
        """Override for models that need different attention implementations."""
        return 'flash_attention_2'

    def _get_embedding_size_hint(self):
        """Override for models with large embeddings."""
        return 1

    def _get_tie_weights(self):
        """Override for models with different tie_weights behavior."""
        return 'model.embed_tokens.weight' if getattr(self.config, 'tie_word_embeddings', False) else None

    def _get_lm_head_size_hint(self):
        """Override for models with large untied LM heads tensors."""
        return 0

    def _get_lm_head_kwargs(self):
        """Override to add model-specific LmHead parameters."""
        return {}

class LlamaForCausalLMPipe(BaseCausalLMPipe, transformers.LlamaForCausalLM):
    CONFIG_CLASS = transformers.LlamaConfig
    TRANSFORMERS_CLASS = transformers.LlamaForCausalLM

class MistralForCausalLMPipe(BaseCausalLMPipe, transformers.MistralForCausalLM):
    CONFIG_CLASS = transformers.MistralConfig
    TRANSFORMERS_CLASS = transformers.MistralForCausalLM

class MixtralForCausalLMPipe(BaseCausalLMPipe, transformers.MixtralForCausalLM):
    CONFIG_CLASS = transformers.MixtralConfig
    TRANSFORMERS_CLASS = transformers.MixtralForCausalLM

class Qwen2ForCausalLMPipe(BaseCausalLMPipe, transformers.Qwen2ForCausalLM):
    CONFIG_CLASS = transformers.Qwen2Config
    TRANSFORMERS_CLASS = transformers.Qwen2ForCausalLM

class Phi3ForCausalLMPipe(BaseCausalLMPipe, transformers.Phi3ForCausalLM):
    CONFIG_CLASS = transformers.Phi3Config
    TRANSFORMERS_CLASS = transformers.Phi3ForCausalLM

class CohereForCausalLMPipe(BaseCausalLMPipe, transformers.CohereForCausalLM):
    CONFIG_CLASS = transformers.CohereConfig
    TRANSFORMERS_CLASS = transformers.CohereForCausalLM

    def _get_embedding_size_hint(self):
        return 4

    def _get_tie_weights(self):
        return 'model.embed_tokens.weight'

    def _get_lm_head_kwargs(self):
        return {'logit_scale': getattr(self.config, 'logit_scale', None)}

class Cohere2ForCausalLMPipe(BaseCausalLMPipe, transformers.Cohere2ForCausalLM):
    CONFIG_CLASS = transformers.Cohere2Config
    TRANSFORMERS_CLASS = transformers.Cohere2ForCausalLM

    def _get_embedding_size_hint(self):
        return 8

    def _get_tie_weights(self):
        return 'model.embed_tokens.weight'

    def _get_lm_head_kwargs(self):
        return {'logit_scale': getattr(self.config, 'logit_scale', None)}

class Gemma2ForCausalLMPipe(BaseCausalLMPipe, transformers.Gemma2ForCausalLM):
    CONFIG_CLASS = transformers.Gemma2Config
    TRANSFORMERS_CLASS = transformers.Gemma2ForCausalLM

    def _get_attention_implementation(self):
        return 'eager'

    def _get_embedding_size_hint(self):
        return 8

    def _get_tie_weights(self):
        return 'model.embed_tokens.weight'

    def _get_lm_head_kwargs(self):
        return {'final_logit_softcapping': getattr(self.config, 'final_logit_softcapping', None)}