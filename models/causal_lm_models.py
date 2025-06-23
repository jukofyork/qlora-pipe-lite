from torch import nn
import accelerate
import torch

import transformers

from models.pipeline_layers import (
    LayerSpec,
    EmbeddingPipe,
    LlamaDecoderLayerPipe,
    LlamaRMSNormPipe,
    LmHeadPipe,
    ComputeMetrics
)
from models.pipeline_model import PipelineModel

class BaseCausalLMPipe(PipelineModel):
    """Base class for all CausalLM pipeline models."""

    CONFIG_CLASS = None  # To be overridden by subclasses
    TRANSFORMERS_CLASS = None  # To be overridden by subclasses

    def __init__(self, config, quantization_config):
        if self.CONFIG_CLASS is None or self.TRANSFORMERS_CLASS is None:
            raise NotImplementedError("Subclasses must define CONFIG_CLASS and TRANSFORMERS_CLASS")

        model_config = self.CONFIG_CLASS.from_pretrained(config['model_dir'])
        model_config._attn_implementation = self._get_attention_implementation()

        torch.set_default_dtype(torch.bfloat16)
        with accelerate.init_empty_weights():
            self.TRANSFORMERS_CLASS.__init__(self, model_config)
            PipelineModel.__init__(self, config, quantization_config, model_config)
        torch.set_default_dtype(torch.float32)

    def _get_attention_implementation(self):
        """Override for models that need different attention implementations."""
        return 'flash_attention_2'

    def _get_embedding_size_hint(self):
        """Override for models with large embeddings."""
        return 1

    def _get_lm_head_kwargs(self):
        """Override to add model-specific LmHead parameters."""
        return {}

    def _get_tie_weights(self):
        """Override for models with different tie_weights behavior."""
        return 'model.embed_tokens.weight' if getattr(self.config, 'tie_word_embeddings', False) else None

    def _get_compute_metrics_kwargs(self):
        """Override to add model-specific ComputeMetrics parameters."""
        return {}

    def to_layer_specs(self):
        result = [
            self._create_initial_layer(),
            self._create_embedding_layer(),
        ]

        for block in self.model.layers:
            result.append(LayerSpec(LlamaDecoderLayerPipe, self.loader_util, block))

        result.append(LayerSpec(LlamaRMSNormPipe, self.loader_util, self.model.norm, _estimated_size=0))
        result.append(self._create_lm_head_layer())
        result.append(LayerSpec(ComputeMetrics, **self._get_compute_metrics_kwargs()))

        return result

    def _create_initial_layer(self):

        def initial_layer(inputs):
            input_ids, attention_mask, labels = inputs
            batch_size, seq_length = input_ids.shape[:2]
            device = input_ids.device
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)
            return input_ids, attention_mask, position_ids, labels

        return initial_layer

    def _create_embedding_layer(self):
        embedding_on_cpu = not self.full_fine_tune
        embedding_size = 1 if embedding_on_cpu else self._get_embedding_size_hint()

        return LayerSpec(
            EmbeddingPipe,
            self.loader_util,
            self.model.embed_tokens,
            self.model,
            embedding_on_cpu=embedding_on_cpu,
            _estimated_size=embedding_size,
        )

    def _create_lm_head_layer(self):
        kwargs = {
            'tie_weights': self._get_tie_weights(),
            '_estimated_size': 0
        }
        kwargs.update(self._get_lm_head_kwargs())

        return LayerSpec(LmHeadPipe, self.loader_util, self.lm_head, **kwargs)

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

    def _get_lm_head_kwargs(self):
        return {'logit_scale': self.logit_scale}

    def _get_tie_weights(self):
        return 'model.embed_tokens.weight'

class Cohere2ForCausalLMPipe(BaseCausalLMPipe, transformers.Cohere2ForCausalLM):
    CONFIG_CLASS = transformers.Cohere2Config
    TRANSFORMERS_CLASS = transformers.Cohere2ForCausalLM

    def _get_embedding_size_hint(self):
        return 8

    def _get_lm_head_kwargs(self):
        return {'logit_scale': self.logit_scale}

    def _get_tie_weights(self):
        return 'model.embed_tokens.weight'

class Gemma2ForCausalLMPipe(BaseCausalLMPipe, transformers.Gemma2ForCausalLM):
    CONFIG_CLASS = transformers.Gemma2Config
    TRANSFORMERS_CLASS = transformers.Gemma2ForCausalLM

    def _get_attention_implementation(self):
        return 'eager'

    def _get_embedding_size_hint(self):
        return 8

    def _get_lm_head_kwargs(self):
        return {'final_logit_softcapping': self.final_logit_softcapping}

    def _get_tie_weights(self):
        return 'model.embed_tokens.weight'

    def _get_compute_metrics_kwargs(self):
        return {'_estimated_size': 8}