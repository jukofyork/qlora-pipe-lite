from deepspeed.runtime.pipe.module import LayerSpec, TiedLayerSpec
import accelerate
import torch

from models.modules import (
    PrepareInputsPipe,
    EmbeddingPipe,
    DecoderLayerPipe,
    FinalNormPipe,
    LmHeadPipe,
    LossPipe,
)
from models.pipeline_model import PipelineModel

class BaseCausalLmPipe(PipelineModel):
    """Base class for all Causal LM pipeline models."""

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
        """Build the pipeline as LayerSpec/TiedLayerSpec sequence."""
        embedding_on_cpu = not self.full_fine_tune

        result = []

        # Prepare Inputs
        result.append(LayerSpec(PrepareInputsPipe))

        tie_weights = self._get_tie_weights()
        use_tied = bool(self.full_fine_tune) and bool(tie_weights)

        normalize_embedding_sqrt = self._normalize_embedding_sqrt()
        require_local_rotary = self._require_local_rotary()

        def _build_tied_embedding(forward_fn=None):
            return TiedLayerSpec(
                'tied_embedding',
                EmbeddingPipe,
                self.module_loader,
                self.model.embed_tokens,
                self.model,
                embedding_on_cpu=False,
                normalize_embedding_sqrt=normalize_embedding_sqrt,
                require_local_rotary=require_local_rotary,
                forward_fn=forward_fn,
                tied_weight_attr='orig.weight'
            )

        # Embedding
        if use_tied:
            # Share the same EmbeddingPipe module instance for embedding/head to preserve DS tied-module semantics.
            result.append(_build_tied_embedding())
        else:
            result.append(LayerSpec(
                EmbeddingPipe,
                self.module_loader,
                self.model.embed_tokens,
                self.model,
                embedding_on_cpu=embedding_on_cpu,
                normalize_embedding_sqrt=normalize_embedding_sqrt,
                require_local_rotary=require_local_rotary
            ))

        # Decoder Layers
        for layer_idx, block in enumerate(self.model.layers):
            result.append(LayerSpec(
                DecoderLayerPipe,
                self.module_loader,
                block,
                layer_idx,
                require_local_rotary=require_local_rotary
            ))

        # Final Norm
        result.append(LayerSpec(FinalNormPipe, self.module_loader, self.model.norm))

        # LM Head
        if use_tied:
            # Share the same EmbeddingPipe module instance for embedding/head to preserve DS tied-module semantics.
            result.append(_build_tied_embedding(LmHeadPipe.make_tied_lm_head_forward(**self._get_lm_head_kwargs())))
        else:
            result.append(LayerSpec(
                LmHeadPipe,
                self.module_loader,
                self.lm_head,
                tie_weights=tie_weights,
                **self._get_lm_head_kwargs()
            ))

        result.append(LayerSpec(LossPipe))

        return result

    def _get_attention_implementation(self):
        """Override for models that need different attention implementations."""
        return 'flash_attention_2'

    def _get_tie_weights(self):
        """Override for models with different tie_weights behavior."""
        return 'model.embed_tokens.weight' if getattr(self.config, 'tie_word_embeddings', False) else None

    def _get_lm_head_kwargs(self):
        """Override to add model-specific LmHead parameters."""
        return {}

    def _normalize_embedding_sqrt(self):
        """Override in subclasses that require sqrt(hidden_size) embedding scaling (e.g., Gemma 2/3)."""
        return False

    def _require_local_rotary(self):
        """Override in subclasses that need dual rotary embeddings (e.g., Gemma 3)."""
        return False