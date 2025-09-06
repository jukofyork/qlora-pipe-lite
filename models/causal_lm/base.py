from deepspeed.runtime.pipe.module import LayerSpec, TiedLayerSpec
import accelerate
import torch

from models.modules import (
    PrepareInputsPipe,
    EmbeddingPipe,
    Gemma3EmbeddingPipe,
    DecoderLayerPipe,
    Gemma3DecoderLayerPipe,
    NormPipe,
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
        result.append(LayerSpec(PrepareInputsPipe))

        if bool(self.full_fine_tune) and bool(getattr(self.config, 'tie_word_embeddings', False)):
            # 1) Tied embedding at the start
            result.append(TiedLayerSpec(
                'tok_emb',
                Gemma3EmbeddingPipe if self._require_local_rotary() else EmbeddingPipe,
                self.module_loader,
                self.model.embed_tokens,
                self.model,
                embedding_on_cpu=False,
                normalize_embedding_sqrt=self._normalize_embedding_sqrt(),
                tied_weight_attr='orig.weight'
            ))

            # 2) Decoder layers
            for layer_idx, block in enumerate(self.model.layers):
                result.append(LayerSpec(
                    Gemma3DecoderLayerPipe if self._require_local_rotary() else DecoderLayerPipe,
                    self.module_loader,
                    block,
                    layer_idx
                ))

            # 3) Final norm
            result.append(LayerSpec(NormPipe, self.module_loader, self.model.norm))

            # 4) Tied "head" using the same module instance with a custom forward
            lm_head_kwargs = self._get_lm_head_kwargs()
            logit_scale = lm_head_kwargs.get('logit_scale', None)
            final_logit_softcapping = lm_head_kwargs.get('final_logit_softcapping', None)

            def tied_lm_head_forward(tied_module, inputs):
                # inputs are (hidden_states, labels) coming from NormPipe
                hidden_states, labels = inputs

                if logit_scale is not None:
                    hidden_states = hidden_states * logit_scale

                weight = tied_module.orig.weight  # [vocab, hidden]
                logits = torch.matmul(hidden_states, weight.t())  # [b, s, hidden] x [hidden, vocab] -> [b, s, vocab]

                if final_logit_softcapping is not None:
                    logits = logits / final_logit_softcapping
                    logits = torch.tanh(logits)
                    logits = logits * final_logit_softcapping

                return logits, labels

            result.append(TiedLayerSpec(
                'tok_emb',
                Gemma3EmbeddingPipe if self._require_local_rotary() else EmbeddingPipe,
                self.module_loader,
                self.model.embed_tokens,
                self.model,
                embedding_on_cpu=False,
                normalize_embedding_sqrt=self._normalize_embedding_sqrt(),
                forward_fn=tied_lm_head_forward,
                tied_weight_attr='orig.weight'
            ))

            # 5) Loss
            result.append(LayerSpec(LossPipe))
            return result

        # Non-FFT or models without tied embeddings: original untied pipeline
        result.append(LayerSpec(
            Gemma3EmbeddingPipe if self._require_local_rotary() else EmbeddingPipe,
            self.module_loader,
            self.model.embed_tokens,
            self.model,
            embedding_on_cpu=embedding_on_cpu,
            normalize_embedding_sqrt=self._normalize_embedding_sqrt()
        ))

        for layer_idx, block in enumerate(self.model.layers):
            result.append(LayerSpec(
                Gemma3DecoderLayerPipe if self._require_local_rotary() else DecoderLayerPipe,
                self.module_loader,
                block,
                layer_idx
            ))

        result.append(LayerSpec(NormPipe, self.module_loader, self.model.norm))

        result.append(LayerSpec(
            LmHeadPipe,
            self.module_loader,
            self.lm_head,
            tie_weights=self._get_tie_weights(),
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