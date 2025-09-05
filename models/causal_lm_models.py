from deepspeed.runtime.pipe.module import LayerSpec, TiedLayerSpec
from torch import nn
import accelerate
import torch

import transformers

from models.pipeline_layers import (
    PrepareInputsPipe,
    EmbeddingPipe,
    LlamaDecoderLayerPipe,
    LlamaRMSNormPipe,
    LmHeadPipe,
    ComputeLoss
)
from models.pipeline_model import PipelineModel

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

        result = []
        result.append(LayerSpec(PrepareInputsPipe))

        use_tied_layers = bool(self.full_fine_tune) and bool(getattr(self.config, 'tie_word_embeddings', False))

        if use_tied_layers:
            # Full fine-tuning with tied embeddings/head: use a single tied module (EmbeddingPipe)
            # and provide a custom forward for the head that projects with the embedding weight.
            #
            # DeepSpeed ties gradients across occurrences of the same module key. We tie the single
            # parameter at orig.weight (nn.Embedding weight) across stages.
            #
            # Note: We intentionally do NOT instantiate LmHeadPipe in this mode.

            # 1) Tied embedding at the start
            result.append(TiedLayerSpec(
                'tok_emb',
                EmbeddingPipe,
                self.module_loader,
                self.model.embed_tokens,
                self.model,
                embedding_on_cpu=False,
                tied_weight_attr='orig.weight'
            ))

            # 2) Decoder layers
            for layer_idx, block in enumerate(self.model.layers):
                result.append(LayerSpec(LlamaDecoderLayerPipe, self.module_loader, block, layer_idx))

            # 3) Final RMSNorm
            result.append(LayerSpec(LlamaRMSNormPipe, self.module_loader, self.model.norm))

            # 4) Tied "head" using the same module instance with a custom forward
            lm_head_kwargs = self._get_lm_head_kwargs()
            logit_scale = lm_head_kwargs.get('logit_scale', None)
            final_logit_softcapping = lm_head_kwargs.get('final_logit_softcapping', None)

            def tied_lm_head_forward(tied_module, inputs):
                # inputs are (hidden_states, labels) coming from LlamaRMSNormPipe
                hidden_states, labels = inputs

                # Optional logit scale (apply to hidden states for identical math)
                if logit_scale is not None:
                    hidden_states = hidden_states * logit_scale

                # Project using the embedding weight
                weight = tied_module.orig.weight  # [vocab, hidden]
                logits = torch.matmul(hidden_states, weight.t())  # [b, s, hidden] x [hidden, vocab] -> [b, s, vocab]

                # Optional final logit softcapping
                if final_logit_softcapping is not None:
                    logits = logits / final_logit_softcapping
                    logits = torch.tanh(logits)
                    logits = logits * final_logit_softcapping

                return logits, labels

            result.append(TiedLayerSpec(
                'tok_emb',
                EmbeddingPipe,  # reuse the same module class/key
                self.module_loader,
                self.model.embed_tokens,
                self.model,
                embedding_on_cpu=False,
                forward_fn=tied_lm_head_forward,
                tied_weight_attr='orig.weight'
            ))

            # 5) Loss
            result.append(LayerSpec(ComputeLoss))
            return result

        # Non-FFT or models without tied embeddings: original untied pipeline
        result.append(LayerSpec(
            EmbeddingPipe,
            self.module_loader,
            self.model.embed_tokens,
            self.model,
            embedding_on_cpu=embedding_on_cpu
        ))

        for layer_idx, block in enumerate(self.model.layers):
            result.append(LayerSpec(LlamaDecoderLayerPipe, self.module_loader, block, layer_idx))

        result.append(LayerSpec(LlamaRMSNormPipe, self.module_loader, self.model.norm))

        result.append(LayerSpec(
            LmHeadPipe,
            self.module_loader,
            self.lm_head,
            tie_weights=self._get_tie_weights(),
            **self._get_lm_head_kwargs()
        ))

        result.append(LayerSpec(ComputeLoss))
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

class Qwen3ForCausalLMPipe(BaseCausalLMPipe, transformers.Qwen3ForCausalLM):
    CONFIG_CLASS = transformers.Qwen3Config
    TRANSFORMERS_CLASS = transformers.Qwen3ForCausalLM

class Phi3ForCausalLMPipe(BaseCausalLMPipe, transformers.Phi3ForCausalLM):
    CONFIG_CLASS = transformers.Phi3Config
    TRANSFORMERS_CLASS = transformers.Phi3ForCausalLM

class CohereForCausalLMPipe(BaseCausalLMPipe, transformers.CohereForCausalLM):
    CONFIG_CLASS = transformers.CohereConfig
    TRANSFORMERS_CLASS = transformers.CohereForCausalLM

    def _get_tie_weights(self):
        return 'model.embed_tokens.weight'

    def _get_lm_head_kwargs(self):
        return {'logit_scale': getattr(self.config, 'logit_scale', None)}

class Cohere2ForCausalLMPipe(BaseCausalLMPipe, transformers.Cohere2ForCausalLM):
    CONFIG_CLASS = transformers.Cohere2Config
    TRANSFORMERS_CLASS = transformers.Cohere2ForCausalLM

    def _get_tie_weights(self):
        return 'model.embed_tokens.weight'

    def _get_lm_head_kwargs(self):
        return {'logit_scale': getattr(self.config, 'logit_scale', None)}

class Gemma2ForCausalLMPipe(BaseCausalLMPipe, transformers.Gemma2ForCausalLM):
    CONFIG_CLASS = transformers.Gemma2Config
    TRANSFORMERS_CLASS = transformers.Gemma2ForCausalLM

    def _get_attention_implementation(self):
        return 'eager'

    def _get_tie_weights(self):
        return 'model.embed_tokens.weight'

    def _get_lm_head_kwargs(self):
        return {'final_logit_softcapping': getattr(self.config, 'final_logit_softcapping', None)}