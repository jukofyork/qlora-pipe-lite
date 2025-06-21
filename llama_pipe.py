import torch
from torch import nn
import transformers
import accelerate

from pipeline_model import LayerSpec, PipelineModel

from kernels.cross_entropy_loss import Fast_CrossEntropyLoss


class EmbeddingPipe(nn.Module):
    def __init__(self, loader_util, orig, model, embedding_on_cpu=False):
        super().__init__()
        self.orig = orig
        # The original model object, e.g. LlamaModel. Use a list so the nn.Module isn't registered to this module.
        self.model = [model]
        self.embedding_on_cpu = embedding_on_cpu
        loader_util.load_state_dict_into_module(self)

    def forward(self, inputs):
        input_ids, attention_mask, position_ids, labels = inputs
        original_device = input_ids.device
        if self.embedding_on_cpu:
            self.orig.to('cpu')
            input_ids = input_ids.to('cpu')
        inputs_embeds = self.orig(input_ids).to(original_device)

        original_attention_mask = attention_mask
        past_key_values = None  # always None for training
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        if self.model[0].config.model_type in ['mistral', 'mixtral']:
            attention_mask = self.model[0]._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values, False
            )
        else:
            attention_mask = self.model[0]._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values, False
            )
        if attention_mask is None:
            # With FA, attention_mask can end up being None. But with deepspeed we can't pass None
            # between GPUs. So force it back to the original attention_mask.
            attention_mask = original_attention_mask

        hidden_states = inputs_embeds
        if self.model[0].config.model_type == 'gemma2':
            normalizer = torch.tensor(self.model[0].config.hidden_size**0.5, dtype=hidden_states.dtype)
            hidden_states = hidden_states * normalizer

        # Compute rotary embeddings
        cos, sin = self.model[0].rotary_emb(hidden_states, position_ids)

        # We have to do this so activation checkpointing with reentrant checkpoint function (the default) works.
        # We could just use non-reentrant instead, but that has some weird bug with flash attn where the memory usage is very high.
        hidden_states.requires_grad_(True)
        # Without flash attn, the attention_mask is a float. With pipeline parallel, any float tensors sent across GPUs must have requires_grad.
        # This is a workaround, theoretically there's no reason to require this.
        if torch.is_floating_point(attention_mask):
            attention_mask.requires_grad_(True)
        if torch.is_floating_point(cos):
            cos.requires_grad_(True)
        if torch.is_floating_point(sin):
            sin.requires_grad_(True)
        return hidden_states, attention_mask, cos, sin, labels


class LlamaDecoderLayerPipe(nn.Module):
    def __init__(self, loader_util, orig):
        super().__init__()
        self.orig = orig
        loader_util.load_state_dict_into_module(self)

    def forward(self, inputs):
        hidden_states, attention_mask, cos, sin, labels = inputs
        result = (self.orig(hidden_states, attention_mask=attention_mask, position_embeddings=(cos, sin))[0], attention_mask, cos, sin, labels)
        return result
    
    
class LlamaRMSNormPipe(nn.Module):
    def __init__(self, loader_util, orig):
        super().__init__()
        self.orig = orig
        loader_util.load_state_dict_into_module(self)

    def forward(self, inputs):
        hidden_states, _, _, _, labels = inputs
        return self.orig(hidden_states), labels


class LmHeadPipe(nn.Module):
    def __init__(self, loader_util, lm_head, logit_scale=None, final_logit_softcapping=None, tie_weights=None):
        super().__init__()
        # Unlike the other wrapper classes, this is called lm_head and not orig. Because this is directly a
        # nn.Linear layer, it needs to keep the same attribute name so quantization knows not to quantize it.
        self.lm_head = lm_head
        self.logit_scale = logit_scale
        self.final_logit_softcapping = final_logit_softcapping
        if tie_weights:
            self.lm_head.weight.original_name = tie_weights
        loader_util.load_state_dict_into_module(self)

    def forward(self, inputs):
        hidden_states, labels = inputs
        if self.logit_scale is not None:
            hidden_states = hidden_states * self.logit_scale
        logits = self.lm_head(hidden_states)
        if self.final_logit_softcapping is not None:
            logits = logits / self.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.final_logit_softcapping
        return logits, labels


class ComputeMetrics(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        logits, labels = inputs
        shift_logits = logits
        extra_ignored_labels = torch.full((labels.shape[0], 1), -100, device=logits.device)
        shift_labels = torch.hstack((labels[..., 1:], extra_ignored_labels))
        # Flatten the tokens
        vocab_size = shift_logits.size(-1)
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        valid_loss = (shift_labels >= 0)

        loss_unreduced = Fast_CrossEntropyLoss.apply(shift_logits, shift_labels)[valid_loss]
       
        return loss_unreduced.mean()


class LlamaForCausalLMPipe(PipelineModel, transformers.LlamaForCausalLM):
    def __init__(self, config, quantization_config):
        model_config = transformers.LlamaConfig.from_pretrained(config['model'])
        model_config._attn_implementation = 'flash_attention_2'
        torch.set_default_dtype(torch.bfloat16)
        with accelerate.init_empty_weights():
            transformers.LlamaForCausalLM.__init__(self, model_config)
            PipelineModel.__init__(self, config, quantization_config, model_config)
        torch.set_default_dtype(torch.float32)

    def to_layer_specs(self):
        def initial_layer(inputs):
            input_ids, attention_mask, labels = inputs
            batch_size, seq_length = input_ids.shape[:2]
            device = input_ids.device
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)
            return input_ids, attention_mask, position_ids, labels

        result = [
            initial_layer,
            LayerSpec(
                EmbeddingPipe,
                self.loader_util,
                self.model.embed_tokens,
                self.model,
                embedding_on_cpu=not self.full_fine_tune
            ),
        ]
        for block in self.model.layers:
            result.append(LayerSpec(LlamaDecoderLayerPipe, self.loader_util, block))
        result.append(LayerSpec(LlamaRMSNormPipe, self.loader_util, self.model.norm, _estimated_size=0))
        result.append(LayerSpec(
            LmHeadPipe,
            self.loader_util,
            self.lm_head,
            tie_weights='model.embed_tokens.weight' if self.config.tie_word_embeddings else None,
            _estimated_size=0
        ))
        result.append(LayerSpec(ComputeMetrics))
        return result


class MistralForCausalLMPipe(PipelineModel, transformers.MistralForCausalLM):
    def __init__(self, config, quantization_config):
        model_config = transformers.MistralConfig.from_pretrained(config['model'])
        model_config._attn_implementation = 'flash_attention_2'
        torch.set_default_dtype(torch.bfloat16)
        with accelerate.init_empty_weights():
            transformers.MistralForCausalLM.__init__(self, model_config)
            PipelineModel.__init__(self, config, quantization_config, model_config)
        torch.set_default_dtype(torch.float32)

    def to_layer_specs(self):
        def initial_layer(inputs):
            input_ids, attention_mask, labels = inputs
            batch_size, seq_length = input_ids.shape[:2]
            device = input_ids.device
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)
            return input_ids, attention_mask, position_ids, labels

        result = [
            initial_layer,
            LayerSpec(
                EmbeddingPipe,
                self.loader_util,
                self.model.embed_tokens,
                self.model,
                embedding_on_cpu=not self.full_fine_tune
            ),
        ]
        for block in self.model.layers:
            result.append(LayerSpec(LlamaDecoderLayerPipe, self.loader_util, block))
        result.append(LayerSpec(LlamaRMSNormPipe, self.loader_util, self.model.norm, _estimated_size=0))
        result.append(LayerSpec(LmHeadPipe, self.loader_util, self.lm_head, _estimated_size=0))
        result.append(LayerSpec(ComputeMetrics))
        return result
    

class MixtralForCausalLMPipe(PipelineModel, transformers.MixtralForCausalLM):
    def __init__(self, config, quantization_config):
        model_config = transformers.MixtralConfig.from_pretrained(config['model'])
        model_config._attn_implementation = 'flash_attention_2'
        torch.set_default_dtype(torch.bfloat16)
        with accelerate.init_empty_weights():
            transformers.MixtralForCausalLM.__init__(self, model_config)
            PipelineModel.__init__(self, config, quantization_config, model_config)
        torch.set_default_dtype(torch.float32)

    def to_layer_specs(self):
        def initial_layer(inputs):
            input_ids, attention_mask, labels = inputs
            batch_size, seq_length = input_ids.shape[:2]
            device = input_ids.device
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)
            return input_ids, attention_mask, position_ids, labels

        result = [
            initial_layer,
            LayerSpec(
                EmbeddingPipe,
                self.loader_util,
                self.model.embed_tokens,
                self.model,
                embedding_on_cpu=not self.full_fine_tune
            ),
        ]
        for block in self.model.layers:
            result.append(LayerSpec(LlamaDecoderLayerPipe, self.loader_util, block))
        result.append(LayerSpec(LlamaRMSNormPipe, self.loader_util, self.model.norm, _estimated_size=0))
        result.append(LayerSpec(LmHeadPipe, self.loader_util, self.lm_head, _estimated_size=0))
        result.append(LayerSpec(ComputeMetrics))
        return result


class Qwen2ForCausalLMPipe(PipelineModel, transformers.Qwen2ForCausalLM):
    def __init__(self, config, quantization_config):
        model_config = transformers.Qwen2Config.from_pretrained(config['model'])
        model_config._attn_implementation = 'flash_attention_2'
        torch.set_default_dtype(torch.bfloat16)
        with accelerate.init_empty_weights():
            transformers.Qwen2ForCausalLM.__init__(self, model_config)
            PipelineModel.__init__(self, config, quantization_config, model_config)
        torch.set_default_dtype(torch.float32)

    def to_layer_specs(self):
        def initial_layer(inputs):
            input_ids, attention_mask, labels = inputs
            batch_size, seq_length = input_ids.shape[:2]
            device = input_ids.device
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)
            return input_ids, attention_mask, position_ids, labels

        result = [
            initial_layer,
            LayerSpec(
                EmbeddingPipe,
                self.loader_util,
                self.model.embed_tokens,
                self.model,
                embedding_on_cpu=not self.full_fine_tune
            ),
        ]
        for block in self.model.layers:
            result.append(LayerSpec(LlamaDecoderLayerPipe, self.loader_util, block))
        result.append(LayerSpec(LlamaRMSNormPipe, self.loader_util, self.model.norm, _estimated_size=0))
        result.append(LayerSpec(
            LmHeadPipe,
            self.loader_util,
            self.lm_head,
            tie_weights='model.embed_tokens.weight' if self.config.tie_word_embeddings else None,
            _estimated_size=0
        ))
        result.append(LayerSpec(ComputeMetrics))
        return result


class Phi3ForCausalLMPipe(PipelineModel, transformers.Phi3ForCausalLM):
    def __init__(self, config, quantization_config):
        model_config = transformers.Phi3Config.from_pretrained(config['model'])
        model_config._attn_implementation = 'flash_attention_2'
        torch.set_default_dtype(torch.bfloat16)
        with accelerate.init_empty_weights():
            transformers.Phi3ForCausalLM.__init__(self, model_config)
            PipelineModel.__init__(self, config, quantization_config, model_config)
        torch.set_default_dtype(torch.float32)

    def to_layer_specs(self):
        def initial_layer(inputs):
            input_ids, attention_mask, labels = inputs
            batch_size, seq_length = input_ids.shape[:2]
            device = input_ids.device
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)
            return input_ids, attention_mask, position_ids, labels

        result = [
            initial_layer,
            LayerSpec(
                EmbeddingPipe,
                self.loader_util,
                self.model.embed_tokens,
                self.model,
                embedding_on_cpu=not self.full_fine_tune
            ),
        ]
        for block in self.model.layers:
            result.append(LayerSpec(LlamaDecoderLayerPipe, self.loader_util, block))
        result.append(LayerSpec(LlamaRMSNormPipe, self.loader_util, self.model.norm, _estimated_size=0))
        result.append(LayerSpec(LmHeadPipe, self.loader_util, self.lm_head, _estimated_size=0))
        result.append(LayerSpec(ComputeMetrics))
        return result


class CohereForCausalLMPipe(PipelineModel, transformers.CohereForCausalLM):
    def __init__(self, config, quantization_config):
        model_config = transformers.CohereConfig.from_pretrained(config['model'])
        model_config._attn_implementation = 'flash_attention_2'
        torch.set_default_dtype(torch.bfloat16)
        with accelerate.init_empty_weights():
            transformers.CohereForCausalLM.__init__(self, model_config)
            PipelineModel.__init__(self, config, quantization_config, model_config)
        torch.set_default_dtype(torch.float32)

    def to_layer_specs(self):
        # the embedding table for this model is huge; load balance it better with some heuristics
        embedding_relative_size = 4

        def initial_layer(inputs):
            input_ids, attention_mask, labels = inputs
            batch_size, seq_length = input_ids.shape[:2]
            device = input_ids.device
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)
            return input_ids, attention_mask, position_ids, labels

        embedding_on_cpu = not self.full_fine_tune
        result = [
            initial_layer,
            LayerSpec(
                EmbeddingPipe,
                self.loader_util,
                self.model.embed_tokens,
                self.model,
                embedding_on_cpu=embedding_on_cpu,
                _estimated_size=1 if embedding_on_cpu else embedding_relative_size,
            ),
        ]
        for block in self.model.layers:
            result.append(LayerSpec(LlamaDecoderLayerPipe, self.loader_util, block))
        result.append(LayerSpec(LlamaRMSNormPipe, self.loader_util, self.model.norm, _estimated_size=0))
        result.append(
            LayerSpec(
                LmHeadPipe,
                self.loader_util,
                self.lm_head,
                logit_scale=self.logit_scale,
                tie_weights='model.embed_tokens.weight'
            )
        )
        result.append(LayerSpec(ComputeMetrics))
        return result


class Cohere2ForCausalLMPipe(PipelineModel, transformers.Cohere2ForCausalLM):
    def __init__(self, config, quantization_config):
        model_config = transformers.Cohere2Config.from_pretrained(config['model'])
        model_config._attn_implementation = 'flash_attention_2'
        torch.set_default_dtype(torch.bfloat16)
        with accelerate.init_empty_weights():
            transformers.Cohere2ForCausalLM.__init__(self, model_config)
            PipelineModel.__init__(self, config, quantization_config, model_config)
        torch.set_default_dtype(torch.float32)

    def to_layer_specs(self):
        # the embedding table for this model is huge; load balance it better with some heuristics
        embedding_relative_size = 8

        def initial_layer(inputs):
            input_ids, attention_mask, labels = inputs
            batch_size, seq_length = input_ids.shape[:2]
            device = input_ids.device
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)
            return input_ids, attention_mask, position_ids, labels

        embedding_on_cpu = not self.full_fine_tune
        result = [
            initial_layer,
            LayerSpec(
                EmbeddingPipe,
                self.loader_util,
                self.model.embed_tokens,
                self.model,
                embedding_on_cpu=embedding_on_cpu,
                _estimated_size=2 if embedding_on_cpu else embedding_relative_size,
            ),
        ]
        for block in self.model.layers:
            result.append(LayerSpec(LlamaDecoderLayerPipe, self.loader_util, block))
        result.append(LayerSpec(LlamaRMSNormPipe, self.loader_util, self.model.norm, _estimated_size=0))
        result.append(
            LayerSpec(
                LmHeadPipe,
                self.loader_util,
                self.lm_head,
                logit_scale=self.logit_scale,
                tie_weights='model.embed_tokens.weight'
            )
        )
        result.append(LayerSpec(ComputeMetrics))
        return result


class Gemma2ForCausalLMPipe(PipelineModel, transformers.Gemma2ForCausalLM):
    def __init__(self, config, quantization_config):
        model_config = transformers.Gemma2Config.from_pretrained(config['model'])
        # TODO: change this when Gemma works with other attn implementations
        model_config._attn_implementation = 'eager'
        torch.set_default_dtype(torch.bfloat16)
        with accelerate.init_empty_weights():
            transformers.Gemma2ForCausalLM.__init__(self, model_config)
            PipelineModel.__init__(self, config, quantization_config, model_config)
        torch.set_default_dtype(torch.float32)

    def to_layer_specs(self):
        # the embedding table for this model is huge; load balance it better with some heuristics
        # this value optimized for LoRA, pipeline_stages=2
        embedding_relative_size = 8

        def initial_layer(inputs):
            input_ids, attention_mask, labels = inputs
            batch_size, seq_length = input_ids.shape[:2]
            device = input_ids.device
            position_ids = torch.arange(
                0, seq_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)
            return input_ids, attention_mask, position_ids, labels

        embedding_on_cpu = not self.full_fine_tune
        result = [
            initial_layer,
            LayerSpec(
                EmbeddingPipe,
                self.loader_util,
                self.model.embed_tokens,
                self.model,
                embedding_on_cpu=embedding_on_cpu,
                _estimated_size=1 if embedding_on_cpu else embedding_relative_size,
            ),
        ]
        for block in self.model.layers:
            result.append(LayerSpec(LlamaDecoderLayerPipe, self.loader_util, block))
        result.append(LayerSpec(LlamaRMSNormPipe, self.loader_util, self.model.norm, _estimated_size=0))
        result.append(
            LayerSpec(
                LmHeadPipe,
                self.loader_util,
                self.lm_head,
                final_logit_softcapping=self.final_logit_softcapping,
                tie_weights='model.embed_tokens.weight'
            )
        )
        result.append(LayerSpec(ComputeMetrics, _estimated_size=embedding_relative_size))
        return result