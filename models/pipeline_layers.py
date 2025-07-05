from deepspeed.runtime.pipe import module as pipe_module
from torch import nn
import torch

from kernels.cross_entropy_loss import fast_cross_entropy_loss

class LayerSpec(pipe_module.LayerSpec):

    def __init__(self, typename, *module_args, **module_kwargs):
        super().__init__(typename, *module_args, **module_kwargs)

    def build(self):
        self.module_kwargs.pop('_estimated_size', None)
        return self.typename(*self.module_args, **self.module_kwargs)

    @property
    def estimated_size(self):
        return self.module_kwargs.get('_estimated_size', 1)

class PrepareInputsPipe(nn.Module):
    """Adds position_ids to the input tuple for pipeline processing."""

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        input_ids, attention_mask, labels, sample_weights = inputs
        batch_size, seq_length = input_ids.shape[:2]
        device = input_ids.device
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0)
        return input_ids, attention_mask, position_ids, labels, sample_weights

class EmbeddingPipe(nn.Module):

    def __init__(self, loader_util, orig, model, embedding_on_cpu=False):
        super().__init__()
        self.orig = orig
        # The original model object, e.g. LlamaModel. Use a list so the nn.Module isn't registered to this module.
        self.model = [model]
        self.embedding_on_cpu = embedding_on_cpu
        loader_util.load_state_dict_into_module(self)

    def forward(self, inputs):
        input_ids, attention_mask, position_ids, labels, sample_weights = inputs
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
            normalizer = torch.tensor(self.model[0].config.hidden_size ** 0.5, dtype=hidden_states.dtype)
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
        if torch.is_floating_point(sample_weights):
            sample_weights.requires_grad_(True)
        return hidden_states, attention_mask, cos, sin, labels, sample_weights

class LlamaDecoderLayerPipe(nn.Module):

    def __init__(self, loader_util, orig):
        super().__init__()
        self.orig = orig
        loader_util.load_state_dict_into_module(self)

    def forward(self, inputs):
        hidden_states, attention_mask, cos, sin, labels, sample_weights = inputs
        result = (
            self.orig(hidden_states, attention_mask=attention_mask, position_embeddings=(cos, sin))[0],
            attention_mask, cos, sin, labels, sample_weights
        )
        return result

class LlamaRMSNormPipe(nn.Module):

    def __init__(self, loader_util, orig):
        super().__init__()
        self.orig = orig
        loader_util.load_state_dict_into_module(self)

    def forward(self, inputs):
        hidden_states, _, _, _, labels, sample_weights = inputs
        return self.orig(hidden_states), labels, sample_weights

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
        hidden_states, labels, sample_weights = inputs
        if self.logit_scale is not None:
            hidden_states = hidden_states * self.logit_scale
        logits = self.lm_head(hidden_states)
        if self.final_logit_softcapping is not None:
            logits = logits / self.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.final_logit_softcapping
        return logits, labels, sample_weights

class ComputeMetrics(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        logits, labels, sample_weights = inputs
        batch_size, seq_len, vocab_size = logits.shape

        # Shift labels for causal LM: [labels[1:], -100_padding]
        shift_labels = torch.cat([
            labels[:, 1:],
            torch.full((batch_size, 1), -100, device=labels.device, dtype=labels.dtype)
        ], dim=1)

        # Use absolute values for loss (sign was used for Control Adapter negate)
        abs_sample_weights = torch.abs(sample_weights)

        # Return the mean loss for the batch
        return fast_cross_entropy_loss(
            logits,  # (batch_size, seq_len, vocab_size)
            shift_labels,  # (batch_size, seq_len)
            abs_sample_weights,  # (batch_size, seq_len)
        )