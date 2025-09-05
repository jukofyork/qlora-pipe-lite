from torch import nn
import torch

from kernels.cross_entropy_loss import fast_cross_entropy_loss

class PrepareInputsPipe(nn.Module):
    """Adds position_ids to the input tuple for pipeline processing."""

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        input_ids, attention_mask, control_classes, labels = inputs
        batch_size, seq_length = input_ids.shape[:2]
        device = input_ids.device
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0)
        return input_ids, attention_mask, position_ids, control_classes, labels

class EmbeddingPipe(nn.Module):

    def __init__(self, module_loader, orig, model, embedding_on_cpu=False):
        super().__init__()
        self.orig = orig
        # The original model object, e.g. LlamaModel. Use a list so the nn.Module isn't registered to this module.
        self._model = [model]
        # Extract rotary_emb as direct attribute so module.to() moves it to correct device.
        # Previously was accessed via self.model[0].rotary_emb which caused device mismatch with embedding_on_cpu=True.
        self.rotary_emb = model.rotary_emb
        module_loader.load_state_dict_into_module(self)
        # For tied weights case: keep embedding weights on CPU (separate from GPU copy in LmHeadPipe)
        if embedding_on_cpu:
            self.orig.to('cpu')

    @property
    def model(self):
        return self._model[0]

    def forward(self, inputs):
        input_ids, attention_mask, position_ids, control_classes, labels = inputs
        # For tied weights case: only input_ids move CPU<->GPU, not the large embedding weights
        if self.orig.weight.device.type == 'cpu':
            original_device = input_ids.device
            input_ids = input_ids.to('cpu')
            inputs_embeds = self.orig(input_ids).to(original_device)
        else:
            inputs_embeds = self.orig(input_ids)

        original_attention_mask = attention_mask
        past_key_values = None  # always None for training
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        attention_mask = self.model._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, False
        )
        if attention_mask is None:
            # With FA, attention_mask can end up being None. But with deepspeed we can't pass None
            # between GPUs. So force it back to the original attention_mask.
            attention_mask = original_attention_mask

        hidden_states = inputs_embeds
        if self.model.config.model_type == 'gemma2':
            normalizer = torch.tensor(self.model.config.hidden_size ** 0.5, dtype=hidden_states.dtype)
            hidden_states = hidden_states * normalizer

        # Compute rotary embeddings - using direct attribute (moved to correct device by module.to())
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        # DeepSpeed pipeline parallelism requires_grad workarounds:
        # 1. hidden_states: Required for activation checkpointing with reentrant checkpoint function (default).
        #    Could use non-reentrant instead, but has memory usage bug with flash attention.
        # 2. Other floating point tensors: Required when sending between pipeline stages.
        #    Theoretically unnecessary, but pipeline parallel hangs without this.
        for tensor in [hidden_states, attention_mask, cos, sin]:
            if torch.is_floating_point(tensor):
                tensor.requires_grad_(True)

        return hidden_states, attention_mask, cos, sin, cache_position, control_classes, labels

class LlamaDecoderLayerPipe(nn.Module):

    def __init__(self, module_loader, orig, layer_idx=None):
        super().__init__()
        self.orig = orig
        self.layer_idx = layer_idx
        module_loader.load_state_dict_into_module(self)

    def forward(self, inputs):
        hidden_states, attention_mask, cos, sin, cache_position, control_classes, labels = inputs
        result = (
            self.orig(hidden_states, attention_mask=attention_mask, position_embeddings=(cos, sin), cache_position=cache_position)[0],
            attention_mask, cos, sin, cache_position, control_classes, labels
        )
        return result

class LlamaRMSNormPipe(nn.Module):

    def __init__(self, module_loader, orig):
        super().__init__()
        self.orig = orig
        module_loader.load_state_dict_into_module(self)

    def forward(self, inputs):
        hidden_states, _, _, _, _, _, labels = inputs
        return self.orig(hidden_states), labels

class LmHeadPipe(nn.Module):

    def __init__(self, module_loader, lm_head, logit_scale=None, final_logit_softcapping=None, tie_weights=None):
        super().__init__()
        # Unlike the other wrapper classes, this is called lm_head and not orig. Because this is directly a
        # nn.Linear layer, it needs to keep the same attribute name so quantization knows not to quantize it.
        self.lm_head = lm_head
        self.logit_scale = logit_scale
        self.final_logit_softcapping = final_logit_softcapping
        # For tied weights case (checkpoint may omit lm_head.weight), temporarily alias
        # original_name to embedding for loading only, then restore canonical name for saving.
        canonical_name = getattr(self.lm_head.weight, 'original_name', 'lm_head.weight')
        if tie_weights:
            self.lm_head.weight.original_name = tie_weights

        module_loader.load_state_dict_into_module(self)

        # Restore canonical name in non-quantized modes (quantized modes strip attributes)
        if tie_weights and getattr(module_loader, 'quantization_config', None) is None:
            # Only restore if attribute still exists (it is removed for BNB 4-bit)
            try:
                self.lm_head.weight.original_name = canonical_name
            except Exception:
                pass

    def forward(self, inputs):
        hidden_states, labels = inputs
        if self.logit_scale is not None:
            hidden_states = hidden_states * self.logit_scale
        # For tied weights case: uses separate GPU copy of embedding weights
        logits = self.lm_head(hidden_states)
        if self.final_logit_softcapping is not None:
            logits = logits / self.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.final_logit_softcapping
        return logits, labels

class ComputeLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        logits, labels = inputs

        batch_size, seq_len, vocab_size = logits.shape

        # Shift labels for causal LM: [labels[1:], -100_padding]
        shift_labels = torch.cat([
            labels[:, 1:],
            torch.full((batch_size, 1), -100, device=labels.device, dtype=labels.dtype)
        ], dim=1)

        # Return mean loss (fast_cross_entropy_loss should compute mean)
        return fast_cross_entropy_loss(
            logits,       # (batch_size, seq_len, vocab_size)
            shift_labels  # (batch_size, seq_len)
        )
