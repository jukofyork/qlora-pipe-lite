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
        input_ids, attention_mask, control_class, labels = inputs
        batch_size, seq_length = input_ids.shape[:2]
        device = input_ids.device
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0)
        return input_ids, attention_mask, position_ids, control_class, labels

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
        input_ids, attention_mask, position_ids, control_class, labels = inputs
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

        return hidden_states, attention_mask, cos, sin, control_class, labels

class LlamaDecoderLayerPipe(nn.Module):

    def __init__(self, module_loader, orig, layer_idx=None):
        super().__init__()
        self.orig = orig
        self.layer_idx = layer_idx
        module_loader.load_state_dict_into_module(self)

    def forward(self, inputs):
        hidden_states, attention_mask, cos, sin, control_class, labels = inputs
        result = (
            self.orig(hidden_states, attention_mask=attention_mask, position_embeddings=(cos, sin))[0],
            attention_mask, cos, sin, control_class, labels
        )
        return result

class LlamaRMSNormPipe(nn.Module):

    def __init__(self, module_loader, orig):
        super().__init__()
        self.orig = orig
        module_loader.load_state_dict_into_module(self)

    def forward(self, inputs):
        hidden_states, _, _, _, control_class, labels = inputs
        return self.orig(hidden_states), control_class, labels

class LmHeadPipe(nn.Module):

    def __init__(self, module_loader, lm_head, logit_scale=None, final_logit_softcapping=None, tie_weights=None):
        super().__init__()
        # Unlike the other wrapper classes, this is called lm_head and not orig. Because this is directly a
        # nn.Linear layer, it needs to keep the same attribute name so quantization knows not to quantize it.
        self.lm_head = lm_head
        self.logit_scale = logit_scale
        self.final_logit_softcapping = final_logit_softcapping
        # For tied weights case, the same checkpoint weights are loaded into both EmbeddingPipe and LmHeadPipe
        if tie_weights:
            self.lm_head.weight.original_name = tie_weights
        module_loader.load_state_dict_into_module(self)

    def forward(self, inputs):
        hidden_states, control_class, labels = inputs
        if self.logit_scale is not None:
            hidden_states = hidden_states * self.logit_scale
        # For tied weights case: uses separate GPU copy of embedding weights
        logits = self.lm_head(hidden_states)
        if self.final_logit_softcapping is not None:
            logits = logits / self.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.final_logit_softcapping
        return logits, control_class, labels

class ComputeMetrics(nn.Module):

    def __init__(self, symmetry_lambda: float=0.0, eps: float=1e-8):
        """
        symmetry_lambda – λ in CE₊ + CE₋ + λ (Δ/S)²
                          set 0.0 to disable (ordinary training)
        eps             – numerical guard for division by zero
        """
        super().__init__()
        self.symmetry_lambda = symmetry_lambda
        self.eps = eps

    def forward(self, inputs):
        logits, control_class, labels = inputs
        batch_size, seq_len, _ = logits.shape

        # Shift labels for causal LM: [labels[1:], -100_padding]
        shift_labels = torch.cat([
            labels[:, 1:],
            torch.full((batch_size, 1), -100, device=labels.device, dtype=labels.dtype)
        ], dim=1)

        # Calculate the mean top1 accuracy for the batch.
        with torch.no_grad():
            mask_all = shift_labels != -100
            assert mask_all.any(), "All labels are masked (-100), so no valid targets for top1_accuracy calculation"
            top1_accuracy = (torch.argmax(logits, -1) == shift_labels).masked_select(mask_all).float().mean()

        # Split the batch into +1 / −1 classes
        # control_class has shape (batch,) -> expand to (batch, seq_len)
        mask_pos = (control_class == 1).unsqueeze(1).expand(-1, seq_len)
        mask_neg = (control_class == -1).unsqueeze(1).expand(-1, seq_len)

        def class_ce(mask):
            # Clone & mask labels; skip call if class absent (avoids Triton assert)
            lab = shift_labels.clone()
            lab[~mask] = -100
            if torch.count_nonzero(lab != -100) == 0:
                return logits.new_zeros(())  # scalar 0.0, no grad
            return fast_cross_entropy_loss(logits, lab)  # mean over class tokens

        ce_pos = class_ce(mask_pos)
        ce_neg = class_ce(mask_neg)

        ce_loss_sum = ce_pos + ce_neg

        # Check if penalty should be applied (autograd-safe Python bools with short-circuit)
        apply_penalty = (self.symmetry_lambda != 0.0) and \
                        bool(torch.count_nonzero(mask_pos & (shift_labels != -100))) and \
                        bool(torch.count_nonzero(mask_neg & (shift_labels != -100)))

        if apply_penalty:
            # Symmetry penalty: λ · (Δ / S)²
            ce_loss_difference = ce_pos - ce_neg
            symmetry_penalty = self.symmetry_lambda * (ce_loss_difference / (ce_loss_sum + self.eps)) ** 2
            penalised_loss = ce_pos + ce_neg + symmetry_penalty
            return (penalised_loss, top1_accuracy, ce_pos.detach(), ce_neg.detach(), symmetry_penalty.detach())
        else:
            return (ce_loss_sum, top1_accuracy)