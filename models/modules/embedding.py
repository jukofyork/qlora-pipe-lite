# models/modules/embedding.py
from torch import nn
import torch

class _BaseEmbeddingPipe(nn.Module):
    """
    Common embedding + mask update + cache_position logic for embedding stages.

    Behavior:
    - Loads token embedding weights (optionally kept on CPU for tied-weights fine-tuning)
    - Produces input embeddings and updates causal mask via model._update_causal_mask
    - Applies Gemma2/Gemma3 hidden-state scaling by sqrt(hidden_size) to match HF behavior
    - Provides rotary outputs via subclass hook and marks float tensors requires_grad for DS pipeline

    Parameters:
        module_loader        : PipelineModel loader that materializes weights and handles optional quantization
        orig                 : nn.Embedding (or equivalent) used for token embeddings
        model                : Parent transformer model providing _update_causal_mask and rotary modules
        embedding_on_cpu     : If True, keep orig weights on CPU and only move input_ids across devices
        normalize_embedding_sqrt : If True, scale hidden_states by sqrt(hidden_size) (Gemma2/Gemma3 behavior)
        require_local_rotary : If True, also exposes model.rotary_emb_local as attribute

    Inputs:
        (input_ids, attention_mask, position_ids, control_classes, labels)

    Outputs (prefix):
        (hidden_states, attention_mask, <subclass rotary outputs...>, cache_position, control_classes, labels)
    """

    def __init__(self, module_loader, orig, model, embedding_on_cpu=False, require_local_rotary=False, normalize_embedding_sqrt=False):
        super().__init__()
        self.orig = orig
        # The original model object, e.g. LlamaModel. Use a list so the nn.Module isn't registered to this module.
        self._model = [model]
        # Extract rotary_emb as direct attribute so module.to() moves it to correct device.
        self.rotary_emb = model.rotary_emb
        if require_local_rotary:
            self.rotary_emb_local = model.rotary_emb_local
        self.normalize_embedding_sqrt = normalize_embedding_sqrt
        module_loader.load_state_dict_into_module(self)
        # For tied weights case: keep embedding weights on CPU (separate from GPU copy in LmHeadPipe)
        if embedding_on_cpu:
            self.orig.to('cpu')

    @property
    def model(self):
        return self._model[0]

    def _common_preamble(self, inputs):
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
        if self.normalize_embedding_sqrt:
            normalizer = torch.tensor(self.model.config.hidden_size ** 0.5, dtype=hidden_states.dtype)
            hidden_states = hidden_states * normalizer

        return hidden_states, attention_mask, position_ids, cache_position, control_classes, labels

    def _compute_rotary_outputs(self, hidden_states, position_ids):
        """Subclasses must return a tuple/list of rotary tensors to pass downstream."""
        raise NotImplementedError()

    def forward(self, inputs):
        hidden_states, attention_mask, position_ids, cache_position, control_classes, labels = self._common_preamble(inputs)
        rotary_outputs = self._compute_rotary_outputs(hidden_states, position_ids)

        # DeepSpeed pipeline parallelism requires_grad workarounds:
        for tensor in [hidden_states, attention_mask, *rotary_outputs]:
            if torch.is_floating_point(tensor):
                tensor.requires_grad_(True)

        return (
            hidden_states,
            attention_mask,
            *rotary_outputs,
            cache_position,
            control_classes,
            labels
        )

class EmbeddingPipe(_BaseEmbeddingPipe):
    """
    Embedding + causal mask update + rotary embeddings stage.

    Behavior:
    - Loads token embedding weights (optionally kept on CPU for tied-weights fine-tuning)
    - Produces input embeddings and updates causal mask via model._update_causal_mask
    - Computes rotary embeddings (cos, sin) via the model's rotary_emb
    - Optional hidden-state scaling by sqrt(hidden_size) when normalize_embedding_sqrt=True
    """

    def __init__(self, module_loader, orig, model, embedding_on_cpu=False, normalize_embedding_sqrt=False):
        super().__init__(
            module_loader, orig, model,
            embedding_on_cpu=embedding_on_cpu, require_local_rotary=False, normalize_embedding_sqrt=normalize_embedding_sqrt
        )

    def _compute_rotary_outputs(self, hidden_states, position_ids):
        # Compute rotary embeddings - using direct attribute (moved to correct device by module.to())
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        return cos, sin

class Gemma3EmbeddingPipe(_BaseEmbeddingPipe):
    """
    Gemma 3 embedding + causal mask update + dual rotary embeddings stage.

    Behavior:
    - Computes BOTH global and local rotary embeddings (cos, sin) pairs
    """

    def __init__(self, module_loader, orig, model, embedding_on_cpu=False, normalize_embedding_sqrt=False):
        super().__init__(
            module_loader, orig, model,
            embedding_on_cpu=embedding_on_cpu,
            require_local_rotary=True,
            normalize_embedding_sqrt=normalize_embedding_sqrt
        )

    def _compute_rotary_outputs(self, hidden_states, position_ids):
        cos_global, sin_global = self.rotary_emb(hidden_states, position_ids)
        cos_local, sin_local = self.rotary_emb_local(hidden_states, position_ids)
        return cos_global, sin_global, cos_local, sin_local