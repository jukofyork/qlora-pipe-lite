from torch import nn
import torch

class EmbeddingPipe(nn.Module):
    """
    Common embedding + mask update + (optional) dual rotary embeddings stage.

    Behavior:
    - Loads token embedding weights (optionally kept on CPU for tied-weights fine-tuning)
    - Produces input embeddings and updates causal mask via model._update_causal_mask
    - Computes rotary embeddings (cos, sin) via the model's rotary_emb
    - When require_local_rotary=True, also computes local rotary (cos, sin) via model.rotary_emb_local
    - Optional hidden-state scaling by sqrt(hidden_size) when normalize_embedding_sqrt=True
    - Marks float tensors requires_grad=True for DeepSpeed pipeline parallelism

    Parameters:
        module_loader            : PipelineModel loader that materializes weights and handles optional quantization
        orig                     : nn.Embedding (or equivalent) used for token embeddings
        model                    : Parent transformer model providing _update_causal_mask and rotary modules
        embedding_on_cpu         : If True, keep orig weights on CPU and only move input_ids across devices
        normalize_embedding_sqrt : If True, scale hidden_states by sqrt(hidden_size) (Gemma2/Gemma3 behavior)
        require_local_rotary     : If True, also uses model.rotary_emb_local to return a second (cos, sin) pair

    Inputs:
        (input_ids, attention_mask, position_ids, control_classes, labels)

    Outputs:
        When require_local_rotary=False:
          (hidden_states, attention_mask, cos, sin, cache_position, control_classes, labels)
        When require_local_rotary=True:
          (hidden_states, attention_mask, cos_global, sin_global, cos_local, sin_local, cache_position, control_classes, labels)
    """

    def __init__(
        self,
        module_loader,
        orig,
        model,
        embedding_on_cpu: bool=False,
        normalize_embedding_sqrt: bool=False,
        require_local_rotary: bool=False,
    ):
        super().__init__()

        # Fail-fast validation
        if not hasattr(module_loader, "load_state_dict_into_module"):
            raise AttributeError("module_loader must define load_state_dict_into_module(self_or_module)")
        if not hasattr(model, "_update_causal_mask"):
            raise AttributeError("model must define _update_causal_mask(...)")
        if not hasattr(model, "rotary_emb"):
            raise AttributeError("model must provide rotary_emb")
        if require_local_rotary and not hasattr(model, "rotary_emb_local"):
            raise AttributeError("require_local_rotary=True requires model.rotary_emb_local")

        self.orig = orig
        # Keep reference without registering the full model as a submodule
        self._model = [model]

        # Expose rotary modules so module.to() moves them properly with this module
        self.rotary_emb = model.rotary_emb
        if require_local_rotary:
            self.rotary_emb_local = model.rotary_emb_local

        self.normalize_embedding_sqrt = normalize_embedding_sqrt
        self.require_local_rotary = require_local_rotary

        # Materialize and optionally quantize weights into this module
        module_loader.load_state_dict_into_module(self)

        # For tied weights case: keep embedding weights on CPU (separate from GPU copy in LmHeadPipe)
        if embedding_on_cpu:
            self.orig.to("cpu")

    @property
    def model(self):
        return self._model[0]

    def _common_preamble(self, input_ids, attention_mask, position_ids, control_classes, labels):
        # For tied weights case: only input_ids move CPU<->GPU, not the large embedding weights
        if self.orig.weight.device.type == "cpu":
            original_device = input_ids.device
            inputs_embeds = self.orig(input_ids.to("cpu")).to(original_device)
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
            normalizer = torch.tensor(self.model.config.hidden_size ** 0.5, dtype=hidden_states.dtype, device=hidden_states.device)
            hidden_states = hidden_states * normalizer

        return hidden_states, attention_mask, position_ids, cache_position, control_classes, labels

    def _compute_rotary_outputs(self, hidden_states, position_ids):
        # Compute rotary embeddings - using direct attribute (moved to correct device by module.to())
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        if self.require_local_rotary:
            cos_local, sin_local = self.rotary_emb_local(hidden_states, position_ids)
            return cos, sin, cos_local, sin_local
        return cos, sin

    def forward(self, inputs):
        # Fail-fast boundary validation
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 5:
            raise ValueError("forward expects a 5-tuple: (input_ids, attention_mask, position_ids, control_classes, labels)")

        input_ids, attention_mask, position_ids, control_classes, labels = inputs

        hidden_states, attention_mask, position_ids, cache_position, control_classes, labels = self._common_preamble(
            input_ids, attention_mask, position_ids, control_classes, labels
        )
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
            labels,
        )