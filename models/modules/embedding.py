from torch import nn
import torch

class EmbeddingPipe(nn.Module):
    """
    Common embedding + mask update + rotary embeddings stage.

    Behavior:
    - Loads token embedding weights (optionally kept on CPU for tied-weights fine-tuning)
    - Produces input embeddings and updates causal mask via model._update_causal_mask
    - Computes rotary embeddings (cos, sin) via the model's rotary_emb
    - Optional hidden-state scaling by sqrt(hidden_size) when normalize_embedding_sqrt=True
    - Marks float tensors requires_grad=True for DeepSpeed pipeline parallelism

    Parameters:
        module_loader            : PipelineModel loader that materializes weights and handles optional quantization
        orig                     : nn.Embedding (or equivalent) used for token embeddings
        model                    : Parent transformer model providing _update_causal_mask and rotary modules
        embedding_on_cpu         : If True, keep orig weights on CPU and only move input_ids across devices
        normalize_embedding_sqrt : If True, scale hidden_states by sqrt(hidden_size) (Gemma2 behavior)

    Inputs:
        (input_ids, attention_mask, position_ids, control_classes, labels)

    Outputs:
        (hidden_states, attention_mask, cos, sin, cache_position, control_classes, labels)
    """

    def __init__(
        self,
        module_loader,
        orig,
        model,
        embedding_on_cpu: bool=False,
        normalize_embedding_sqrt: bool=False
    ):
        super().__init__()

        if not hasattr(module_loader, "load_state_dict_into_module"):
            raise AttributeError("module_loader must define load_state_dict_into_module(self_or_module)")
        if not hasattr(model, "_update_causal_mask"):
            raise AttributeError("model must define _update_causal_mask(...)")
        if not hasattr(model, "rotary_emb"):
            raise AttributeError("model must provide rotary_emb")

        self.orig = orig

        # The original model object, e.g. LlamaModel. Use a list so the nn.Module isn't registered to this module.
        self._model = [model]

        # Extract rotary_emb as direct attribute so module.to() moves it to correct device.
        # Previously was accessed via self.model[0].rotary_emb which caused device mismatch with embedding_on_cpu=True.
        self.rotary_emb = model.rotary_emb

        # Precompute scale factor once; use Python float to avoid per-forward tensor allocs
        self.scale_factor = (float(model.config.hidden_size) ** 0.5) if normalize_embedding_sqrt else None

        # Materialize and optionally quantize weights into this module
        module_loader.load_state_dict_into_module(self)

        # For tied weights case: keep embedding weights on CPU (separate from GPU copy in LmHeadPipe)
        if embedding_on_cpu:
            self.orig.to("cpu")

    # Alias for DeepSpeed TiedLayerSpec (expects a top-level attribute name)
    @property
    def weight(self):
        return self.orig.weight

    @weight.setter
    def weight(self, v):
        self.orig.weight = v

    @property
    def model(self):
        return self._model[0]

    def forward(self, inputs):
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 5:
            raise ValueError("forward expects a 5-tuple: (input_ids, attention_mask, position_ids, control_classes, labels)")

        input_ids, attention_mask, position_ids, control_classes, labels = inputs

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

        #  Scale the hidden states if needed for Gemma2.
        if self.scale_factor is not None:
            hidden_states = hidden_states * self.scale_factor

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