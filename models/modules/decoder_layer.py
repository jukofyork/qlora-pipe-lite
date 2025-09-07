from torch import nn

class DecoderLayerPipe(nn.Module):
    """
    Single decoder block wrapper for a pipeline stage.

    Behavior:
    - Loads weights for a single HF transformer decoder layer into this stage
    - Applies the layer using precomputed rotary embeddings and causal mask
    - Supports optional dual rotary embeddings (global+local) when require_local_rotary=True
    - Passes through pipeline-carry tensors unchanged

    Parameters:
        module_loader        : Loader that materializes weights and handles optional quantization
        orig                 : The original HF decoder layer module (e.g., LlamaDecoderLayer or Gemma3DecoderLayer)
        layer_idx            : Optional index for logging/identification
        require_local_rotary : If True, expects and forwards both global and local rotary embeddings

    Inputs:
        require_local_rotary=False:
            (hidden_states, attention_mask, cos, sin, cache_position, control_classes, labels)
        require_local_rotary=True:
            (hidden_states, attention_mask, cos, sin, cos_local, sin_local, cache_position, control_classes, labels)

    Outputs:
        Mirrors the input shape with updated hidden_states.
    """

    def __init__(self, module_loader, orig, layer_idx=None, require_local_rotary: bool=False):
        super().__init__()
        self.orig = orig
        self.layer_idx = layer_idx
        self.require_local_rotary = require_local_rotary
        module_loader.load_state_dict_into_module(self)

    def forward(self, inputs):
        if not self.require_local_rotary:
            if not isinstance(inputs, (tuple, list)) or len(inputs) != 7:
                raise ValueError(
                    "DecoderLayerPipe(require_local_rotary=False) expects 7-tuple: "
                    "(hidden_states, attention_mask, cos, sin, cache_position, control_classes, labels)"
                )
            hidden_states, attention_mask, cos, sin, cache_position, control_classes, labels = inputs
            hidden_states = self.orig(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=(cos, sin),
                cache_position=cache_position,
            )[0]
            return (
                hidden_states, attention_mask, cos, sin, cache_position, control_classes, labels
            )
        else:
            if not isinstance(inputs, (tuple, list)) or len(inputs) != 9:
                raise ValueError(
                    "DecoderLayerPipe(require_local_rotary=True) expects 9-tuple: "
                    "(hidden_states, attention_mask, cos, sin, cos_local, sin_local, cache_position, control_classes, labels)"
                )
            hidden_states, attention_mask, cos, sin, cos_local, sin_local, cache_position, control_classes, labels = inputs
            hidden_states = self.orig(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings_global=(cos, sin),
                position_embeddings_local=(cos_local, sin_local),
                cache_position=cache_position,
            )[0]
            return (
                hidden_states, attention_mask, cos, sin, cos_local, sin_local, cache_position, control_classes, labels
            )