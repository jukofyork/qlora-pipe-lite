from torch import nn

class DecoderLayerPipe(nn.Module):
    """
    Single decoder block wrapper for a pipeline stage.

    Behavior:
    - Loads weights for a single HF transformer decoder layer into this stage
    - Applies the layer using precomputed rotary embeddings and causal mask
    - Passes through pipeline-carry tensors unchanged

    Parameters:
        module_loader : Loader that materializes weights and handles optional quantization
        orig          : The original HF decoder layer module (e.g., LlamaDecoderLayer)
        layer_idx     : Optional index for logging/identification

    Inputs:
        (hidden_states, attention_mask, cos, sin, cache_position, control_classes, labels)

    Outputs:
        (hidden_states', attention_mask, cos, sin, cache_position, control_classes, labels)
    """

    def __init__(self, module_loader, orig, layer_idx=None):
        super().__init__()
        self.orig = orig
        self.layer_idx = layer_idx
        module_loader.load_state_dict_into_module(self)

    def forward(self, inputs):
        hidden_states, attention_mask, cos, sin, cache_position, control_classes, labels = inputs
        return (
            self.orig(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=(cos, sin),
                cache_position=cache_position,
            )[0],
            attention_mask, cos, sin, cache_position, control_classes, labels,
        )

class Gemma3DecoderLayerPipe(nn.Module):
    """
    Single Gemma 3 decoder block wrapper for a pipeline stage (dual rotary embeddings).

    Behavior:
    - Loads weights for a single HF transformer decoder layer into this stage
    - Applies the layer using BOTH global and local rotary embeddings and the causal mask
    - Passes through pipeline-carry tensors unchanged

    Parameters:
        module_loader : Loader that materializes weights and handles optional quantization
        orig          : The original HF decoder layer module (Gemma3DecoderLayer)
        layer_idx     : Optional index for logging/identification

    Inputs:
        (hidden_states, attention_mask, cos_global, sin_global, cos_local, sin_local, cache_position, control_classes, labels)

    Outputs:
        (hidden_states', attention_mask, cos_global, sin_global, cos_local, sin_local, cache_position, control_classes, labels)
    """

    def __init__(self, module_loader, orig, layer_idx=None):
        super().__init__()
        self.orig = orig
        self.layer_idx = layer_idx
        module_loader.load_state_dict_into_module(self)

    def forward(self, inputs):
        hidden_states, attention_mask, cos_global, sin_global, cos_local, sin_local, cache_position, control_classes, labels = inputs
        return (
            self.orig(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings_global=(cos_global, sin_global),
                position_embeddings_local=(cos_local, sin_local),
                cache_position=cache_position,
            )[0],
            attention_mask, cos_global, sin_global, cos_local, sin_local, cache_position, control_classes, labels,
        )