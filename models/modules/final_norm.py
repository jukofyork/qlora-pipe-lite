from torch import nn

class FinalNormPipe(nn.Module):
    """
    Final normalization stage wrapper.

    Behavior:
    - Applies the model's final RMSNorm (or equivalent) to hidden_states
    - Reduces the tuple to (hidden_states, labels) for the head stage

    Parameters:
        module_loader : PipelineModel loader that materializes weights
        orig          : Normalization module (e.g., LlamaRMSNorm)

    Inputs:
        (hidden_states, *_, labels)

    Outputs:
        (hidden_states, labels)
    """

    def __init__(self, module_loader, orig):
        super().__init__()
        self.orig = orig
        module_loader.load_state_dict_into_module(self)

    def forward(self, inputs):
        hidden_states, *_, labels = inputs
        return self.orig(hidden_states), labels