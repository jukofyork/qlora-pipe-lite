from torch import nn
import torch

class PrepareInputsPipe(nn.Module):
    """
    Prepare inputs for pipeline processing by adding position_ids.

    Behavior:
    - Derives monotonically increasing position_ids [0..seq_len-1] per sequence
    - Appends position_ids to the input tuple expected by downstream stages

    Inputs:
        (input_ids, attention_mask, control_classes, labels)

    Outputs:
        (input_ids, attention_mask, position_ids, control_classes, labels)

    Notes:
    - position_ids are created on input_ids.device
    - Sequence length is inferred from input_ids.shape[1]
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        input_ids, attention_mask, control_classes, labels = inputs
        batch_size, seq_length = input_ids.shape[:2]
        device = input_ids.device
        position_ids = torch.arange(0, seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0)
        return input_ids, attention_mask, position_ids, control_classes, labels