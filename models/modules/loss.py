from torch import nn
import torch

from kernels.cross_entropy_loss import fast_cross_entropy_loss

class LossPipe(nn.Module):
    """
    Causal LM loss stage.

    Behavior:
    - Shifts labels right by 1 token with -100 padding on the last position
    - Computes token-level cross-entropy using a fused/optimized kernel
    - Returns mean loss

    Inputs:
        (logits, labels)

    Outputs:
        loss (scalar tensor)

    Notes:
    - Assumes logits are [batch, seq_len, vocab] and labels are [batch, seq_len]
    """

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
            logits,  # (batch_size, seq_len, vocab_size)
            shift_labels  # (batch_size, seq_len)
        )