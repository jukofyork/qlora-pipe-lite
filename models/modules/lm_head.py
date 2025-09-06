from torch import nn
import torch

class LmHeadPipe(nn.Module):
    """
    LM head projection stage.

    Behavior:
    - Projects hidden states to vocabulary logits via lm_head (nn.Linear)
    - Optional logit scaling and final softcapping
    - Supports tied-weight checkpoints by temporarily aliasing original_name during load

    Parameters:
        module_loader           : PipelineModel loader that materializes weights
        lm_head                 : Projection module (nn.Linear) to vocab size
        logit_scale             : Optional scalar multiplier applied to hidden states before projection
        final_logit_softcapping : Optional symmetric tanh soft cap value applied to logits
        tie_weights             : Optional original_name alias to share weights with embeddings during load

    Inputs:
        (hidden_states, labels)

    Outputs:
        (logits, labels)
    """

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