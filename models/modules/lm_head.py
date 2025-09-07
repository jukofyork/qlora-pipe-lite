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
        hidden_states = self._apply_logit_scale(hidden_states, self.logit_scale)
        # For tied weights case: uses separate GPU copy of embedding weights
        logits = self.lm_head(hidden_states)
        logits = self._apply_softcap(logits, self.final_logit_softcapping)
        return logits, labels

    @staticmethod
    def _apply_logit_scale(hidden_states, scale):
        if scale is None:
            return hidden_states
        return hidden_states * scale

    @staticmethod
    def _apply_softcap(logits, cap):
        if cap is None:
            return logits
        logits = logits / cap
        logits = torch.tanh(logits)
        logits = logits * cap
        return logits

    @staticmethod
    def make_tied_lm_head_forward(logit_scale=None, final_logit_softcapping=None, **_ignored):
        """
        Factory for a forward_fn compatible with TiedLayerSpec that reuses the EmbeddingPipe module instance.

        NOTE: Accepts extra kwargs to avoid TypeError if _get_lm_head_kwargs() grows new keys.

        The returned function has signature: (tied_module, inputs) -> (logits, labels)
        """

        def tied_lm_head_forward(tied_module, inputs):
            # inputs are (hidden_states, labels) coming from FinalNormPipe
            hidden_states, labels = inputs
            weight = tied_module.orig.weight  # [vocab, hidden]

            hidden_states = LmHeadPipe._apply_logit_scale(hidden_states, logit_scale)
            logits = torch.matmul(hidden_states, weight.t())
            logits = LmHeadPipe._apply_softcap(logits, final_logit_softcapping)

            return logits, labels

        return tied_lm_head_forward