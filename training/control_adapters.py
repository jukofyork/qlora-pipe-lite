import torch

def apply_control_adapters(model, layers_to_transform, adapter_rank, adapter_dropout, adapter_dtype=torch.float32):
    """
    Inject Control Adapter parameters into decoder layers and patch their forward method.

    Parameters:
        model               : Pipeline model containing DecoderLayerPipe modules (with 'orig' submodule).
        layers_to_transform : list[int] | None, layer indices to transform; None transforms all decoder layers.
        adapter_rank        : int, the rank (r) for the adapter subspace (shape: hidden_size x r).
        adapter_dropout     : float in [0, 1], dropout probability applied to residual delta before projection.
        adapter_dtype       : torch.dtype for the adapter parameters (e.g., torch.float32).

    Behavior:
        - For each selected decoder layer:
          * Adds control_dropout module (Dropout or Identity)
          * Creates trainable parameters:
              - control_Q: [hidden_size, r], orthogonally initialized
              - control_S: [r], zeros initialized (log-parameterization of λ: 1 + λ = exp(S))
          * Casts control_Q and control_S to adapter_dtype
          * Patches layer.forward with a wrapper that:
              - Computes residual delta
              - Projects into Q-subspace
              - Applies class-conditional transform using expm1(S) and its inverse
              - Masks class 0 positions to zero contribution
              - Adds adapter contribution back to residual stream
        - Sets original_name for saving compatibility
        - Sets requires_grad True for control_Q/control_S and False for other parameters on the layer
    """

    def patch_decoder_layer_control_adapter(module):

        def control_adapter_forward(inputs):
            hidden_states, attention_mask, cos, sin, cache_position, control_classes, labels = inputs

            # Shift control_classes for causal LM: [control_classes[1:], 0_padding]
            batch_size, seq_len = control_classes.shape
            shift_control_classes = torch.cat([
                control_classes[:, 1:],
                torch.full((batch_size, 1), 0, device=control_classes.device, dtype=control_classes.dtype)
            ], dim=1)

            # Save input for residual computation
            input_hidden_states = hidden_states

            layer_output = module.orig(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=(cos, sin),
                cache_position=cache_position
            )[0]
            torch_result_dtype = layer_output.dtype

            # Compute residual delta, apply optional dropout, then cast to adapter dtype
            layer_delta = layer_output - input_hidden_states
            x = module.control_dropout(layer_delta).to(module.control_Q.dtype)

            # Project to adapter subspace via Q (semi-orthogonal columns: Q^T Q ≈ I_r)
            x_q = x @ module.control_Q  # [batch_size, seq_len, adapter_rank]

            # Per-rank eigenvalue offsets in the Q-subspace using a log-parameterization:
            # λ = exp(S) - 1, so 1 + λ = exp(S) > 0 and the forward operator is I + diag(λ)
            lambda_pos_coeff = torch.expm1(module.control_S)  # [adapter_rank]

            # Exact inverse for the -1 branch (exact if Q is orthonormal; otherwise approximate):
            # (I + Q diag(λ) Q^T)^{-1} - I = -Q diag(λ/(1+λ)) Q^T
            # Identity: λ' = -λ/(1+λ) = exp(-S) - 1
            lambda_neg_coeff = torch.expm1(-module.control_S)  # [adapter_rank]

            # +1 class delta: Q diag(λ) Q^T x
            out_pos = ((x_q * lambda_pos_coeff) @ module.control_Q.T)  # [batch_size, seq_len, hidden_size]

            # -1 class delta (exact inverse): Q diag(λ') Q^T x
            out_neg = ((x_q * lambda_neg_coeff) @ module.control_Q.T)  # [batch_size, seq_len, hidden_size]

            # Select +1 branch where class==1; otherwise use inverse branch (class -1)
            class1_mask = (shift_control_classes == 1).unsqueeze(-1)  # broadcast to [batch_size, seq_len, 1]
            adapter_output = torch.where(class1_mask, out_pos, out_neg)

            # 0 class: zero-out (no loss; labels = -100)
            class0_mask = (shift_control_classes == 0).unsqueeze(-1)  # broadcast to [batch_size, seq_len, 1]
            adapter_output = torch.where(class0_mask, torch.zeros_like(adapter_output), adapter_output)

            # Cast adapter contribution back to original dtype and add to the residual stream
            result = layer_output + adapter_output.to(torch_result_dtype)  # [batch_size, seq_len, hidden_size]

            return (result, attention_mask, cos, sin, cache_position, control_classes, labels)

        return control_adapter_forward

    for name, module in model.named_modules():
        if 'decoderlayerpipe' in module.__class__.__name__.lower():
            layer_idx = module.layer_idx
            should_transform = (layers_to_transform is None or layer_idx in layers_to_transform)

            if should_transform:
                device = next(module.orig.parameters()).device
                hidden_size = module.orig.hidden_size

                # Add dropout
                module.control_dropout = torch.nn.Dropout(p=adapter_dropout) if adapter_dropout > 0 else torch.nn.Identity()
                module.control_dropout = module.control_dropout.to(device)

                # Create Control Adapter parameters
                module.control_Q = torch.nn.Parameter(torch.empty(hidden_size, adapter_rank, device=device))
                module.control_S = torch.nn.Parameter(torch.empty(adapter_rank, device=device))

                # Add original_name for saving compatibility
                module.control_Q.original_name = f"base_model.model.model.layers.{layer_idx}.control_Q"
                module.control_S.original_name = f"base_model.model.model.layers.{layer_idx}.control_S"

                # Initialize
                torch.nn.init.orthogonal_(module.control_Q)
                torch.nn.init.zeros_(module.control_S)

                # Cast to the desired dtype
                module.control_Q.data = module.control_Q.data.to(adapter_dtype)
                module.control_S.data = module.control_S.data.to(adapter_dtype)

                # Replace forward with Control Adapter version
                module.forward = patch_decoder_layer_control_adapter(module)

    # Set original_name for all parameters (for saving compatibility)
    for name, p in model.named_parameters():
        if not hasattr(p, 'original_name'):
            p.original_name = name

    # Disable gradients for all base model parameters, enable only for Control Adapters
    for name, p in model.named_parameters():
        if 'control_Q' in name or 'control_S' in name:
            p.requires_grad = True
        else:
            p.requires_grad = False