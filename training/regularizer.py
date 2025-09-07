from deepspeed import comm as dist
import torch

from constants import DEFAULT_CONTROL_ADAPTER_GAMMA

class Regularizer:
    """
    Handles regularization for LoRA and Control Adapter parameters.

    Behavior:
    - Apply analytic, in-place regularization updates for LoRA or Control Adapters
    - Compute per-rank statistics (norms, residuals, decay deltas)
    - Aggregate statistics across pipeline stages via DeepSpeed collectives

    Usage:
    - Call apply_regularization(model, config, lr) once per training step
    - Path selection is based on config['use_control_adapters']
    - Returns a dict of aggregated metrics (avg/min/max) per key
    """

    def __init__(self, pipeline_engine):
        """
        Initialize with the model engine (for device and pipeline reductions).
        """
        self.pipeline_engine = pipeline_engine

    def apply_regularization(self, model, config, lr):
        """Apply regularization to LoRA or Control Adapter parameters and return aggregated statistics.

        Behavior:
            - If config['use_control_adapters'] is True, applies control adapter regularization
              (orthogonality maintenance on Q and L2 shrinkage on S in log-space).
            - Otherwise, applies LoRA regularization (L2 on composite W = B·A).
            - Local stats are computed and then aggregated across pipeline stages.

        Returns:
            dict[str, float]: Global summary metrics containing avg/min/max for each reported key.
        """
        use_control_adapters = config.get('use_control_adapters', False)
        lora_weight_decay = config.get('lora_weight_decay', 0.0)

        if use_control_adapters:
            gamma = config.get('control_adapter_gamma', DEFAULT_CONTROL_ADAPTER_GAMMA)
            local_stats = self._apply_control_adapter_regularization_local(
                model=model,
                lr=lr,
                gamma=gamma,
                lora_weight_decay=lora_weight_decay,
            )
        else:
            local_stats = self._apply_lora_regularization_local(
                model=model,
                lr=lr,
                lora_weight_decay=lora_weight_decay,
            )

        return self._aggregate_statistics(local_stats)

    def _apply_lora_regularization_local(self, model, lr, lora_weight_decay):
        """Apply L2 regularization to the composite matrix W = B·A using L = ½||W||_F².

        Returns:
            dict with:
              - 'norms'         : tensor of Frobenius norms (||W||_F) per pair of LoRA parameters
              - 'weight_decay'  : tensor of Frobenius norm reductions due to the regularization (if applied)

        Notes:
            - Weight decay gradients are computed analytically and applied in-place to A and B.
            - Requires A and B parameters to be float32 for numerical stability when lora_weight_decay > 0.
            - Empty tensor(s) returned if no applicable parameters on this stage.
        """
        norms = []
        weight_decay = []

        assert lora_weight_decay >= 0, f"lora_weight_decay ({lora_weight_decay}) must be >= 0"

        for name, param in model.named_parameters():
            if 'lora_A' in name:
                A = param
                B_name = name.replace('lora_A', 'lora_B')

                try:
                    B = next(p for n, p in model.named_parameters() if n == B_name)
                except StopIteration:
                    raise RuntimeError(f"Could not find corresponding B parameter '{B_name}' for A parameter '{name}'")

                with torch.no_grad():
                    W = B @ A
                    W_norm = W.norm().item()

                # L2-norm regularisation of the composite matrix using L = ½||W||_F²:
                if lora_weight_decay > 0:
                    # The very tiny values end up cancelling out to zero for float16/bfloat16 and decays all stay zero...
                    assert A.dtype == torch.float32, f"LoRA A ({A.dtype}) must be float32"
                    assert B.dtype == torch.float32, f"LoRA B ({B.dtype}) must be float32"

                    # Save the initial norm so we can calculate the weight decay after the (optional) updates
                    W_norm_initial = W_norm

                    with torch.no_grad():
                        # ∂L/∂W = W, as ∂(½||W||_F²)/∂W = W
                        # ∂L/∂A = Bᵗ(∂L/∂W) = BᵗW
                        # ∂L/∂B = (∂L/∂W)Aᵗ = WAᵗ
                        grad_A = B.t() @ W
                        grad_B = W @ A.t()

                        # Modify the tensors in place
                        A.sub_(lr * lora_weight_decay * grad_A)
                        B.sub_(lr * lora_weight_decay * grad_B)

                        # Recompute W and its norm using updated A and B
                        W = B @ A
                        W_norm = W.norm().item()

                    # Save weight decay
                    weight_decay.append(W_norm_initial - W_norm)

                # Save the norms after the (optional) weight-decay update
                norms.append(W_norm)

        # Convert to tensors, handling empty case
        if len(norms) > 0:
            norms_tensor = torch.tensor(norms, dtype=torch.float32, device=self.pipeline_engine.device)
        else:
            norms_tensor = torch.empty(0, dtype=torch.float32, device=self.pipeline_engine.device)

        # Only return weight decay if regularization was applied
        if lora_weight_decay > 0:
            # Convert to tensors, handling empty case
            if len(weight_decay) > 0:
                weight_decay_tensor = torch.tensor(weight_decay, dtype=torch.float32, device=self.pipeline_engine.device)
            else:
                weight_decay_tensor = torch.empty(0, dtype=torch.float32, device=self.pipeline_engine.device)
            return {
                'norms': norms_tensor,
                'weight_decay': weight_decay_tensor
            }
        else:
            return {
                'norms': norms_tensor
            }

    def _apply_control_adapter_regularization_local(self, model, lr, gamma, lora_weight_decay):
        """Apply orthogonality regularization to Q and weight decay to S.

        - Q orthogonality : Newton step (full or partial) to maintain Q^T Q ≈ I.
        - S shrinkage     : L2 weight decay on the log-eigenvalues S where 1 + λ = exp(S).

        Math:
          - Q orthogonalization:
            F(Q) = ||Q^T Q − I||_F², grad F = 4 Q (Q^T Q − I).
            A Newton step for F(Q) = 0 yields Q ← Q − (1/2) Q (Q^T Q − I).
            We use Q ← Q − γ Q (Q^T Q − I), with γ ∈ (0, 0.5] for stability.
          - S penalty: L_S = ½||S||² ⇒ ∂L_S/∂S = S.
            With α = lr * lora_weight_decay, the update is S ← S − αS = (1 − α)S.
            Since 1 + λ = exp(S), this implies multiplicative shrinkage:
              1 + λ ← exp((1 − α)S) = (1 + λ)^{1−α}
            So λ is symmetrically pulled toward 0 in log-space.
          - Spectral norm tracking:
            For W = Q diag(λ) Q^T, if Q^T Q = I then eig(W) = {λ_i} and ||W||₂ = max_i |λ_i|.
            With semi-orthogonality (Q^T Q ≈ I), this remains a good approximation.

        Returns:
            dict with:
              - 'norms'         : tensor of spectral norms (||W||₂ ≈ max|λ|) per layer
              - 'orthogonality' : tensor of orthogonality residuals (||Q^T Q − I||_F²)
              - 'weight_decay'  : tensor of spectral norm reductions due to S shrinkage (if applied)

         Notes:
             - Regularization gradients are computed analytically and applied in-place to Q and S.
             - Recommend Q parameters to be float32 for numerical stability, but should still work with bfloat16.
             - Requires S parameters to be float32 for numerical stability when lora_weight_decay > 0.
             - Empty tensor(s) returned if no applicable parameters on this stage.
        """
        norms = []
        orthogonality = []
        weight_decay = []

        assert gamma > 0, f"control_adapter_gamma ({gamma}) must be > 0 to maintain semi-orthogonality"
        assert gamma <= 0.5, f"control_adapter_gamma ({gamma}) must be <= 0.5 to avoid overshooting"

        assert lora_weight_decay >= 0, f"lora_weight_decay ({lora_weight_decay}) must be >= 0"

        identity_matrix = None

        for name, param in model.named_parameters():
            if 'control_Q' in name:
                Q = param

                with torch.no_grad():
                    # Reuse the same identity matrix for orthogonality regularization (all Q have same rank)
                    if identity_matrix is None:
                        identity_matrix = torch.eye(Q.size(1), device=Q.device, dtype=torch.float32)

                    # Compute Newton step: γ Q (Q^T Q − I) in fp32 for stability
                    Q_f32 = Q.float()
                    QTQ_minus_I = Q_f32.t() @ Q_f32 - identity_matrix
                    newton_step = (Q_f32 @ QTQ_minus_I).mul_(gamma)

                    # Modify the tensor in place
                    Q.sub_(newton_step.to(Q.dtype))

                    # Compute orthogonality error after the update (in fp32)
                    Q_f32 = Q.float()
                    QTQ_minus_I = Q_f32.t() @ Q_f32 - identity_matrix
                    orthogonality_error = (torch.norm(QTQ_minus_I, 'fro') ** 2).item()

                # Save orthogonality error
                orthogonality.append(orthogonality_error)

            if 'control_S' in name:
                S = param

                with torch.no_grad():
                    # Convert from log-parameterization: λ = exp(S) - 1, so 1 + λ = exp(S)
                    lambda_vec = torch.expm1(S)

                    # Calculate the spectral norm of W: ||W||₂ = max|λᵢ|
                    W_norm = torch.max(torch.abs(lambda_vec)).item()

                # L2-norm regularization of the log-eigenvalue vector S = log(λ + 1)
                if lora_weight_decay > 0:
                    # The very tiny values end up cancelling out to zero for float16/bfloat16 and decays all stay zero...
                    assert S.dtype == torch.float32, f"Control Adapter S ({S.dtype}) must be float32"

                    # Save the initial norm so we can calculate the weight decay after the (optional) updates
                    W_norm_initial = W_norm

                    with torch.no_grad():
                        # L2 regularization of S: L = ½||S||²
                        # ∂L/∂S = S
                        grad_S = S

                        # Modify the tensor in place
                        S.sub_(lr * lora_weight_decay * grad_S)

                        # Recompute lambda_vec and its norm using updated S
                        lambda_vec = torch.expm1(S)
                        W_norm = torch.max(torch.abs(lambda_vec)).item()

                    # Save weight decay
                    weight_decay.append(W_norm_initial - W_norm)

                # Save the norms after the (optional) weight-decay update
                norms.append(W_norm)

        # Convert to tensors, handling empty case
        if len(orthogonality) > 0:
            orthogonality_tensor = torch.tensor(orthogonality, dtype=torch.float32, device=self.pipeline_engine.device)
        else:
            orthogonality_tensor = torch.empty(0, dtype=torch.float32, device=self.pipeline_engine.device)

        # Convert to tensors, handling empty case
        if len(norms) > 0:
            norms_tensor = torch.tensor(norms, dtype=torch.float32, device=self.pipeline_engine.device)
        else:
            norms_tensor = torch.empty(0, dtype=torch.float32, device=self.pipeline_engine.device)

        # Only return weight decay if regularization was applied
        if lora_weight_decay > 0:
            # Convert to tensors, handling empty case
            if len(weight_decay) > 0:
                weight_decay_tensor = torch.tensor(weight_decay, dtype=torch.float32, device=self.pipeline_engine.device)
            else:
                weight_decay_tensor = torch.empty(0, dtype=torch.float32, device=self.pipeline_engine.device)
            return {
                'norms': norms_tensor,
                'orthogonality': orthogonality_tensor,
                'weight_decay': weight_decay_tensor
            }
        else:
            return {
                'norms': norms_tensor,
                'orthogonality': orthogonality_tensor
            }

    def _aggregate_statistics(self, local_stats):
        """Aggregate LoRA and Control Adapter statistics across pipeline stages and compute global statistics.

        Reductions:
            - If pipeline parallelism is enabled, SUM/MIN/MAX are computed across pipe stages
            - Empty tensors contribute neutral elements via sentinels for MIN/MAX

        Output:
            For each key in local_stats (e.g., 'norms', 'orthogonality', 'weight_decay'),
            returns three scalar entries:
                - '{key}_avg'
                - '{key}_min'
                - '{key}_max'
        """
        global_stats = {}

        for key, tensor in local_stats.items():
            if tensor.numel() > 0:
                count = torch.tensor(tensor.numel(), dtype=torch.float32, device=self.pipeline_engine.device)
                sum_val = torch.sum(tensor)
                min_val = torch.min(tensor)
                max_val = torch.max(tensor)
            else:
                count = torch.tensor(0.0, dtype=torch.float32, device=self.pipeline_engine.device)
                sum_val = torch.tensor(0.0, dtype=torch.float32, device=self.pipeline_engine.device)
                # Use sentinels so MIN/MAX reductions ignore empty ranks
                min_val = torch.tensor(float('inf'), dtype=torch.float32, device=self.pipeline_engine.device)
                max_val = torch.tensor(float('-inf'), dtype=torch.float32, device=self.pipeline_engine.device)

            # Aggregate across pipeline stages if using pipeline parallelism
            if self.pipeline_engine.is_pipe_parallel:
                pp_group = self.pipeline_engine.grid.get_pipe_parallel_group()
                dist.all_reduce(count, op=dist.ReduceOp.SUM, group=pp_group)
                dist.all_reduce(sum_val, op=dist.ReduceOp.SUM, group=pp_group)
                dist.all_reduce(min_val, op=dist.ReduceOp.MIN, group=pp_group)
                dist.all_reduce(max_val, op=dist.ReduceOp.MAX, group=pp_group)

            # Compute global statistics
            if count.item() > 0:
                global_avg = (sum_val / count).item()
                global_min = min_val.item()
                global_max = max_val.item()
            else:
                global_avg = 0
                global_min = 0
                global_max = 0

            global_stats[f'{key}_avg'] = global_avg
            global_stats[f'{key}_min'] = global_min
            global_stats[f'{key}_max'] = global_max

        return global_stats