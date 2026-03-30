from deepspeed import comm as dist
import torch

class Regularizer:
    """
    Handles regularization for LoRA and Control Adapter parameters.

    Behavior:
    - Apply analytic, in-place regularization updates for LoRA or Control Adapters
    - Compute per-rank statistics (norms, residuals, decay deltas)
    - Aggregate statistics across pipeline stages via DeepSpeed collectives

    Usage:
    - Call apply_regularization(model, config, lr) once per training step
    - Applies L2 weight decay to composite W = B @ A matrices
    - Returns a dict of aggregated metrics (avg/min/max) per key
    """

    def __init__(self, pipeline_engine):
        """
        Initialize with the model engine (for device and pipeline reductions).
        """
        self.pipeline_engine = pipeline_engine

    def apply_regularization(self, model, config, lr):
        """Apply L2 regularization to LoRA or Control Adapter composite matrices and return aggregated statistics.

        Behavior:
            - Searches for both lora_A/lora_B and control_A/control_B parameter pairs
            - Applies L2 weight decay to composite W = B @ A using L = ½||W||_F²
            - Local stats are computed and then aggregated across pipeline stages

        Returns:
            dict[str, float]: Global summary metrics containing avg/min/max for each reported key.
        """
        lora_weight_decay = config.get('lora_weight_decay', 0.0)
        assert lora_weight_decay >= 0, f"lora_weight_decay ({lora_weight_decay}) must be >= 0"

        norms = []
        weight_decay = []

        for name, param in model.named_parameters():
            if 'lora_A' in name or 'control_A' in name:
                A = param

                # Determine corresponding B name
                if 'lora_A' in name:
                    B_name = name.replace('lora_A', 'lora_B')
                else:  # control_A
                    B_name = name.replace('control_A', 'control_B')

                # Find B parameter
                try:
                    B = next(p for n, p in model.named_parameters() if n == B_name)
                except StopIteration:
                    raise RuntimeError(f"Could not find corresponding B parameter '{B_name}' for A parameter '{name}'")

                with torch.no_grad():
                    W = B @ A
                    W_norm = W.norm().item()

                # L2-norm regularization of the composite matrix using L = ½||W||_F²:
                if lora_weight_decay > 0:
                    # The very tiny values end up cancelling out to zero for float16/bfloat16 and decays all stay zero...
                    assert A.dtype == torch.float32, f"Adapter A ({A.dtype}) must be float32"
                    assert B.dtype == torch.float32, f"Adapter B ({B.dtype}) must be float32"

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

        # Build local stats dict
        local_stats = {'norms': norms_tensor}

        # Only include weight decay if regularization was applied
        if lora_weight_decay > 0:
            if len(weight_decay) > 0:
                weight_decay_tensor = torch.tensor(weight_decay, dtype=torch.float32, device=self.pipeline_engine.device)
            else:
                weight_decay_tensor = torch.empty(0, dtype=torch.float32, device=self.pipeline_engine.device)
            local_stats['weight_decay'] = weight_decay_tensor

        return self._aggregate_statistics(local_stats)

    def _aggregate_statistics(self, local_stats):
        """Aggregate LoRA and Control Adapter statistics across pipeline stages and compute global statistics.

        Reductions:
            - If pipeline parallelism is enabled, SUM/MIN/MAX are computed across pipe stages
            - Empty tensors contribute neutral elements via sentinels for MIN/MAX

        Output:
            For each key in local_stats (e.g., 'norms', 'weight_decay'),
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