from deepspeed import comm as dist
from torch.utils.tensorboard import SummaryWriter
import gc
import time
import torch

from constants import (
    DEFAULT_CONTROL_ADAPTER_GAMMA,
    DEFAULT_CHECKPOINT_INTERVAL_HOURS,
    DEFAULT_MAX_CHECKPOINTS,
    DEFAULT_EVALS_PER_EPOCH
)
from training.checkpoint_manager import load_checkpoint, save_checkpoint, prune_checkpoints
from training.model_saver import save_lora, save_full_model
from utils.utils import is_main_process, log

class Trainer:
    """Handles the main training loop, evaluation, checkpointing, and model saving."""

    def __init__(
        self,
        config,
        model_engine,
        train_dataloader,
        eval_dataloader,
        run_dir,
        pipeline_model,
        args,
        lora_config,
        optimizer,
        eval_gradient_accumulation_steps,
        resume_from_checkpoint=False
    ):
        self.config = config
        self.model_engine = model_engine
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.run_dir = run_dir
        self.pipeline_model = pipeline_model
        self.args = args
        self.lora_config = lora_config
        self.optimizer = optimizer
        self.resume_from_checkpoint = resume_from_checkpoint

        # Extract config values with defaults
        self.model_dir = config['model_dir']
        self.epochs = config.get('epochs', 1)
        self.eval_gradient_accumulation_steps = eval_gradient_accumulation_steps
        self.checkpoint_interval_hours = config.get('checkpoint_interval_hours', DEFAULT_CHECKPOINT_INTERVAL_HOURS)
        self.max_checkpoints = config.get('max_checkpoints', DEFAULT_MAX_CHECKPOINTS)

        self.tb_writer = SummaryWriter(log_dir=run_dir) if is_main_process() else None
        self.last_checkpoint_time = time.time() if is_main_process() else None

        # Calculate and set total training steps
        model_engine.set_dataloader(train_dataloader)
        steps_per_epoch = len(train_dataloader) // model_engine.gradient_accumulation_steps()
        model_engine.total_steps = steps_per_epoch * self.epochs

        # Calculate evaluation step indices per epoch
        evals_per_epoch = config.get('evals_per_epoch', DEFAULT_EVALS_PER_EPOCH)
        self.eval_step_indices = self._calculate_eval_steps(steps_per_epoch, evals_per_epoch)

    def train(self):
        """Main training loop."""

        last_eval_loss = None

        # Attempt to resume from checkpoint if requested
        if self.resume_from_checkpoint:
            last_eval_loss = load_checkpoint(self.model_engine, self.train_dataloader, self.run_dir)

        # Fresh run or no checkpoint found, so run initial evaluation
        if last_eval_loss is None:
            last_eval_loss = self.evaluate()
            if is_main_process():
                self._print_eval_progress(last_eval_loss)

        # Main training loop
        current_epoch = self.train_dataloader.epoch
        while self.train_dataloader.epoch <= self.epochs:
            self._cleanup_memory()

            # Forward/backward pass and optimizer step
            metrics = self.model_engine.train_batch()
            self.train_dataloader.sync_epoch()

            # Apply adapter-specific regularization and log statistics
            if self.lora_config is not None:
                stats = self._apply_regularization(
                    self.pipeline_model,
                    self.config,
                    self.optimizer.param_groups[0]['lr']
                )
                for key, value in stats.items():
                    self._write_scalar_metric(f'train/{key}', value)

            # Log training metrics
            self._write_metrics('train', metrics)
            self._write_scalar_metric('train/lr', self.optimizer.param_groups[0]['lr'])

            # Periodic evaluation
            if self.model_engine.global_steps in self.eval_step_indices:
                loss = self.evaluate()
                if is_main_process():
                    self._print_eval_progress(loss, last_eval_loss)
                last_eval_loss = loss

            if self.train_dataloader.epoch > current_epoch:
                # Always save a checkpoint at the start of each epoch
                self._checkpoint_if_needed(last_eval_loss, True)
                self._save_model(f'epoch{current_epoch}')
                current_epoch = self.train_dataloader.epoch
            else:
                # Time-based checkpointing
                self._checkpoint_if_needed(last_eval_loss)

        log('TRAINING COMPLETE!')

    def evaluate(self):
        """Run evaluation on the eval dataset and return average loss."""
        orig_micro_batches = self.model_engine.micro_batches
        self.model_engine.micro_batches = self.eval_gradient_accumulation_steps
        iterator = iter(self.eval_dataloader)
        all_metrics = None

        # Collect metrics from all eval batches
        while True:
            metrics = self.model_engine.eval_batch(iterator)
            self.eval_dataloader.sync_epoch()
            if all_metrics is None:
                all_metrics = [[] for _ in range(len(metrics))]
            if self.eval_dataloader.epoch == 2:
                break
            for i, metric in enumerate(metrics):
                all_metrics[i].append(metric)

        # Reset dataloader and restore original batch size
        self.eval_dataloader.reset()
        self.model_engine.micro_batches = orig_micro_batches
        eval_metrics = [torch.cat(metric_list) for metric_list in all_metrics]

        # Log evaluation metrics
        self._write_metrics('eval', eval_metrics)

        return self._extract_loss(eval_metrics)

    # Private helper methods
    def _calculate_eval_steps(self, steps_per_epoch, evals_per_epoch):
        """Calculate which steps to run evaluation on."""
        # Avoid requesting more evals than steps and avoid step 0
        evals = max(1, min(evals_per_epoch, steps_per_epoch))
        eval_steps = set()
        for epoch in range(self.epochs):
            for i in range(1, evals + 1):
                step_in_epoch = max(1, round(i * steps_per_epoch / evals))
                actual_step = epoch * steps_per_epoch + step_in_epoch
                eval_steps.add(actual_step)
        return eval_steps

    def _cleanup_memory(self):
        """Clean up GPU memory before training step."""
        gc.collect()
        torch.cuda.empty_cache()

    def _apply_lora_regularization_local(self, model, config, lr):
        """Apply Frobenius norm-based weight decay to *COMPOSITE* of the LoRA parameters.

        Applies L2 regularization to the composite matrix W = B·A using L = ½||W||_F².
        NOTE: Weight decay gradients are computed analytically and applied in-place to A and B.

        Returns:
            dict:
                - 'norms'        : tensor of Frobenius norms ||W||_F per LoRA parameter (empty if none)
                - 'weight_decay' : tensor of Frobenius norm reduction due to regularization
                                   (only included when regularization is configured via lora_weight_decay > 0;
                                   empty tensor if no applicable parameters on this stage)

        Notes:
            - Requires A and B parameters to be float32 for numerical stability when lora_weight_decay > 0.
        """
        norms = []
        weight_decay = []

        lora_weight_decay = config.get('lora_weight_decay', 0.0)

        for name, param in model.named_parameters():
            if 'lora_A' in name:
                A = param
                B_name = name.replace('lora_A', 'lora_B')

                try:
                    B = next(p for n, p in model.named_parameters() if n == B_name)
                except StopIteration:
                    raise RuntimeError(f"Could not find corresponding B parameter '{B_name}' for A parameter '{name}'")

                W = B @ A
                W_norm = W.norm().item()

                # L2-norm regularisation of the composite matrix using L = ½||W||_F²:
                if lora_weight_decay > 0:
                    # The very tiny values end up cancelling out to zero for float16/bfloat16 and decays all stay zero...
                    assert A.dtype == torch.float32, f"LoRA A ({A.dtype}) must be float32"
                    assert B.dtype == torch.float32, f"LoRA B ({B.dtype}) must be float32"

                    # Save the initial norm so we can calculate the weight decay after the (optional) updates
                    W_norm_initial = W_norm

                    # ∂L/∂W = W, as ∂(½||W||_F²)/∂W = W
                    # ∂L/∂A = Bᵗ(∂L/∂W) = BᵗW
                    # ∂L/∂B = (∂L/∂W)Aᵗ = WAᵗ
                    grad_A = B.t() @ W
                    grad_B = W @ A.t()

                    # Modify the tensors in place
                    with torch.no_grad():
                        A.sub_(lr * lora_weight_decay * grad_A)
                        B.sub_(lr * lora_weight_decay * grad_B)

                    # Recompute W and its norm using updated A and B
                    W = B @ A
                    W_norm = W.norm().item()

                    # Save weight decay
                    weight_decay.append(W_norm_initial - W_norm)

                # Save the norms after the (optional) updates
                norms.append(W_norm)

        # Convert to tensors, handling empty case
        if len(norms) > 0:
            norms_tensor = torch.tensor(norms, dtype=torch.float32, device=self.model_engine.device)
        else:
            norms_tensor = torch.empty(0, dtype=torch.float32, device=self.model_engine.device)

        # Only return weight decay if regularization was applied
        if lora_weight_decay > 0:
            if len(weight_decay) > 0:
                weight_decay_tensor = torch.tensor(weight_decay, dtype=torch.float32, device=self.model_engine.device)
            else:
                weight_decay_tensor = torch.empty(0, dtype=torch.float32, device=self.model_engine.device)
            return {
                'norms': norms_tensor,
                'weight_decay': weight_decay_tensor
            }
        else:
            return {
                'norms': norms_tensor
            }

    def _apply_control_adapter_regularization_local(self, model, config):
       """Apply orthogonality regularization to Q.

        Applies orthogonality regularization (Newton-style step) to maintain Q^T Q ≈ I.
        NOTE: Orthogonality regularization gradients are computed analytically and applied in-place to Q.

       Returns:
           dict:
               - 'norms'         : tensor of spectral norms of W per parameter (||W||₂ = max|λᵢ|)
               - 'orthogonality' : tensor of ||Q^T Q − I||_F² values (empty if none)
       """
       norms = []
       orthogonality = []

       gamma = config.get('control_adapter_gamma', DEFAULT_CONTROL_ADAPTER_GAMMA)
       assert gamma > 0, f"control_adapter_gamma ({gamma}) must be > 0 to maintain semi-orthogonality"
       assert gamma <= 0.5, f"control_adapter_gamma ({gamma}) must be <= 0.5 to avoid overshooting"

       # Reuse the same identity matrix for orthogonality regularization (all Q have same rank)
       identity_matrix = None

       for name, param in model.named_parameters():
           if 'control_Q' in name:
               Q = param

               # Create identity matrix once (all Q parameters should have same rank)
               if identity_matrix is None:
                   identity_matrix = torch.eye(Q.size(1), device=Q.device, dtype=Q.dtype)

               # *** Newton method for orthogonalization has optimal step size of 0.5 ***
               #
               # For solving Q^T Q = I, the Newton update rule is:
               #   Q_{k+1} = Q_k - (1/2) * Q_k * (Q_k^T Q_k - I)
               #
               # The factor 1/2 is derived from Newton's method for F(Q) = Q^T Q - I = 0
               # and provides quadratic convergence. Step sizes > 0.5 can cause divergence,
               # while step sizes < 0.5 converge slower but more stably.
               #
               # Therefore gamma should be clamped to [0, 0.5].
               #
               # *** Orthogonality regularizer on Q (Q ∈ ℝ^{H×r}) ***
               #
               #   F(Q) = ||QᵀQ − I_r||_F²
               # Gradient:
               #   ∂F/∂Q = 4·Q·(QᵀQ − I_r)    [since (QᵀQ − I_r) is symmetric]
               #
               # Newton step: Q ← Q − (1/2)·Q·(QᵀQ − I_r)
               # Gradient step: Q ← Q − γ·4·Q·(QᵀQ − I_r) = Q ← Q − 4γ·Q·(QᵀQ − I_r)
               # These are equivalent when 4γ = 1/2, i.e., γ = 1/8 (implying effective Hessian = 8I)
               #
               # Implementation: we use Q·(QᵀQ − I_r) directly with step size γ ∈ [0, 0.5]
               QTQ_minus_I = Q.t() @ Q - identity_matrix
               newton_direction = Q @ QTQ_minus_I

               # Modify the tensor in place
               with torch.no_grad():
                   Q.sub_(gamma * newton_direction)

               # Recompute orthogonality error after the update
               QTQ_minus_I_updated = Q.t() @ Q - identity_matrix
               orthogonality_error = torch.norm(QTQ_minus_I_updated, 'fro') ** 2
               orthogonality.append(orthogonality_error.item())

           if 'control_S' in name:
               S = param

               # Convert from log-parameterization: λ = exp(S) - 1, so 1 + λ = exp(S)
               lambda_vec = torch.expm1(S)

               # Save the spectral norm of W: ||W||₂ = max|λᵢ|
               W_norm = torch.max(torch.abs(lambda_vec)).item()
               norms.append(W_norm)

       # Convert to tensors, handling empty case
       if len(orthogonality) > 0:
           orthogonality_tensor = torch.tensor(orthogonality, dtype=torch.float32, device=self.model_engine.device)
       else:
           orthogonality_tensor = torch.empty(0, dtype=torch.float32, device=self.model_engine.device)

       # Convert to tensors, handling empty case
       if len(norms) > 0:
           norms_tensor = torch.tensor(norms, dtype=torch.float32, device=self.model_engine.device)
       else:
           norms_tensor = torch.empty(0, dtype=torch.float32, device=self.model_engine.device)

       return {
           'norms': norms_tensor,
           'orthogonality': orthogonality_tensor
       }

    def _aggregate_statistics(self, local_stats):
        """Aggregate LoRA and Control Adapter statistics across pipeline stages and compute global statistics."""
        global_stats = {}

        for key, tensor in local_stats.items():
            if tensor.numel() > 0:
                count = torch.tensor(tensor.numel(), dtype=torch.float32, device=self.model_engine.device)
                sum_val = torch.sum(tensor)
                min_val = torch.min(tensor)
                max_val = torch.max(tensor)
            else:
                count = torch.tensor(0.0, dtype=torch.float32, device=self.model_engine.device)
                sum_val = torch.tensor(0.0, dtype=torch.float32, device=self.model_engine.device)
                # Use sentinels so MIN/MAX reductions ignore empty ranks
                min_val = torch.tensor(float('inf'), dtype=torch.float32, device=self.model_engine.device)
                max_val = torch.tensor(float('-inf'), dtype=torch.float32, device=self.model_engine.device)

            # Aggregate across pipeline stages if using pipeline parallelism
            if self.model_engine.is_pipe_parallel:
                pp_group = self.model_engine.grid.get_pipe_parallel_group()
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

            # Store in output dictionary with descriptive names
            global_stats[f'{key}_avg'] = global_avg
            global_stats[f'{key}_min'] = global_min
            global_stats[f'{key}_max'] = global_max

        return global_stats

    def _apply_regularization(self, model, config, lr):
        """Apply regularization to LoRA or Control Adapter parameters and return aggregated statistics."""
        if config.get('use_control_adapters', False):
            local_stats = self._apply_control_adapter_regularization_local(model, config)
        else:
            local_stats = self._apply_lora_regularization_local(model, config, lr)
        return self._aggregate_statistics(local_stats)

    def _extract_loss(self, metrics):
        """Extract loss (first metric) as a scalar value."""
        return metrics[0].mean().item()

    def _write_metrics(self, prefix, metrics):
        """Write all metrics to tensorboard."""
        self._write_scalar_metric(f'{prefix}/loss', metrics[0].mean().item())
        self._write_scalar_metric(f'{prefix}/accuracy_top1', metrics[1].mean().item())

    def _write_scalar_metric(self, name, value):
        """Log scalar value to tensorboard using current global step."""
        if is_main_process():
            self.tb_writer.add_scalar(name, value, self.model_engine.global_steps)

    def _print_eval_progress(self, current_loss, last_loss=None):
        """Print evaluation progress with optional percentage change."""
        step = self.model_engine.global_steps
        if step == 0:  # Initial evaluation
            log(f'Initial evaluation loss: {current_loss:.4f}')
        elif last_loss is None:
            log(f'Step {step} evaluation loss: {current_loss:.4f}')
        else:
            percent_change = (current_loss / last_loss - 1) * 100
            log(f'Step {step} evaluation loss: {current_loss:.4f} (last: {last_loss:.4f}, Δ: {percent_change:.2f}%)')

    def _should_checkpoint(self):
        """Check if it's time to checkpoint based on time interval."""
        if not is_main_process():
            return False

        current_time = time.time()
        if (current_time - self.last_checkpoint_time) / 3600 >= self.checkpoint_interval_hours:
            self.last_checkpoint_time = current_time
            return True
        return False

    def _checkpoint_if_needed(self, last_eval_loss, force=False):
        """Save checkpoint and broadcast decision to all processes."""
        do_checkpoint = force or self._should_checkpoint()

        # Broadcast decision to ensure all processes checkpoint together
        result = [do_checkpoint]
        torch.distributed.broadcast_object_list(result, src=0)
        do_checkpoint = result[0]

        if do_checkpoint:
            save_checkpoint(self.model_engine, self.train_dataloader, self.run_dir, last_eval_loss)
            if is_main_process():
                prune_checkpoints(self.run_dir, self.max_checkpoints)

    def _save_model(self, name):
        """Save the trained model (LoRA adapters, Control Adapters, or full model)."""
        if self.lora_config is None:
            save_full_model(self.model_engine, self.pipeline_model, self.model_dir, self.run_dir, name)
        else:
            save_lora(self.model_engine, self.pipeline_model, self.lora_config, self.run_dir, name)