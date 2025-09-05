from torch.utils.tensorboard import SummaryWriter
import gc
import time
import torch

from constants import (
    DEFAULT_CHECKPOINT_INTERVAL_HOURS,
    DEFAULT_MAX_CHECKPOINTS,
    DEFAULT_EVALS_PER_EPOCH
)
from training.checkpoint_manager import load_checkpoint, save_checkpoint, prune_checkpoints
from training.model_saver import save_lora, save_full_model
from training.regularization_manager import RegularizationManager
from utils.utils import is_main_process, log, seconds_to_time_str

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
        self.checkpoint_interval_hours = config.get('checkpoint_interval_hours', DEFAULT_CHECKPOINT_INTERVAL_HOURS)
        self.max_checkpoints = config.get('max_checkpoints', DEFAULT_MAX_CHECKPOINTS)

        self.tb_writer = SummaryWriter(log_dir=run_dir) if is_main_process() else None
        self.last_checkpoint_time = time.time() if is_main_process() else None

        # Initialize regularization manager
        self.regularization_manager = RegularizationManager(model_engine)

        # Calculate and set total training steps
        model_engine.set_dataloader(train_dataloader)
        steps_per_epoch = len(train_dataloader) // model_engine.gradient_accumulation_steps()
        self.total_steps = steps_per_epoch * self.epochs

        # Calculate evaluation step indices per epoch
        evals_per_epoch = config.get('evals_per_epoch', DEFAULT_EVALS_PER_EPOCH)
        self.eval_step_indices = self._calculate_eval_steps(steps_per_epoch, evals_per_epoch)

    def train(self):
        """Main training loop."""

        # Add timing tracking (align with old CustomPipelineEngine behavior)
        start_time = None
        start_step = None
        last_print_time = None

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
            loss = self.model_engine.train_batch()
            self.train_dataloader.sync_epoch()

            # Compute and log training loss (broadcast so rank-0 has the true loss)
            train_loss_local = self._attempt_to_extract_loss(loss)
            train_loss = self._broadcast_from_last_stage(train_loss_local)
            self._write_scalar_metric('train/loss', train_loss)
            self._write_scalar_metric('train/lr', self.optimizer.param_groups[0]['lr'])

            # Track timing from first step to avoid startup bias
            if is_main_process() and start_step is None:
                start_step = self.model_engine.global_steps - 1
                start_time = time.time()
                last_print_time = start_time

            # Custom progress logging matching previous format
            if is_main_process() and self.model_engine.global_steps % self.model_engine.steps_per_print() == 0:
                if start_step is not None and last_print_time is not None:
                    now = time.time()
                    iter_elapsed = now - last_print_time
                    last_print_time = now
                    iter_time = iter_elapsed / self.model_engine.steps_per_print()
                    iter_throughput = self.model_engine.train_batch_size() / iter_time

                    total_elapsed = now - start_time
                    steps_completed = self.model_engine.global_steps - start_step
                    if steps_completed > 0:
                        eta = (total_elapsed / steps_completed) * (self.total_steps - self.model_engine.global_steps)
                        log(f'step: {self.model_engine.global_steps} / {self.total_steps}, '
                            f'loss: {train_loss:0.4f}, '
                            f'throughput: {iter_throughput:0.3f} sequences/s, '
                            f'elapsed: {seconds_to_time_str(total_elapsed)}, '
                            f'eta: {seconds_to_time_str(eta)}')

            # Apply adapter-specific regularization and log statistics
            if self.lora_config is not None:
                stats = self.regularization_manager.apply_regularization(
                    self.pipeline_model,
                    self.config,
                    self.optimizer.param_groups[0]['lr']
                )
                for key, value in stats.items():
                    self._write_scalar_metric(f'train/{key}', value)

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
        """Run evaluation over the entire eval dataset and return average loss."""
        iterator = iter(self.eval_dataloader)
        start_epoch = self.eval_dataloader.epoch
        losses = []

        while True:
            # DeepSpeed eval_batch already reduces and broadcasts loss to all pipeline stages
            output = self.model_engine.eval_batch(iterator)
            # DeepSpeed returns a scalar Tensor loss (reduced and broadcast)
            losses.append(output.detach().float().mean().item())

            self.eval_dataloader.sync_epoch()
            if self.eval_dataloader.epoch > start_epoch:
                break

        self.eval_dataloader.reset()
        eval_loss = float(sum(losses) / len(losses)) if losses else 0.0
        self._write_scalar_metric('eval/loss', eval_loss)
        return eval_loss

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

    def _attempt_to_extract_loss(self, output):
        """Return scalar loss as float on last pipeline stage; None otherwise."""
        if not self.model_engine.is_last_stage():
            return None
        # DeepSpeed returns a scalar Tensor loss (reduced and broadcast)
        return output.detach().float().mean().item()

    def _broadcast_from_last_stage(self, value):
        """Broadcast a scalar from the last pipeline stage to all ranks and return it."""
        src_rank = self.model_engine.grid.stage_to_global(self.model_engine.num_stages - 1)
        payload = [value if self.model_engine.is_last_stage() else None]
        torch.distributed.broadcast_object_list(payload, src=src_rank)
        return payload[0]

    def _write_scalar_metric(self, name, value):
        """Rank-0 only: log scalar to TensorBoard at current global step."""
        if is_main_process():
            self.tb_writer.add_scalar(name, value, self.model_engine.global_steps)

    def _print_eval_progress(self, current_loss, last_loss=None):
        """Print evaluation progress with optional percentage change."""
        step = self.model_engine.global_steps
        if step == 0:  # Initial evaluation
            log(f'Initial evaluation loss: {current_loss:.4f}')
        elif last_loss is None or last_loss <= 0:
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