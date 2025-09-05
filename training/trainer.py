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
from training.regularization_manager import RegularizationManager
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

        # Initialize regularization manager
        self.regularization_manager = RegularizationManager(model_engine)

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
                stats = self.regularization_manager.apply_regularization(
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

        # Log evaluation metrics and return the mean loss
        return  self._write_metrics('eval', eval_metrics)

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

    def _write_metrics(self, prefix, metrics):
        """Write all metrics to tensorboard and return mean loss."""
        loss = metrics[0].mean().item()
        accuracy_top1 = metrics[1].mean().item()
        self._write_scalar_metric(f'{prefix}/loss', loss)
        self._write_scalar_metric(f'{prefix}/accuracy_top1', accuracy_top1)
        return loss

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