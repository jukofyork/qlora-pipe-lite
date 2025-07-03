from aiohttp._http_parser import name
from torch.utils.tensorboard import SummaryWriter
import gc
import time
import torch

from constants import DEFAULT_CHECKPOINT_INTERVAL_HOURS, DEFAULT_MAX_CHECKPOINTS, DEFAULT_EVALS_PER_RUN
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
        self.resume_from_checkpoint = resume_from_checkpoint

        # Extract config values with defaults
        self.model_dir = config['model_dir']
        self.epochs = config.get('epochs', 1)
        self.eval_gradient_accumulation_steps = config.get('eval_gradient_accumulation_steps', 1)
        self.checkpoint_interval_hours = config.get('checkpoint_interval_hours', DEFAULT_CHECKPOINT_INTERVAL_HOURS)
        self.max_checkpoints = config.get('max_checkpoints', DEFAULT_MAX_CHECKPOINTS)
        self.use_control_adapters = config.get('use_control_adapters', False)

        self.tb_writer = SummaryWriter(log_dir=run_dir) if is_main_process() else None
        self.last_checkpoint_time = time.time() if is_main_process() else None

        # Calculate and set total training steps
        model_engine.set_dataloader(train_dataloader)
        steps_per_epoch = len(train_dataloader) // model_engine.gradient_accumulation_steps()
        model_engine.total_steps = steps_per_epoch * self.epochs

        # Calculate evaluation step indices to use across the entire run
        evals_per_run = config.get('evals_per_run', DEFAULT_EVALS_PER_RUN)
        self.eval_step_indices = self._calculate_eval_steps(model_engine.total_steps, evals_per_run)

    def train(self):
        """Main training loop."""
        step = 1

        # Resume from checkpoint if requested
        if self.resume_from_checkpoint:
            resumed_step = load_checkpoint(self.model_engine, self.train_dataloader, self.run_dir)
            if resumed_step is not None:
                step = resumed_step

        # Initial evaluation
        last_eval_loss = self.evaluate(step - 1)
        if is_main_process():
            self._print_eval_progress(step - 1, last_eval_loss)

        # Main training loop
        current_epoch = self.train_dataloader.epoch
        while self.train_dataloader.epoch <= self.epochs:
            self._cleanup_memory()

            # Forward/backward pass and optimizer step
            metrics = self.model_engine.train_batch()
            self.train_dataloader.sync_epoch()

            # Log training metrics
            if is_main_process():
                self._write_metrics('train', metrics, step)

            # Periodic evaluation
            if step in self.eval_step_indices:
                loss = self.evaluate(step)
                if is_main_process():
                    self._print_eval_progress(step, loss, last_eval_loss)
                last_eval_loss = loss

            # Time-based checkpointing
            self._save_checkpoint(step)

            # Check for epoch change and save model
            if self.train_dataloader.epoch > current_epoch:
                self._save_model(f'epoch{current_epoch}')
                current_epoch = self.train_dataloader.epoch

            step += 1

        log('TRAINING COMPLETE!')

    def evaluate(self, step):
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
        if is_main_process():
            self._write_metrics('eval', eval_metrics, step)

        return self._extract_loss(eval_metrics)

    # Private helper methods
    def _calculate_eval_steps(self, total_steps, evals_per_run):
        """Calculate which steps to run evaluation on."""
        eval_steps = set()
        for i in range(1, evals_per_run):
            step_in_run = round(i * total_steps / (evals_per_run - 1))
            eval_steps.add(step_in_run)
        return eval_steps

    def _cleanup_memory(self):
        """Clean up GPU memory before training step."""
        gc.collect()
        torch.cuda.empty_cache()

    def _extract_loss(self, metrics):
        """Extract loss (first metric) as a scalar value."""
        return metrics[0].mean().item()

    def _write_metrics(self, prefix, metrics, step):
        """Write all metrics to tensorboard."""
        self.tb_writer.add_scalar(f'{prefix}/loss', metrics[0].mean().item(), step)

    def _print_eval_progress(self, step, current_loss, last_loss=None):
        """Print evaluation progress with optional percentage change."""
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

    def _save_checkpoint(self, step):
        """Save checkpoint and broadcast decision to all processes."""
        do_checkpoint = self._should_checkpoint()

        # Broadcast decision to ensure all processes checkpoint together
        result = [do_checkpoint]
        torch.distributed.broadcast_object_list(result, src=0)
        do_checkpoint = result[0]

        if do_checkpoint:
            save_checkpoint(self.model_engine, self.train_dataloader, self.run_dir, step)
            if is_main_process():
                prune_checkpoints(self.run_dir, self.max_checkpoints)

    def _save_model(self, name):
        """Save the trained model (LoRA adapters, Control Adapters, or full model)."""
        if self.lora_config is None:
            save_full_model(
                self.model_engine,
                self.pipeline_model,
                self.model_dir,
                self.run_dir,
                name
            )
        else:
            save_lora(
                self.model_engine,
                self.pipeline_model,
                self.lora_config,
                self.run_dir,
                name,
                expand_control_adapters=self.use_control_adapters
            )