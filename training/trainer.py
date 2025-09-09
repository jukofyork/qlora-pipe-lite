from huggingface_hub import save_torch_state_dict
from torch.utils.tensorboard import SummaryWriter
import deepspeed
import gc
import glob
import os
import shutil
import time
import torch

from constants import (
    DEFAULT_CHECKPOINT_INTERVAL_HOURS,
    DEFAULT_MAX_CHECKPOINTS,
    DEFAULT_EVALS_PER_EPOCH,
)
from training.regularizer import Regularizer
from utils.utils import is_main_process, log, seconds_to_time_str, safe_rmtree

class Trainer:
    """
    Orchestrates training with a DeepSpeed pipeline engine.

    Behavior:
    - Runs the main training loop across global steps (epoch is derived from steps)
    - Evaluates periodically within each epoch (using eval_gradient_accumulation_steps)
    - Saves checkpoints (time-based and epoch-boundary) with pruning
    - Saves full model or LoRA adapters
    - Logs metrics to TensorBoard (rank-0 only)
    - Applies optional adapter-specific regularization

    Parameters:
        config                           : Dict-like configuration.
        pipeline_engine                  : DeepSpeed pipeline engine.
        train_dataloader                 : PipelineDataLoader for training.
        eval_dataloader                  : PipelineDataLoader for evaluation.
        run_dir                          : Directory for checkpoints and logs.
        pipeline_model                   : Model reference for saving/regularization.
        args                             : Arbitrary arguments namespace.
        lora_config                      : Optional LoRA configuration.
        optimizer                        : Optimizer instance.
        eval_gradient_accumulation_steps : Micro-batches to accumulate per eval step.
        resume_from_checkpoint           : If True, resume from the latest checkpoint.

    Attributes:
        model_dir                         : Output model directory from config.
        epochs                            : Total number of training epochs.
        checkpoint_interval_hours         : Minimum hours between checkpoints.
        max_checkpoints                   : Max number of checkpoints retained.
        tb_writer                         : TensorBoard SummaryWriter (rank-0 only) or None.
        last_checkpoint_time              : Timestamp of the last checkpoint (rank-0 only) or None.
        regularizer                       : Adapter-specific regularization helper.
        steps_per_epoch                   : Optimizer steps per epoch.
        total_steps                       : Total global training steps across all epochs.
        eval_step_indices                 : Global step indices where evaluation is triggered.
        eval_gradient_accumulation_steps  : Micro-batches to accumulate per eval step.
    """

    def __init__(
        self,
        config,
        pipeline_engine,
        train_dataloader,
        eval_dataloader,
        run_dir,
        pipeline_model,
        args,
        lora_config,
        optimizer,
        eval_gradient_accumulation_steps,
        resume_from_checkpoint=False,
    ):
        self.config = config
        self.pipeline_engine = pipeline_engine
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.run_dir = run_dir
        self.pipeline_model = pipeline_model
        self.args = args
        self.lora_config = lora_config
        self.optimizer = optimizer
        self.eval_gradient_accumulation_steps = eval_gradient_accumulation_steps
        self.resume_from_checkpoint = resume_from_checkpoint

        self.model_dir = config['model_dir']
        self.epochs = config.get('epochs', 1)
        self.checkpoint_interval_hours = config.get('checkpoint_interval_hours', DEFAULT_CHECKPOINT_INTERVAL_HOURS)
        self.max_checkpoints = config.get('max_checkpoints', DEFAULT_MAX_CHECKPOINTS)

        self.tb_writer = SummaryWriter(log_dir=run_dir) if is_main_process() else None
        self.last_checkpoint_time = time.time() if is_main_process() else None

        self.regularizer = Regularizer(pipeline_engine)

        pipeline_engine.set_dataloader(train_dataloader)

        training_steps = len(train_dataloader)
        if training_steps == 0:
            raise RuntimeError('Training dataloader has no data after truncation')

        steps_per_epoch = training_steps // pipeline_engine.gradient_accumulation_steps()
        self.steps_per_epoch = steps_per_epoch
        self.total_steps = self.steps_per_epoch * self.epochs

        evals_per_epoch = config.get('evals_per_epoch', DEFAULT_EVALS_PER_EPOCH)
        self.eval_step_indices = self._calculate_eval_steps(self.steps_per_epoch, evals_per_epoch)

    def train(self):
        """
        Run the main training loop until total_steps are reached.

        Workflow:
        - Optionally resume from checkpoint and run initial evaluation
        - For each training step:
          * Clean up memory pressure on GPU
          * Execute train_batch()
          * Extract/broadcast loss and log metrics (loss, LR)
          * Apply adapter regularization (if configured) and log stats
          * Periodically evaluate based on precomputed step indices
          * Checkpoint at epoch boundaries and optionally by time interval
        """
        last_eval_loss = None

        if self.resume_from_checkpoint:
            last_eval_loss = self._load_checkpoint()

        if last_eval_loss is None:
            last_eval_loss = self.evaluate()

        # Track timing from first step to avoid startup bias
        start_step = self.pipeline_engine.global_steps
        start_time = time.time()
        last_log_time = start_time

        prev_epoch = self.get_current_epoch()
        while self.pipeline_engine.global_steps < self.total_steps:
            self._cleanup_memory()

            # Forward/backward pass and optimizer step
            loss = self.pipeline_engine.train_batch()

            # Current learning rate
            learning_rate = self.optimizer.param_groups[0]['lr']

            # Broadcast training loss from last state so rank-0 has the loss
            train_loss_local = None
            if self.pipeline_engine.is_last_stage():
                train_loss_local = loss.detach().float().mean().item()
            train_loss = self._broadcast_from_last_stage(train_loss_local)

            # Apply adapter-specific regularization
            other_metrics_dict = None
            if self.lora_config is not None:
                other_metrics_dict = self.regularizer.apply_regularization(
                    self.pipeline_model,
                    self.config,
                    learning_rate
                )

            # Print and log training/regularization stats
            if is_main_process():
                last_log_time = self._log_training_progress(
                    start_time=start_time,
                    start_step=start_step,
                    last_log_time=last_log_time,
                    train_loss=train_loss,
                    learning_rate=learning_rate,
                    other_metrics_dict=other_metrics_dict
                )

            # Periodic evaluation
            # NOTE: We exclude evaluation time from next iter throughput and from elapsed/ETA
            if self.pipeline_engine.global_steps in self.eval_step_indices:
                eval_start_time = time.time()
                last_eval_loss = self.evaluate(last_eval_loss)
                dt = time.time() - eval_start_time
                last_log_time += dt
                start_time += dt

            # Epoch boundary detection (based on global step)
            current_epoch = self.get_current_epoch()
            if current_epoch > prev_epoch:
                self._checkpoint_if_needed(last_eval_loss, True)
                self._save_model(f'epoch{prev_epoch}')
                prev_epoch = current_epoch
            else:
                self._checkpoint_if_needed(last_eval_loss)

        log('TRAINING COMPLETE!')
        if is_main_process() and self.tb_writer is not None:
            self.tb_writer.flush()
            self.tb_writer.close()

    def evaluate(self, last_loss=None):
        """
        Run evaluation over the entire eval dataset using the configured eval gradient
        accumulation (eval_gradient_accumulation_steps) and return the average loss.

        Behavior:
        - Prints a short status line on rank-0 before and after evaluation
        - Iterates exactly len(eval_dataloader) groups using pipeline_engine.eval_batch
          with num_micro_batches=eval_gradient_accumulation_steps
        - Drops incomplete final batches via the eval dataloader's internal truncation
        - Resets the eval_dataloader after completion
        - Logs 'eval/loss' to TensorBoard on rank-0
        - If last_loss is provided (> 0), also prints the percent change vs the previous eval

        Args:
            last_loss (float | None): Previous evaluation loss for Δ% reporting.

        Returns:
            float: Average evaluation loss over the eval dataset.
        """
        # Validate evaluation dataset has data
        eval_steps = len(self.eval_dataloader)
        if eval_steps == 0:
            raise RuntimeError('Evaluation dataloader has no data after truncation')

        # Print this to make it obvious the program hasn't just hung for large evaluations
        if is_main_process():
            log('running evaluation... (NOTE: this may take a long time without any output!!!)')

        iterator = iter(self.eval_dataloader)

        # Run evaluation over all batches
        losses = []
        for _ in range(eval_steps):
            # Use the configured eval micro-batch count to match eval_dataloader without changing engine state
            output = self.pipeline_engine.eval_batch(
                iterator,
                num_micro_batches=self.eval_gradient_accumulation_steps
            )
            losses.append(output.detach().float().mean().item())

        # Reset dataloader for next evaluation cycle
        self.eval_dataloader.reset()

        # Calculate average evaluation loss
        eval_loss = float(sum(losses) / len(losses))

        # Log evaluation results
        if is_main_process():
            self._log_evaluation_progress(eval_loss=eval_loss, last_loss=last_loss)

        # The current eval_loss will become last_loss on next call
        return eval_loss

    def get_current_epoch(self):
        """Compute the 1-based epoch from global_steps and steps_per_epoch."""
        return (self.pipeline_engine.global_steps // self.steps_per_epoch) + 1

    def _calculate_eval_steps(self, steps_per_epoch, evals_per_epoch):
        """Precompute global steps on which to run evaluation within each epoch."""
        evals = max(1, min(evals_per_epoch, steps_per_epoch))
        eval_steps = set()
        for epoch in range(self.epochs):
            for i in range(1, evals + 1):
                step_in_epoch = max(1, round(i * steps_per_epoch / evals))
                actual_step = epoch * steps_per_epoch + step_in_epoch
                eval_steps.add(actual_step)
        return eval_steps

    def _cleanup_memory(self):
        """Reduce GPU memory pressure prior to a training step."""
        gc.collect()
        torch.cuda.empty_cache()

    def _broadcast_from_last_stage(self, value):
        """Return scalar loss on all ranks via PP-group all-reduce.

        Only the last pipeline stage contributes a nonzero value; SUM yields the loss.
        Avoids broadcast_object_list and single-source rank mapping issues.
        """
        pp_group = self.pipeline_engine.grid.get_pipe_parallel_group()
        device = getattr(self.pipeline_engine, 'device', None)
        if device is None:
            device = torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else torch.device('cpu')

        contrib = torch.tensor(
            0.0 if value is None else float(value),
            dtype=torch.float32,
            device=device,
        )
        deepspeed.comm.all_reduce(contrib, group=pp_group)  # SUM within PP group
        return float(contrib.item())

    def _log_training_progress(
        self,
        start_time,
        start_step,
        last_log_time,
        train_loss,
        learning_rate,
        other_metrics_dict=None
    ):
        """
        Log training progress to console and tensorboard.

        Args:
            start_time (float): Wall-clock timestamp when training started.
            start_step (int): Global step at start_time.
            last_log_time (float): Wall-clock timestamp of previous log operation.
            train_loss (float): Latest training loss (broadcast to rank-0).
            learning_rate (float): Current learning rate.
            other_metrics_dict (dict | None): Optional statistics (regularization, etc)

        Returns:
            float: Updated last_log_time.
        """
        now = time.time()
        iter_time = now - last_log_time
        last_log_time = now
        iter_throughput = self.pipeline_engine.train_batch_size() / iter_time if iter_time > 0 else 0.0
        total_elapsed = now - start_time
        steps_completed = max(1, self.pipeline_engine.global_steps - start_step)
        remaining_steps = max(0, self.total_steps - self.pipeline_engine.global_steps)
        eta = (total_elapsed / steps_completed) * remaining_steps

        # Log training progress to console
        log(
            f'step: {self.pipeline_engine.global_steps} / {self.total_steps}, '
            f'loss: {train_loss:0.4f}, '
            f'lr: {learning_rate:.3e}, '
            f'throughput: {iter_throughput:0.3f} sequences/s, '
            f'elapsed: {seconds_to_time_str(total_elapsed)}, '
            f'eta: {seconds_to_time_str(eta)}'
        )

        # Log training progress to tensorboard
        metrics = {'loss': train_loss, 'lr': learning_rate}
        if other_metrics_dict:
            metrics.update(other_metrics_dict)
        self._write_metrics('train', metrics)

        # Return the updated last log time
        return last_log_time

    def _log_evaluation_progress(self, eval_loss, last_loss=None):
        """
        Log evaluation results to console and tensorboard

        Args:
            eval_loss (float)        : Current evaluation loss (required if not start).
            last_loss (float | None) : Previous evaluation loss for Δ% reporting.
        """
        if last_loss is None or last_loss <= 0:
            log(f'initial evaluation loss: {eval_loss:.4f}')
        else:
            delta_loss = (eval_loss / last_loss - 1.0) * 100.0
            log(f'new evaluation loss: {eval_loss:.4f} (last: {last_loss:.4f}, Δ: {delta_loss:.2f}%)')
        self._write_metrics('eval', {'loss': eval_loss})

    def _write_metrics(self, prefix, metrics):
        """Log metrics with the given prefix to TensorBoard."""
        for key, value in metrics.items():
            self._write_scalar_metric(f'{prefix}/{key}', value)

    def _write_scalar_metric(self, name, value):
        """Log a scalar to TensorBoard at the current global step."""
        self.tb_writer.add_scalar(name, value, self.pipeline_engine.global_steps)

    def _should_checkpoint(self):
        """Rank-0 only: determine if a time-based checkpoint should be taken."""
        if not is_main_process():
            return False
        current_time = time.time()
        if (current_time - self.last_checkpoint_time) / 3600 >= self.checkpoint_interval_hours:
            self.last_checkpoint_time = current_time
            return True
        return False

    def _checkpoint_if_needed(self, last_eval_loss, force=False):
        """Save a checkpoint if requested by time or force, and prune older ones."""
        do_checkpoint = force or self._should_checkpoint()
        result = [do_checkpoint]
        torch.distributed.broadcast_object_list(result, src=0)
        do_checkpoint = result[0]
        if do_checkpoint:
            self._save_checkpoint(last_eval_loss)
            if is_main_process():
                self._prune_checkpoints()

    def _load_checkpoint(self):
        """
        Load a checkpoint and restore dataloader/engine state.

        Returns:
            Optional[float]: Last evaluation loss stored in the checkpoint, or None if not found.
        """
        load_path, client_state = self.pipeline_engine.load_checkpoint(
            self.run_dir,
            load_module_strict=False,
            load_optimizer_states=True,
        )
        if load_path is None:
            return None
        self.train_dataloader.load_state_dict(client_state['custom_loader'])
        last_eval_loss = client_state.get('last_eval_loss')
        log(
            f'Resuming training from checkpoint. Resuming at epoch: {self.get_current_epoch()}, '
            f'step: {self.pipeline_engine.global_steps}'
        )
        return last_eval_loss

    def _save_checkpoint(self, last_eval_loss):
        """Save a training checkpoint with current engine and dataloader state."""
        save_root = self.run_dir + '/' if self.run_dir[-1] != '/' else self.run_dir
        self.pipeline_engine.save_checkpoint(
            save_root,
            client_state={
                'custom_loader': self.train_dataloader.state_dict(),
                'last_eval_loss': last_eval_loss,
            },
            save_latest=True,
            exclude_frozen_parameters=True,
        )

    def _prune_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent `self.max_checkpoints`."""
        if self.max_checkpoints <= 0:
            return
        save_root = self.run_dir + '/' if self.run_dir[-1] != '/' else self.run_dir
        checkpoint_pattern = os.path.join(save_root, 'global_step*')
        checkpoints = []
        for path in glob.glob(checkpoint_pattern):
            if os.path.isdir(path):
                basename = os.path.basename(path)
                if basename.startswith('global_step'):
                    try:
                        step_num = int(basename[11:])
                        checkpoints.append((step_num, path))
                    except ValueError:
                        continue
        checkpoints.sort(key=lambda x: x[0])
        while len(checkpoints) > self.max_checkpoints:
            step_num, path = checkpoints.pop(0)
            log(f"Pruning oldest checkpoint: 'global_step{step_num}'")
            safe_rmtree(path)

    def _save_model(self, name):
        """Save trained model artifacts (full model or LoRA adapters)."""
        if self.lora_config is None:
            self._save_full_model(name)
        else:
            self._save_lora(name)

    def _save_lora(self, name):
        """
        Save LoRA adapters in distributed fashion across pipeline stages.

        - Each pipeline stage writes its trainable parameters to a temporary shard
        - Rank (dp=0, stage=0) merges shards and writes adapter_model.bin
        - LoRA config is saved alongside artifacts
        """
        save_root = self.run_dir + '/' if self.run_dir[-1] != '/' else self.run_dir
        save_dir = save_root + name
        dp_id, stage_id = self._get_process_ranks()

        tmp_dir = self._prepare_save_directory(save_dir, dp_id, stage_id)
        self._save_partial_state_dict(self.pipeline_model, tmp_dir, stage_id, dp_id, trainable_only=True)
        deepspeed.comm.barrier()

        if dp_id == 0 and stage_id == 0:
            state_dict = self._merge_and_save_state_dict(tmp_dir, save_dir, dp_id, stage_id)
            torch.save(state_dict, os.path.join(save_dir, 'adapter_model.bin'))
            self.lora_config.save_pretrained(save_dir)
            safe_rmtree(tmp_dir)

    def _save_full_model(self, name):
        """
        Save full model in distributed fashion across pipeline stages.

        - Each pipeline stage writes its full parameter shard
        - Rank (dp=0, stage=0) merges shards and saves sharded model files
        - Copies tokenizer/config files from the base model directory
        """
        save_root = self.run_dir + '/' if self.run_dir[-1] != '/' else self.run_dir
        save_dir = save_root + name
        dp_id, stage_id = self._get_process_ranks()

        tmp_dir = self._prepare_save_directory(save_dir, dp_id, stage_id)
        self._save_partial_state_dict(self.pipeline_model, tmp_dir, stage_id, dp_id, trainable_only=False)
        deepspeed.comm.barrier()

        if dp_id == 0 and stage_id == 0:
            state_dict = self._merge_and_save_state_dict(tmp_dir, save_dir, dp_id, stage_id)
            save_torch_state_dict(state_dict, save_dir, max_shard_size='5GB')
            self._copy_model_files(self.model_dir, save_dir)
            safe_rmtree(tmp_dir)

    def _get_process_ranks(self):
        """Get data parallel and pipeline parallel ranks for current process."""
        dp_id = self.pipeline_engine.grid.get_data_parallel_rank()
        stage_id = self.pipeline_engine.grid.get_pipe_parallel_rank()
        return dp_id, stage_id

    def _prepare_save_directory(self, save_dir, dp_id, stage_id):
        """Create temporary directory for distributed saving."""
        tmp_dir = os.path.join(save_dir, 'tmp')
        if dp_id == 0 and stage_id == 0:
            os.makedirs(tmp_dir, exist_ok=False)
        deepspeed.comm.barrier()
        return tmp_dir

    def _save_partial_state_dict(self, pipeline_model, tmp_dir, stage_id, dp_id, trainable_only=False):
        """Save partial state dict for current pipeline stage."""
        if dp_id != 0:
            return
        partial_state_dict = {}
        for name, p in pipeline_model.named_parameters():
            if trainable_only and not p.requires_grad:
                continue
            if trainable_only:
                if not hasattr(p, 'original_name'):
                    log(f'WARNING: parameter {name} requires_grad but does not have original_name. Not saving it.')
                    continue
                param_name = p.original_name.replace('.default', '').replace('.modules_to_save', '')
            else:
                param_name = p.original_name
            partial_state_dict[param_name] = p.detach()
        torch.save(partial_state_dict, os.path.join(tmp_dir, f'state_dict_{stage_id}.bin'))

    def _merge_and_save_state_dict(self, tmp_dir, save_dir, dp_id, stage_id):
        """Merge partial state dicts from all pipeline stages."""
        if dp_id != 0 or stage_id != 0:
            return None
        state_dict = {}
        for path in glob.glob(os.path.join(tmp_dir, '*.bin')):
            state_dict.update(torch.load(path, map_location='cpu', weights_only=True))
        return state_dict

    def _copy_model_files(self, model_dir, save_dir):
        """Copy additional model files (tokenizer, config, etc.) to save directory."""
        # Copy all JSON files
        for path in glob.glob(os.path.join(model_dir, '*.json')):
            shutil.copy(path, save_dir)
        # Also copy tokenizer.model if present
        tok_model_path = os.path.join(model_dir, 'tokenizer.model')
        if os.path.exists(tok_model_path):
            shutil.copy(tok_model_path, save_dir)