from torch.utils.tensorboard import SummaryWriter
import deepspeed
import gc
import glob
import os
import shutil
import time
import torch

from huggingface_hub import save_torch_state_dict

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
    - Evaluates periodically within each epoch
    - Saves checkpoints (time-based and epoch-boundary) with pruning
    - Saves full model or LoRA adapters
    - Logs metrics to TensorBoard (rank-0 only)
    - Applies optional adapter-specific regularization

    Parameters:
        config                 : Dict-like configuration.
        model_engine           : DeepSpeed pipeline engine.
        train_dataloader       : PipelineDataLoader for training.
        eval_dataloader        : PipelineDataLoader for evaluation (GAS=1 recommended).
        run_dir                : Directory for checkpoints and logs.
        pipeline_model         : Model reference for saving/regularization.
        args                   : Arbitrary arguments namespace.
        lora_config            : Optional LoRA configuration.
        optimizer              : Optimizer instance.
        resume_from_checkpoint : If True, resume from the latest checkpoint.

    Attributes:
        model_dir                 : Output model directory from config.
        epochs                    : Total number of training epochs.
        checkpoint_interval_hours : Minimum hours between checkpoints.
        max_checkpoints           : Max number of checkpoints retained.
        tb_writer                 : TensorBoard SummaryWriter (rank-0 only) or None.
        last_checkpoint_time      : Timestamp of the last checkpoint (rank-0 only) or None.
        regularizer               : Adapter-specific regularization helper.
        steps_per_epoch           : Optimizer steps per epoch.
        total_steps               : Total global training steps across all epochs.
        eval_step_indices         : Global step indices where evaluation is triggered.
    """

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
        resume_from_checkpoint=False,
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

        self.model_dir = config['model_dir']
        self.epochs = config.get('epochs', 1)
        self.checkpoint_interval_hours = config.get('checkpoint_interval_hours', DEFAULT_CHECKPOINT_INTERVAL_HOURS)
        self.max_checkpoints = config.get('max_checkpoints', DEFAULT_MAX_CHECKPOINTS)

        self.tb_writer = SummaryWriter(log_dir=run_dir) if is_main_process() else None
        self.last_checkpoint_time = time.time() if is_main_process() else None

        self.regularizer = Regularizer(model_engine)

        model_engine.set_dataloader(train_dataloader)

        training_steps = len(train_dataloader)
        if training_steps == 0:
            raise RuntimeError('Training dataloader has no data after truncation')

        steps_per_epoch = training_steps // model_engine.gradient_accumulation_steps()
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
        start_time = None
        start_step = None
        last_print_time = None

        last_eval_loss = None

        if self.resume_from_checkpoint:
            last_eval_loss = self._load_checkpoint()

        if last_eval_loss is None:
            last_eval_loss = self.evaluate()
            if is_main_process():
                self._print_eval_progress(last_eval_loss)

        prev_epoch = self.get_current_epoch()
        while self.model_engine.global_steps < self.total_steps:
            self._cleanup_memory()

            # Forward/backward pass and optimizer step
            loss = self.model_engine.train_batch()

            # Current learning rate
            learning_rate = self.optimizer.param_groups[0]['lr']

            # Compute and log training loss (broadcast so rank-0 has the true loss)
            train_loss_local = self._attempt_to_extract_loss(loss)
            train_loss = self._broadcast_from_last_stage(train_loss_local)
            self._write_train_step_metrics(train_loss, learning_rate)

            # Track timing from first step to avoid startup bias
            if is_main_process() and start_step is None:
                start_step = self.model_engine.global_steps - 1
                start_time = time.time()
                last_print_time = start_time

            # Custom progress logging
            last_print_time = self._maybe_print_train_progress(
                start_time=start_time,
                start_step=start_step,
                last_print_time=last_print_time,
                train_loss=train_loss,
            )

            # Apply adapter-specific regularization and log statistics
            if self.lora_config is not None:
                stats = self.regularizer.apply_regularization(self.pipeline_model, self.config, learning_rate)
                self._write_regularizer_metrics(stats)

            # Periodic evaluation
            if self.model_engine.global_steps in self.eval_step_indices:
                loss = self.evaluate()
                if is_main_process():
                    self._print_eval_progress(loss, last_eval_loss)
                last_eval_loss = loss

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

    def evaluate(self):
        """
        Run evaluation over the entire eval dataset (GAS=1) and return average loss.

        Behavior:
        - Eval dataloader constructed with gradient_accumulation_steps=1
        - Iterate exactly len(eval_dataloader) micro-batches
        - With truncation in the dataloader, incomplete final batches are dropped
        - Uses model_engine.eval_batch which returns reduced/broadcast loss
        - Resets the eval_dataloader afterward
        """
        eval_steps = len(self.eval_dataloader)
        if eval_steps == 0:
            raise RuntimeError('Evaluation dataloader has no data after truncation')

        iterator = iter(self.eval_dataloader)

        losses = []
        for _ in range(eval_steps):
            output = self.model_engine.eval_batch(iterator)
            losses.append(output.detach().float().mean().item())

        self.eval_dataloader.reset()
        eval_loss = float(sum(losses) / len(losses))
        self._write_scalar_metric('eval/loss', eval_loss)
        return eval_loss

    def get_current_epoch(self):
        """Compute the 1-based epoch from global_steps and steps_per_epoch."""
        return (self.model_engine.global_steps // self.steps_per_epoch) + 1

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

    def _attempt_to_extract_loss(self, output):
        """Return scalar loss (float) on the last pipeline stage; None otherwise."""
        if not self.model_engine.is_last_stage():
            return None
        return output.detach().float().mean().item()

    def _broadcast_from_last_stage(self, value):
        """Broadcast a scalar from the last pipeline stage to all ranks and return it."""
        src_rank = self.model_engine.grid.stage_to_global(self.model_engine.num_stages - 1)
        payload = [value if self.model_engine.is_last_stage() else None]
        torch.distributed.broadcast_object_list(payload, src=src_rank)
        return payload[0]

    def _write_scalar_metric(self, name, value):
        """Rank-0 only: log a scalar to TensorBoard at the current global step."""
        if is_main_process():
            self.tb_writer.add_scalar(name, value, self.model_engine.global_steps)

    def _write_train_step_metrics(self, train_loss, learning_rate):
        """Rank-0 only: log core training step metrics (loss, LR)."""
        self._write_scalar_metric('train/loss', train_loss)
        self._write_scalar_metric('train/lr', learning_rate)

    def _write_regularizer_metrics(self, stats):
        """Rank-0 only: log adapter/regularizer-specific training metrics."""
        for key, value in stats.items():
            self._write_scalar_metric(f'train/{key}', value)

    def _print_eval_progress(self, current_loss, last_loss=None):
        """Rank-0 only: print evaluation progress with optional percentage change."""
        step = self.model_engine.global_steps
        if step == 0:
            log(f'Initial evaluation loss: {current_loss:.4f}')
        elif last_loss is None or last_loss <= 0:
            log(f'Step {step} evaluation loss: {current_loss:.4f}')
        else:
            percent_change = (current_loss / last_loss - 1) * 100
            log(f'Step {step} evaluation loss: {current_loss:.4f} (last: {last_loss:.4f}, Δ: {percent_change:.2f}%)')

    def _maybe_print_train_progress(self, start_time, start_step, last_print_time, train_loss):
        """
        Rank-0 only: periodically print training progress (throughput, elapsed, ETA).

        Returns:
            float | None: Updated last_print_time (or the input if no print was made).
        """
        if not is_main_process():
            return last_print_time
        steps_per_print = self.model_engine.steps_per_print()
        if steps_per_print <= 0 or (self.model_engine.global_steps % steps_per_print) != 0:
            return last_print_time
        if start_step is None or last_print_time is None:
            return last_print_time
        now = time.time()
        iter_elapsed = now - last_print_time
        last_print_time = now
        iter_time = iter_elapsed / steps_per_print if steps_per_print > 0 else float('inf')
        iter_throughput = self.model_engine.train_batch_size() / iter_time if iter_time > 0 else 0.0
        total_elapsed = now - start_time
        steps_completed = self.model_engine.global_steps - start_step
        if steps_completed > 0:
            eta = (total_elapsed / steps_completed) * (self.total_steps - self.model_engine.global_steps)
            log(
                f'step: {self.model_engine.global_steps} / {self.total_steps}, '
                f'loss: {train_loss:0.4f}, '
                f'throughput: {iter_throughput:0.3f} sequences/s, '
                f'elapsed: {seconds_to_time_str(total_elapsed)}, '
                f'eta: {seconds_to_time_str(eta)}'
            )
        return last_print_time

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
        load_path, client_state = self.model_engine.load_checkpoint(
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
            f'step: {self.model_engine.global_steps}'
        )
        return last_eval_loss

    def _save_checkpoint(self, last_eval_loss):
        """Save a training checkpoint with current engine and dataloader state."""
        save_root = self.run_dir + '/' if self.run_dir[-1] != '/' else self.run_dir
        self.model_engine.save_checkpoint(
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
        dp_id = self.model_engine.grid.get_data_parallel_rank()
        stage_id = self.model_engine.grid.get_pipe_parallel_rank()
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