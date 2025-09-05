"""
Training orchestration using a DeepSpeed pipeline engine.

This module provides the Trainer class which:
- Runs the main training loop across epochs and global steps
- Schedules and executes periodic evaluation passes
- Performs time-based and epoch-boundary checkpointing (with pruning)
- Saves models (full model or LoRA adapters) at defined milestones
- Logs key metrics to TensorBoard (rank-0 only)
- Applies optional adapter-specific regularization
- Supports resuming training from a previous checkpoint
"""
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
    DEFAULT_EVALS_PER_EPOCH
)
from training.regularizer import Regularizer
from utils.utils import is_main_process, log, seconds_to_time_str, safe_rmtree

class Trainer:
    """
    Handles the main training loop, evaluation, checkpointing, and model saving.

    This class coordinates a DeepSpeed model engine and pipeline-aware dataloaders to:
    - Perform forward/backward passes with gradient accumulation
    - Log metrics and progress
    - Evaluate periodically within each epoch
    - Save checkpoints and final artifacts

    Parameters:
        config: Dict-like configuration. Recognized keys include:
            - 'model_dir': Directory where final model artifacts are written
            - 'epochs': Number of training epochs (default: 1)
            - 'checkpoint_interval_hours': Minimum hours between checkpoints
            - 'max_checkpoints': Max checkpoints to retain (older are pruned)
            - 'evals_per_epoch': Number of evaluation passes per epoch
        model_engine: DeepSpeed engine (pipeline enabled) with training/eval helpers
        train_dataloader: PipelineDataLoader (or compatible) for training
        eval_dataloader: PipelineDataLoader (or compatible) for evaluation
        run_dir: Directory for run outputs (checkpoints, logs, etc.)
        pipeline_model: Model reference for saving/regularization
        args: Arbitrary arguments namespace (forwarded/available to hooks if needed)
        lora_config: Optional LoRA configuration; if provided, LoRA artifacts are saved
        optimizer: Optimizer instance (used for reporting LR and regularization)
        resume_from_checkpoint: If True, attempt to restore from the latest checkpoint

    Attributes:
        model_dir: Output model directory from config
        epochs: Number of training epochs
        checkpoint_interval_hours: Minimum hours between checkpoints
        max_checkpoints: Max number of checkpoints retained
        tb_writer: TensorBoard SummaryWriter (rank-0 only) or None
        last_checkpoint_time: Timestamp of last checkpoint (rank-0 only) or None
        regularizer: Handles adapter-specific regularization
        total_steps: Total global training steps across all epochs
        eval_step_indices: Set of global step indices where evaluation is triggered
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

        # Initialize regularizer
        self.regularizer = Regularizer(model_engine)

        # Calculate and set total training steps
        model_engine.set_dataloader(train_dataloader)
        steps_per_epoch = len(train_dataloader) // model_engine.gradient_accumulation_steps()
        self.total_steps = steps_per_epoch * self.epochs

        # Calculate evaluation step indices per epoch
        evals_per_epoch = config.get('evals_per_epoch', DEFAULT_EVALS_PER_EPOCH)
        self.eval_step_indices = self._calculate_eval_steps(steps_per_epoch, evals_per_epoch)

    def train(self):
        """
        Main training loop across epochs.

        Workflow:
        - Optionally restore from a previous checkpoint (resumes dataloaders and state)
        - Run an initial evaluation if needed
        - For each training step:
            * Clean up memory pressure on GPU
            * Execute train_batch() and synchronize epoch across pipeline stages
            * Extract/broadcast loss and log metrics (loss, LR)
            * Apply adapter regularization (if configured) and log stats
            * Periodically evaluate based on precomputed step indices
            * Checkpoint at epoch boundaries and optionally by time interval

        Notes:
        - Progress is printed from rank-0 at configured print intervals
        - Evaluation losses are averaged and tracked to show percent change
        """
        # Add timing tracking (align with old CustomPipelineEngine behavior)
        start_time = None
        start_step = None
        last_print_time = None

        last_eval_loss = None

        # Attempt to resume from checkpoint if requested
        if self.resume_from_checkpoint:
            last_eval_loss = self._load_checkpoint()

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
                stats = self.regularizer.apply_regularization(
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
        """
        Run evaluation over the entire eval dataset and return average loss.

        Behavior:
        - Iterates eval_dataloader until it advances to the next epoch
        - Uses model_engine.eval_batch which returns reduced/broadcast loss
        - Resets the eval_dataloader afterward
        - Logs the final averaged loss to TensorBoard

        Returns:
            float: Average evaluation loss across the full validation pass.
        """
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

    def _calculate_eval_steps(self, steps_per_epoch, evals_per_epoch):
        """
        Calculate which global steps to run evaluation on for each epoch.

        Ensures:
        - At least one evaluation per epoch
        - No more evaluations than steps per epoch
        - Avoids step 0

        Parameters:
            steps_per_epoch (int): Number of optimizer steps per epoch.
            evals_per_epoch (int): Requested evaluations per epoch.

        Returns:
            set[int]: Global step indices at which to run evaluation.
        """
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
        """
        Clean up GPU memory pressure prior to a training step.

        Calls Python GC and empties CUDA cache to reduce fragmentation.
        """
        gc.collect()
        torch.cuda.empty_cache()

    def _attempt_to_extract_loss(self, output):
        """
        Return scalar loss (float) on the last pipeline stage; None otherwise.

        Parameters:
            output: DeepSpeed train/eval step output (typically a scalar tensor on last stage).

        Returns:
            Optional[float]: The scalar loss if on the last stage; else None.
        """
        if not self.model_engine.is_last_stage():
            return None
        # DeepSpeed returns a scalar Tensor loss (reduced and broadcast)
        return output.detach().float().mean().item()

    def _broadcast_from_last_stage(self, value):
        """
        Broadcast a scalar from the last pipeline stage to all ranks and return it.

        Parameters:
            value: The scalar value on the last stage (or None on other stages).

        Returns:
            The broadcast scalar value on all ranks.
        """
        src_rank = self.model_engine.grid.stage_to_global(self.model_engine.num_stages - 1)
        payload = [value if self.model_engine.is_last_stage() else None]
        torch.distributed.broadcast_object_list(payload, src=src_rank)
        return payload[0]

    def _write_scalar_metric(self, name, value):
        """
        Rank-0 only: log a scalar to TensorBoard at the current global step.

        Parameters:
            name (str): Metric name.
            value (float): Metric value.
        """
        if is_main_process():
            self.tb_writer.add_scalar(name, value, self.model_engine.global_steps)

    def _print_eval_progress(self, current_loss, last_loss=None):
        """
        Print evaluation progress with optional percentage change.

        Parameters:
            current_loss (float): The current evaluation loss.
            last_loss (Optional[float]): The previous evaluation loss, if available.
        """
        step = self.model_engine.global_steps
        if step == 0:  # Initial evaluation
            log(f'Initial evaluation loss: {current_loss:.4f}')
        elif last_loss is None or last_loss <= 0:
            log(f'Step {step} evaluation loss: {current_loss:.4f}')
        else:
            percent_change = (current_loss / last_loss - 1) * 100
            log(f'Step {step} evaluation loss: {current_loss:.4f} (last: {last_loss:.4f}, Δ: {percent_change:.2f}%)')

    def _should_checkpoint(self):
        """
        Determine if a time-based checkpoint should be taken (rank-0 only).

        Returns:
            bool: True if the configured time interval has elapsed; False otherwise.
        """
        if not is_main_process():
            return False

        current_time = time.time()
        if (current_time - self.last_checkpoint_time) / 3600 >= self.checkpoint_interval_hours:
            self.last_checkpoint_time = current_time
            return True
        return False

    def _checkpoint_if_needed(self, last_eval_loss, force=False):
        """
        Save a checkpoint if requested by time or force, and prune older ones.

        The checkpoint decision is broadcast from rank-0 so that all processes
        act consistently.

        Parameters:
            last_eval_loss (float): The most recent evaluation loss to store with the checkpoint.
            force (bool): If True, checkpoint unconditionally (e.g., at epoch boundaries).
        """
        do_checkpoint = force or self._should_checkpoint()

        # Broadcast decision to ensure all processes checkpoint together
        result = [do_checkpoint]
        torch.distributed.broadcast_object_list(result, src=0)
        do_checkpoint = result[0]

        if do_checkpoint:
            self._save_checkpoint(last_eval_loss)
            if is_main_process():
                self._prune_checkpoints()

    def _save_model(self, name):
        """
        Save the trained model artifacts.

        If LoRA is configured, saves LoRA adapters; otherwise saves the full model.

        Parameters:
            name (str): A human-readable identifier suffix (e.g., 'epoch1').
        """
        if self.lora_config is None:
            self._save_full_model(name)
        else:
            self._save_lora(name)

    def _load_checkpoint(self):
        """
        Load a checkpoint and restore dataloader/engine state.

        Behavior:
        - Uses DeepSpeed engine to load module/optimizer states
        - Restores PipelineDataLoader iteration state
        - Returns the last recorded evaluation loss (if present)

        Returns:
            Optional[float]: The last evaluation loss stored in the checkpoint, or None if not found.
        """
        load_path, client_state = self.model_engine.load_checkpoint(
            self.run_dir,
            load_module_strict=False,
            load_optimizer_states=True
        )

        if load_path is None:
            return None

        self.train_dataloader.load_state_dict(client_state['custom_loader'])
        last_eval_loss = client_state.get('last_eval_loss')

        log(f'Resuming training from checkpoint. Resuming at epoch: {self.train_dataloader.epoch}, step: {self.model_engine.global_steps}')

        return last_eval_loss

    def _save_checkpoint(self, last_eval_loss):
        """
        Save a training checkpoint with current engine and dataloader state.

        Parameters:
            last_eval_loss (float): The most recent evaluation loss to embed in client_state.
        """
        save_root = self.run_dir + '/' if self.run_dir[-1] != '/' else self.run_dir
        self.model_engine.save_checkpoint(
            save_root,
            client_state={
                'custom_loader': self.train_dataloader.state_dict(),
                'last_eval_loss': last_eval_loss
            },
            save_latest=True,
            exclude_frozen_parameters=True,
        )

    def _prune_checkpoints(self):
        """
        Remove old checkpoints, keeping only the most recent `self.max_checkpoints`.

        The method scans for 'global_step*' directories within the run directory,
        sorts them by step number, and removes the oldest ones beyond the retention limit.
        """
        if self.max_checkpoints <= 0:
            return

        save_root = self.run_dir + '/' if self.run_dir[-1] != '/' else self.run_dir

        # Find all checkpoint directories
        checkpoint_pattern = os.path.join(save_root, 'global_step*')
        checkpoints = []

        for path in glob.glob(checkpoint_pattern):
            if os.path.isdir(path):
                # Extract step number for sorting
                basename = os.path.basename(path)
                if basename.startswith('global_step'):
                    try:
                        step_num = int(basename[11:])  # Remove 'global_step' prefix
                        checkpoints.append((step_num, path))
                    except ValueError:
                        continue

        # Sort by step number and keep only the most recent ones
        checkpoints.sort(key=lambda x: x[0])

        # Remove oldest checkpoints
        while len(checkpoints) > self.max_checkpoints:
            step_num, path = checkpoints.pop(0)
            log(f"Pruning oldest checkpoint: 'global_step{step_num}'")
            safe_rmtree(path)

    def _save_lora(self, name):
        """
        Save LoRA adapters in distributed fashion across pipeline stages.

        Behavior:
        - Each pipeline stage writes its trainable parameters to a temporary shard
        - Rank (dp=0, stage=0) merges shards and writes adapter_model.bin
        - LoRA config is saved alongside artifacts
        """
        save_root = self.run_dir + '/' if self.run_dir[-1] != '/' else self.run_dir
        save_dir = save_root + name
        dp_id, stage_id = self._get_process_ranks()

        # Setup temporary directory for distributed saving
        tmp_dir = self._prepare_save_directory(save_dir, dp_id, stage_id)

        # Each pipeline stage saves its trainable parameters
        self._save_partial_state_dict(self.pipeline_model, tmp_dir, stage_id, dp_id, trainable_only=True)

        deepspeed.comm.barrier()

        # Main process merges all partial state dicts and saves final model
        if dp_id == 0 and stage_id == 0:
            state_dict = self._merge_and_save_state_dict(tmp_dir, save_dir, dp_id, stage_id)
            torch.save(state_dict, os.path.join(save_dir, 'adapter_model.bin'))
            self.lora_config.save_pretrained(save_dir)
            safe_rmtree(tmp_dir)

    def _save_full_model(self, name):
        """
        Save full model in distributed fashion across pipeline stages.

        Behavior:
        - Each pipeline stage writes its full parameter shard
        - Rank (dp=0, stage=0) merges shards and saves sharded model files
        - Copies tokenizer/config files from the base model directory
        """
        save_root = self.run_dir + '/' if self.run_dir[-1] != '/' else self.run_dir
        save_dir = save_root + name
        dp_id, stage_id = self._get_process_ranks()

        # Setup temporary directory for distributed saving
        tmp_dir = self._prepare_save_directory(save_dir, dp_id, stage_id)

        # Each pipeline stage saves all its parameters
        self._save_partial_state_dict(self.pipeline_model, tmp_dir, stage_id, dp_id, trainable_only=False)

        deepspeed.comm.barrier()

        # Main process merges all partial state dicts and saves final model
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
            # Skip non-trainable parameters if trainable_only is True
            if trainable_only and not p.requires_grad:
                continue

            if trainable_only:
                # For LoRA: check for original_name and clean parameter names
                if not hasattr(p, 'original_name'):
                    log(f'WARNING: parameter {name} requires_grad but does not have original_name. Not saving it.')
                    continue
                param_name = p.original_name.replace('.default', '').replace('.modules_to_save', '')
            else:
                # For full model: use original_name directly
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
        additional_files_to_copy = [
            'added_tokens.json',
            'config.json',
            'generation_config.json',
            'special_tokens_map.json',
            'tokenizer.json',
            'tokenizer_config.json',
            'tokenizer.model',
        ]

        for path in glob.glob(os.path.join(model_dir, '*')):
            if os.path.basename(path) in additional_files_to_copy:
                shutil.copy(path, save_dir)