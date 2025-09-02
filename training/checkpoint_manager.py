import deepspeed
import glob
import os
import torch

from utils.utils import is_main_process, log, safe_rmtree

def load_checkpoint(model_engine, train_dataloader, run_dir):
    """Load checkpoint and return the last eval loss to resume from."""
    load_path, client_state = model_engine.load_checkpoint(
        run_dir,
        load_module_strict=False,
        load_optimizer_states=True
    )

    if load_path is None:
        return None

    train_dataloader.load_state_dict(client_state['custom_loader'])
    last_eval_loss = client_state.get('last_eval_loss')

    log(f'Resuming training from checkpoint. Resuming at epoch: {train_dataloader.epoch}, step: {model_engine.global_steps}')

    return last_eval_loss

def save_checkpoint(model_engine, train_dataloader, run_dir, last_eval_loss):
    """Save training checkpoint with current state."""
    save_root = run_dir + '/' if run_dir[-1] != '/' else run_dir
    model_engine.save_checkpoint(
        save_root,
        client_state={
            'custom_loader': train_dataloader.state_dict(),
            'last_eval_loss': last_eval_loss
        },
        save_latest=True,
        exclude_frozen_parameters=True,
    )

def prune_checkpoints(run_dir, max_checkpoints):
    """Remove old checkpoints, keeping only the most recent max_checkpoints."""
    if max_checkpoints <= 0:
        return

    save_root = run_dir + '/' if run_dir[-1] != '/' else run_dir

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
    while len(checkpoints) > max_checkpoints:
        step_num, path = checkpoints.pop(0)
        log(f"Pruning oldest checkpoint: 'global_step{step_num}'")
        safe_rmtree(path)