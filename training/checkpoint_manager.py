import deepspeed
import glob
import os
import torch

from utils import is_main_process, safe_rmtree

def load_checkpoint(model_engine, train_dataloader, run_dir):
    """Load checkpoint and return the step to resume from."""
    load_path, client_state = model_engine.load_checkpoint(
        run_dir,
        load_module_strict=False,
        load_optimizer_states=True
    )

    if load_path is None:
        return None

    train_dataloader.load_state_dict(client_state['custom_loader'])
    step = client_state['step'] + 1

    if is_main_process():
        print(f'Resuming training from checkpoint. Resuming at epoch: {train_dataloader.epoch}, step: {step}')

    return step

def save_checkpoint(model_engine, train_dataloader, run_dir, step):
    save_root = run_dir + '/' if run_dir[-1] != '/' else run_dir
    model_engine.save_checkpoint(
        save_root,
        client_state={
            'step': step,
            'custom_loader': train_dataloader.state_dict(),
        },
        save_latest=True,
        exclude_frozen_parameters=True,
    )

def prune_checkpoints(run_dir, max_checkpoints):
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

    while len(checkpoints) > max_checkpoints:
        step_num, path = checkpoints.pop(0)
        print(f"- Deleting checkpoint: 'global_step{step_num}'")
        safe_rmtree(path)