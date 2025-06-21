import glob
import json
import os
import shutil

import deepspeed
import torch
import transformers
from huggingface_hub import save_torch_state_dict

from utils import is_main_process, safe_rmtree


def save_lora(model_engine, pipeline_model, args, lora_config, run_dir, name):
    save_root = run_dir + '/' if run_dir[-1] != '/' else run_dir
    dp_id = model_engine.grid.get_data_parallel_rank()
    stage_id = model_engine.grid.get_pipe_parallel_rank()
    save_dir = save_root + name + '_lora'
    tmp_dir = os.path.join(save_dir, 'tmp')
    if dp_id == 0 and stage_id == 0:
        os.makedirs(tmp_dir, exist_ok=False)
    deepspeed.comm.barrier()
    if dp_id == 0:
        partial_state_dict = {}
        for name, p in pipeline_model.named_parameters():
            if p.requires_grad:
                if not hasattr(p, 'original_name'):
                    print(
                        f'WARNING: parameter {name} requires_grad but does not have original_name. Not saving it.'
                    )
                    continue
                partial_state_dict[p.original_name.replace('.default', '').replace('.modules_to_save', '')] = (
                    p.detach()
                )
        torch.save(partial_state_dict, os.path.join(tmp_dir, f'state_dict_{stage_id}.bin'))
    deepspeed.comm.barrier()
    if dp_id == 0 and stage_id == 0:
        state_dict = {}
        for path in glob.glob(os.path.join(tmp_dir, '*.bin')):
            state_dict.update(torch.load(path, map_location='cpu'))
        torch.save(state_dict, os.path.join(save_dir, 'adapter_model.bin'))
        lora_config.save_pretrained(save_dir)
        shutil.copy(args.config, save_dir)
        if hasattr(args, 'deepspeed_config') and args.deepspeed_config is not None:
            shutil.copy(args.deepspeed_config, save_dir)
        safe_rmtree(tmp_dir)


def save_full_model(model_engine, pipeline_model, args, config, run_dir, name):
    save_root = run_dir + '/' if run_dir[-1] != '/' else run_dir
    dp_id = model_engine.grid.get_data_parallel_rank()
    stage_id = model_engine.grid.get_pipe_parallel_rank()
    save_dir = save_root + name + '_model'
    tmp_dir = os.path.join(save_dir, 'tmp')
    if dp_id == 0 and stage_id == 0:
        os.makedirs(tmp_dir, exist_ok=False)
    deepspeed.comm.barrier()
    if dp_id == 0:
        partial_state_dict = {p.original_name: p.detach() for p in pipeline_model.parameters()}
        torch.save(partial_state_dict, os.path.join(tmp_dir, f'state_dict_{stage_id}.bin'))
    deepspeed.comm.barrier()
    if dp_id == 0 and stage_id == 0:
        state_dict = {}
        for path in glob.glob(os.path.join(tmp_dir, '*.bin')):
            state_dict.update(torch.load(path, map_location='cpu'))
        save_torch_state_dict(state_dict, save_dir, max_shard_size='5GB')
        shutil.copy(args.config, save_dir)
        if hasattr(args, 'deepspeed_config') and args.deepspeed_config is not None:
            shutil.copy(args.deepspeed_config, save_dir)
        additional_files_to_copy = [
            'added_tokens.json',
            'config.json',
            'generation_config.json',
            'special_tokens_map.json',
            'tokenizer.json',
            'tokenizer_config.json',
            'tokenizer.model',
        ]
        for path in glob.glob(os.path.join(config['model'], '*')):
            if os.path.basename(path) in additional_files_to_copy:
                shutil.copy(path, save_dir)
        safe_rmtree(tmp_dir)


def save_model(model_engine, pipeline_model, args, lora_config, config, run_dir, name):
    if lora_config is None:
        save_full_model(model_engine, pipeline_model, args, config, run_dir, name)
    else:
        save_lora(model_engine, pipeline_model, args, lora_config, run_dir, name)


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
        print(f'Deleting checkpoint: global_step{step_num}')
        safe_rmtree(path)
