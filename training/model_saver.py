import glob
import os
import shutil

import deepspeed
import torch
from huggingface_hub import save_torch_state_dict

from utils import safe_rmtree


def save_lora(model_engine, pipeline_model, lora_config, run_dir, name):
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
            state_dict.update(torch.load(path, map_location='cpu', weights_only=True))
        torch.save(state_dict, os.path.join(save_dir, 'adapter_model.bin'))
        lora_config.save_pretrained(save_dir)
        safe_rmtree(tmp_dir)


def save_full_model(model_engine, pipeline_model, model_dir, run_dir, name):
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
            state_dict.update(torch.load(path, map_location='cpu', weights_only=True))
        save_torch_state_dict(state_dict, save_dir, max_shard_size='5GB')
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
        safe_rmtree(tmp_dir)