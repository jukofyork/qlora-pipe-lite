from huggingface_hub import save_torch_state_dict
import deepspeed
import glob
import os
import shutil
import torch

from utils.utils import log, safe_rmtree

def save_lora(model_engine, pipeline_model, lora_config, run_dir, name):
    """Save LoRA adapters in distributed fashion across pipeline stages."""
    save_root = run_dir + '/' if run_dir[-1] != '/' else run_dir
    save_dir = save_root + name
    dp_id, stage_id = _get_process_ranks(model_engine)

    # Setup temporary directory for distributed saving
    tmp_dir = _prepare_save_directory(save_dir, dp_id, stage_id)

    # Each pipeline stage saves its trainable parameters
    _save_partial_state_dict(pipeline_model, tmp_dir, stage_id, dp_id, trainable_only=True)

    deepspeed.comm.barrier()

    # Main process merges all partial state dicts and saves final model
    if dp_id == 0 and stage_id == 0:
        state_dict = _merge_and_save_state_dict(tmp_dir, save_dir, dp_id, stage_id)
        torch.save(state_dict, os.path.join(save_dir, 'adapter_model.bin'))
        lora_config.save_pretrained(save_dir)
        safe_rmtree(tmp_dir)

def save_full_model(model_engine, pipeline_model, model_dir, run_dir, name):
    """Save full model in distributed fashion across pipeline stages."""
    save_root = run_dir + '/' if run_dir[-1] != '/' else run_dir
    save_dir = save_root + name
    dp_id, stage_id = _get_process_ranks(model_engine)

    # Setup temporary directory for distributed saving
    tmp_dir = _prepare_save_directory(save_dir, dp_id, stage_id)

    # Each pipeline stage saves all its parameters
    _save_partial_state_dict(pipeline_model, tmp_dir, stage_id, dp_id, trainable_only=False)

    deepspeed.comm.barrier()

    # Main process merges all partial state dicts and saves final model
    if dp_id == 0 and stage_id == 0:
        state_dict = _merge_and_save_state_dict(tmp_dir, save_dir, dp_id, stage_id)
        save_torch_state_dict(state_dict, save_dir, max_shard_size='5GB')
        _copy_model_files(model_dir, save_dir)
        safe_rmtree(tmp_dir)

# Private helper functions
def _get_process_ranks(model_engine):
    """Get data parallel and pipeline parallel ranks for current process."""
    dp_id = model_engine.grid.get_data_parallel_rank()
    stage_id = model_engine.grid.get_pipe_parallel_rank()
    return dp_id, stage_id

def _prepare_save_directory(save_dir, dp_id, stage_id):
    """Create temporary directory for distributed saving."""
    tmp_dir = os.path.join(save_dir, 'tmp')
    if dp_id == 0 and stage_id == 0:
        os.makedirs(tmp_dir, exist_ok=False)
    deepspeed.comm.barrier()
    return tmp_dir

def _save_partial_state_dict(pipeline_model, tmp_dir, stage_id, dp_id, trainable_only=False):
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

def _merge_and_save_state_dict(tmp_dir, save_dir, dp_id, stage_id):
    """Merge partial state dicts from all pipeline stages."""
    if dp_id != 0 or stage_id != 0:
        return None

    state_dict = {}
    for path in glob.glob(os.path.join(tmp_dir, '*.bin')):
        state_dict.update(torch.load(path, map_location='cpu', weights_only=True))

    return state_dict

def _copy_model_files(model_dir, save_dir):
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