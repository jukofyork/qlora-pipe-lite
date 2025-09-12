#!/usr/bin/env python3
"""
Merge a LoRA adapter into a base model state (in-place into new safetensors shards).

USAGE:
    python merge_lora.py --input /path/to/base_model --adapter /path/to/lora_adapter --output /path/to/output_dir
                         [--scale 1.0] [--no-gpu]
"""

from pathlib import Path
from tqdm import tqdm
import argparse
import json
import os
import safetensors
import safetensors.torch
import shutil
import torch

from training.model_factory import find_lora_weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter weights into base model shards")
    parser.add_argument("--input", required=True, type=str, help="Path to the base model directory.")
    parser.add_argument("--adapter", required=True, type=str, help="Path to the LoRA adapter directory.")
    parser.add_argument("--output", required=True, type=str, help="Path to the output directory.")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for LoRA merging (default: 1.0).")
    parser.add_argument("--no-gpu", action="store_true", help="Use CPU for merging.")
    args = parser.parse_args()

    if not (-2 < args.scale < 2):
        parser.error("--scale must be in the range (-2, 2)")

    input_path, lora_path, output_path = Path(args.input), Path(args.adapter), Path(args.output)
    os.makedirs(output_path, exist_ok=True)

    with open(lora_path / 'adapter_config.json', 'r') as f:
        lora_cfg = json.load(f)
    scale = (lora_cfg['lora_alpha'] / lora_cfg['r']) * args.scale

    device = "cuda" if (not args.no_gpu and torch.cuda.is_available()) else "cpu"

    print('Loading LoRA model...')

    # Check if we have adapter_model.bin or adapter_model.safetensors
    if (lora_path / 'adapter_model.safetensors').exists():
        lora_state = safetensors.torch.load_file(lora_path / 'adapter_model.safetensors')
        if not args.no_gpu and torch.cuda.is_available():
            # Move mapped entries to cuda
            for key, value in tqdm(list(lora_state.items()), desc="Moving adapter to CUDA"):
                lora_state[key] = value.to('cuda')
    else:
        lora_state = torch.load(lora_path / 'adapter_model.bin', map_location=device, weights_only=True)

    shards = [shard for shard in input_path.glob('model*.safetensors')]

    print('Copying unmergable files to output')
    for filepath in input_path.glob('*'):
        if filepath in shards:
            continue
        filepath = Path(filepath)
        if filepath.is_dir():
            continue
        if filepath.suffix == ".gguf":
            # Skip unrelated stray quantizations
            continue
        if filepath.suffix == ".safetensors":
            # Consolidated, possibly
            continue
        print(f'copying {filepath.name} to output')
        shutil.copy(filepath, output_path)

    print('Merging and copying state_dict to output')
    for shard in (pbar := tqdm(shards, desc="Merging shards")):
        tensors = {}
        with safetensors.safe_open(shard, framework='pt', device=device) as f:
            metadata = f.metadata()
            for key in f.keys():
                tensor = f.get_tensor(key)
                lora_A, lora_B = find_lora_weights(key, lora_state)
                if lora_A is not None:
                    pbar.set_description(f'found lora weights for {key}: {tuple(lora_A.size())}, {tuple(lora_B.size())}')
                    old_type = tensor.dtype
                    tensor = tensor.to(torch.float32)
                    lora_delta = scale * lora_B.to(torch.float32) @ lora_A.to(torch.float32)
                    assert lora_delta.shape == tensor.shape, \
                        f"LoRA dimension mismatch for {key}: {lora_delta.shape} vs {tensor.shape}"
                    tensor = (tensor + lora_delta).to(old_type)
                tensors[key] = tensor
        safetensors.torch.save_file(tensors, output_path / shard.name, metadata=metadata)