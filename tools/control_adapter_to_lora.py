#!/usr/bin/env python3
"""
This script converts Control Adapters to an approximate additive LoRA by computing
the multiplicative effect on actual model weights, then approximating the delta via
an SVD decomposition.

Usage: python control_adapter_to_lora.py base_model_path control_adapter_path output_path
"""

from pathlib import Path
from tqdm import tqdm
from typing import List, Dict
import argparse
import json
import re
import safetensors
import safetensors.torch
import torch

from control_adapter_utils import *

def copy_and_patch_adapter_config(input_path: Path, output_path: Path, args):
    """Copy adapter config, patch target_modules based on model type, and optionally patch rank/alpha."""
    config_file = input_path / 'adapter_config.json'
    if not config_file.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {input_path}")

    with open(config_file, 'r') as f:
        config = json.load(f)

    # Compute scale factor from original config before modifying
    original_rank = config['r']
    original_alpha = config.get('lora_alpha', original_rank)
    scale_factor = original_alpha / original_rank

    # Determine target rank (allows exceeding original rank)
    target_rank = args.rank if args.rank is not None else original_rank

    # Set target modules based on model type
    if args.mixtral:
        config['target_modules'] = ["w2"]
    elif args.cohere:
        config['target_modules'] = ["down_proj", "o_proj"]
    else:
        config['target_modules'] = ["down_proj"]

    # Update rank if different from original
    config['r'] = target_rank

    # Always set alpha = rank since scale factor is baked into tensors
    config['lora_alpha'] = target_rank

    with open(output_path / 'adapter_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("Updated and copied 'adapter_config.json'")

    return scale_factor, target_rank

def generate_model_weight_keys(layer_idx: int, args) -> List[str]:
    """Generate model weight keys for a given layer based on model type."""
    base_key = f"model.layers.{layer_idx}"
    target_keys = []

    if args.mixtral:
        for expert_idx in range(args.mixtral):
            target_keys.append(f"{base_key}.block_sparse_moe.experts.{expert_idx}.w2.weight")
    else:
        target_keys.append(f"{base_key}.mlp.down_proj.weight")
        if args.cohere:
            target_keys.append(f"{base_key}.self_attn.o_proj.weight")

    return target_keys

def generate_lora_key(layer_idx: int, target_key: str, adapter_type: str, args) -> str:
    """Generate LoRA A or B key for multiplicative LoRA format."""
    base_key = f"base_model.model.model.layers.{layer_idx}"

    if args.mixtral:
        # Extract expert index from target_key
        expert_match = re.search(r'experts\.(\d+)\.w2\.weight', target_key)
        if expert_match:
            expert_idx = expert_match.group(1)
            lora_base = f"{base_key}.block_sparse_moe.experts.{expert_idx}.w2"
        else:
            raise ValueError(f"Could not parse expert index from {target_key}")
    elif "o_proj" in target_key:
        lora_base = f"{base_key}.self_attn.o_proj"
    else:
        lora_base = f"{base_key}.mlp.down_proj"

    return f"{lora_base}.lora_{adapter_type}.weight"

def load_model_weights(model_path: Path) -> Dict[str, torch.Tensor]:
    """Load model weights from safetensors shards."""
    model_weights = {}
    model_shards = list(model_path.glob('model*.safetensors'))
    if not model_shards:
        raise FileNotFoundError("No model*.safetensors files found in model directory")

    for shard in tqdm(model_shards, desc="Loading model shards"):
        with safetensors.safe_open(shard, framework='pt', device='cpu') as f:
            for key in f.keys():
                model_weights[key] = f.get_tensor(key)

    return model_weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Control Adapters to Additive LoRA format via SVD")
    parser.add_argument("base_model_path", type=str, help="Path to the base model directory")
    parser.add_argument("control_adapter_path", type=str, help="Path to the Control Adapter directory")
    parser.add_argument("output_path", type=str, help="Path to the output LoRA directory")
    parser.add_argument("--rank", type=int, help="Override rank for SVD truncation (default: use original rank)")
    parser.add_argument("--no-gpu", action="store_true", help="Use CPU for SVD computation")
    parser.add_argument("--inverse", action="store_true", help="Use exact inverse delta: (I + W)^{-1} - I")
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--cohere", action="store_true", help="Also target o_proj for Cohere models")
    model_group.add_argument("--mixtral", type=int, metavar="N", help="Target experts.{0..N-1}.w2 for Mixtral models")

    args = parser.parse_args()

    # Move to GPU if available for faster computation
    device = "cpu" if args.no_gpu or not torch.cuda.is_available() else "cuda"

    control_adapter_path = Path(args.control_adapter_path)
    base_model_path = Path(args.base_model_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    scale_factor, target_rank = copy_and_patch_adapter_config(control_adapter_path, output_path, args)

    control_keys, control_state_dict = load_control_adapter_weights(control_adapter_path)

    model_weights = load_model_weights(base_model_path)

    lora_state_dict = {}

    layer_data = parse_control_adapter_keys(control_state_dict)

    print()
    print(f"Converting {len(control_keys)//2} layers ({len(control_keys)} tensors) "
          f"(device='{device}', scale={f'{scale_factor:.4f}'.rstrip('0').rstrip('.')}):")

    for layer_idx in sorted(layer_data.keys()):
        if 'Q' not in layer_data[layer_idx] or 'lambda' not in layer_data[layer_idx]:
            continue

        Q = layer_data[layer_idx]['Q']
        lambda_vec = layer_data[layer_idx]['lambda']
        old_type = Q.dtype

        # Choose forward or exact inverse delta
        if args.inverse:
            lora_delta = apply_control_adapter_inverse_transform(Q, lambda_vec, scale_factor, device)
        else:
            lora_delta = apply_control_adapter_transform(Q, lambda_vec, scale_factor, device)

        target_keys = generate_model_weight_keys(layer_idx, args)

        for target_key in target_keys:
            if target_key not in model_weights:
                continue

            weight = model_weights[target_key].to(device=device, dtype=torch.float32)
            multiplicative_effect = lora_delta @ weight

            U, S, Vt = torch.linalg.svd(multiplicative_effect, full_matrices=False)
            U, S, Vt = U.cpu(), S.cpu(), Vt.cpu()

            # Truncate to target rank
            if target_rank > len(S):
                raise ValueError(f"Requested rank {target_rank} exceeds maximum available rank {len(S)} from SVD")
            sqrt_S = torch.sqrt(S[:target_rank])
            A_approx = torch.diag(sqrt_S) @ Vt[:target_rank,:]
            B_approx = U[:,:target_rank] @ torch.diag(sqrt_S)

            A_approx = A_approx.to(old_type)
            B_approx = B_approx.to(old_type)

            a_key = generate_lora_key(layer_idx, target_key, 'A', args)
            b_key = generate_lora_key(layer_idx, target_key, 'B', args)
            lora_state_dict[a_key] = A_approx
            lora_state_dict[b_key] = B_approx

            base_key = f"base_model.model.model.layers.{layer_idx}"
            print()
            print(f"- Layer {layer_idx}, SVD rank {target_rank}/{len(S)}, "
                  f"{100*torch.sum(S[:target_rank]**2)/torch.sum(S**2):.1f}% of variance explained:")
            print(f"  -- '{base_key}.control_Q' + '{base_key}.control_lambda'")
            print(f"  -> '{a_key}' + '{b_key}'")

    print()
    print(f"Done (total tensors: {len(control_state_dict)} -> {len(lora_state_dict)})")

    safetensors.torch.save_file(lora_state_dict, output_path / 'adapter_model.safetensors')
    print()
    print(f"Converted LoRA adapter saved to: '{output_path}'")