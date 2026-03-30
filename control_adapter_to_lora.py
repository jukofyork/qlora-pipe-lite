#!/usr/bin/env python3
"""
Convert Control Adapters to an additive LoRA via SVD approximation.

Computes the effect of the control adapter on model weights:
  Forward:  delta = W @ weight, where W = B @ A (assumes lora_alpha = r, so scale = 1)
  Inverse:  delta = ((I + W)^{-1} - I) @ weight

Then approximates delta via SVD: delta ≈ B_lora @ A_lora

USAGE:
    python control_adapter_to_lora.py --base /path/to/base_model --adapter /path/to/control_adapter --output /path/to/out_dir
                                      [--inverse] [--rank R] [--no-gpu] [--cohere | --mixtral N]
"""

from pathlib import Path
import argparse
import json
import safetensors.torch
import torch

from training.control_adapters import (
    load_control_adapter_weights,
    parse_control_adapter_keys,
    copy_and_patch_adapter_config,
    generate_model_weight_keys,
    generate_lora_key,
    load_model_weights,
)

def load_adapter_config(adapter_path: Path):
    """Load adapter configuration to get lora_alpha and rank."""
    config_path = adapter_path / 'adapter_config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    if 'lora_alpha' not in config or 'r' not in config:
        raise ValueError("adapter_config.json must contain 'lora_alpha' and 'r'")

    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Control Adapters to Additive LoRA via SVD")
    parser.add_argument("--base", required=True, type=str, help="Path to the base model directory")
    parser.add_argument("--adapter", required=True, type=str, help="Path to the Control Adapter directory")
    parser.add_argument("--output", required=True, type=str, help="Path to the output LoRA directory")
    parser.add_argument("--rank", type=int, default=None, help="Override SVD rank (default: use adapter rank)")
    parser.add_argument("--no-gpu", action="store_true", help="Use CPU for computation")
    parser.add_argument("--inverse", action="store_true", help="Use exact inverse: ((I + W)^{-1} - I) @ weight")
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--cohere", action="store_true", help="Also target o_proj for Cohere models")
    model_group.add_argument("--mixtral", type=int, metavar="N", help="Target experts.{0..N-1}.w2 for Mixtral models")

    args = parser.parse_args()

    # Device selection
    device = "cpu" if args.no_gpu or not torch.cuda.is_available() else "cuda"

    control_adapter_path = Path(args.adapter)
    base_model_path = Path(args.base)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load configuration
    adapter_config = load_adapter_config(control_adapter_path)
    lora_alpha = adapter_config['lora_alpha']
    adapter_rank = adapter_config['r']

    # Verify input adapter has lora_alpha == r (scale = 1)
    assert lora_alpha == adapter_rank, \
        f"Input adapter must have lora_alpha == r (got lora_alpha={lora_alpha}, r={adapter_rank})"

    # Determine output rank
    output_rank = args.rank if args.rank is not None else adapter_rank

    # Copy and patch adapter config
    copy_and_patch_adapter_config(control_adapter_path, output_path, args)

    # Update output config to match output rank (with lora_alpha = r for scale = 1)
    output_config_path = output_path / 'adapter_config.json'
    with open(output_config_path, 'r') as f:
        output_config = json.load(f)
    output_config['r'] = output_rank
    output_config['lora_alpha'] = output_rank
    with open(output_config_path, 'w') as f:
        json.dump(output_config, f, indent=2)

    # Load weights
    control_keys, control_state_dict = load_control_adapter_weights(control_adapter_path)
    model_weights = load_model_weights(base_model_path)

    lora_state_dict = {}
    layer_data = parse_control_adapter_keys(control_state_dict)

    print()
    print(f"Converting {len(layer_data)} layers (device='{device}', output_rank={output_rank}):")
    print()

    for layer_idx in sorted(layer_data.keys()):
        if 'A' not in layer_data[layer_idx] or 'B' not in layer_data[layer_idx]:
            continue

        A = layer_data[layer_idx]['A']
        B = layer_data[layer_idx]['B']
        old_dtype = A.dtype

        # Move to device as float32 for computation
        A_gpu = A.to(device=device, dtype=torch.float32)
        B_gpu = B.to(device=device, dtype=torch.float32)

        # Compute composite matrix W = B @ A (scale = 1 due to lora_alpha == r)
        W = B_gpu @ A_gpu  # [hidden_size, hidden_size]

        target_keys = generate_model_weight_keys(layer_idx, args)

        for target_key in target_keys:
            if target_key not in model_weights:
                continue

            # Load base weight
            weight = model_weights[target_key].to(device=device, dtype=torch.float32)

            # Compute delta based on forward or inverse mode
            if args.inverse:
                # Exact inverse: delta = ((I + W)^{-1} - I) @ weight
                I = torch.eye(W.size(0), device=device, dtype=torch.float32)
                W_inv = torch.linalg.inv(I + W)
                delta = (W_inv - I) @ weight
            else:
                # Forward: delta = W @ weight
                delta = W @ weight

            # SVD approximation of delta
            U, S, Vt = torch.linalg.svd(delta, full_matrices=False)

            # Move back to CPU for rank selection
            U, S, Vt = U.cpu(), S.cpu(), Vt.cpu()

            # Determine actual rank to use
            max_rank = min(output_rank, len(S))
            if max_rank == 0:
                continue

            sqrt_S = torch.sqrt(S[:max_rank])

            # Construct LoRA matrices: delta ≈ B_lora @ A_lora
            A_lora = torch.diag(sqrt_S) @ Vt[:max_rank,:]  # [max_rank, output_size]
            B_lora = U[:,:max_rank] @ torch.diag(sqrt_S)  # [hidden_size, max_rank]

            # Cast to original dtype
            A_lora = A_lora.to(old_dtype)
            B_lora = B_lora.to(old_dtype)

            # Generate keys and save
            a_key = generate_lora_key(layer_idx, target_key, 'A', args)
            b_key = generate_lora_key(layer_idx, target_key, 'B', args)
            lora_state_dict[a_key] = A_lora
            lora_state_dict[b_key] = B_lora

            # Compute variance explained
            total_variance = torch.sum(S ** 2)
            explained_variance = torch.sum(S[:max_rank] ** 2)
            variance_pct = 100 * explained_variance / total_variance if total_variance > 1e-10 else 0.0

            base_key = f"base_model.model.model.layers.{layer_idx}"
            print(f"- Layer {layer_idx}: SVD rank {max_rank}/{len(S)}, {variance_pct:.1f}% variance explained")
            print(f"  -- '{base_key}.control_A.weight' + '{base_key}.control_B.weight'")
            print(f"  -> '{a_key}' + '{b_key}'")
            print()

    print(f"Done (total tensors: {len(control_state_dict)} -> {len(lora_state_dict)})")
    print()

    safetensors.torch.save_file(lora_state_dict, output_path / 'adapter_model.safetensors')
    print(f"Converted LoRA adapter saved to: '{output_path}'")