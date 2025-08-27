#!/usr/bin/env python3
"""
This script converts Control Adapters to a multiplicative LoRA by distributing the
multiplicative effect to specific linear layers within each transformer block.

Usage: python control_adapter_to_multiplicative_lora.py control_adapter_path output_path [--cohere | --mixtral N]
"""

from pathlib import Path
import argparse
import math
import torch

from control_adapter_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Control Adapters to Multiplicative LoRA format")
    parser.add_argument("control_adapter_path", type=str, help="The path to the Control Adapter directory.")
    parser.add_argument("output_path", type=str, help="The path to the LoRA directory.")
    parser.add_argument("--rank", type=int, help="Override rank for SVD truncation (default: use original rank)")
    parser.add_argument("--no-gpu", action="store_true", help="Use CPU for SVD computation")
    add_model_args(parser)

    args = parser.parse_args()
    svd_device = "cpu" if args.no_gpu or not torch.cuda.is_available() else "cuda"
    control_adapter_path = Path(args.control_adapter_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    scale_factor, target_rank, original_rank = copy_and_patch_adapter_config(control_adapter_path, output_path, args)

    control_keys, control_state_dict = load_control_adapter_weights(control_adapter_path)

    lora_state_dict = {}

    layer_data = parse_control_adapter_keys(control_state_dict)

    print(f"Converting {len(control_keys)} control adapter tensors (device='{svd_device}', scale={scale_factor:.4f}):")

    for layer_idx in sorted(layer_data.keys()):
        if 'A' not in layer_data[layer_idx] or 'B' not in layer_data[layer_idx]:
            continue

        A = layer_data[layer_idx]['A']
        B = layer_data[layer_idx]['B']
        old_dtype = A.dtype

        if target_rank < original_rank:
            # Bake in scale factor to combined effect
            combined_effect = scale_factor * (B.to(torch.float32) @ A.to(torch.float32))

            # Compute SVD on target device
            effect_gpu = combined_effect.to(svd_device)
            U, S, Vt = torch.linalg.svd(effect_gpu, full_matrices=False)
            U, S, Vt = U.cpu(), S.cpu(), Vt.cpu()

            # Truncate to target rank to compress
            if target_rank > len(S):
                raise ValueError(f"Requested rank {target_rank} exceeds maximum available rank {len(S)} from SVD")
            sqrt_S = torch.sqrt(S[:target_rank])
            new_A = torch.diag(sqrt_S) @ Vt[:target_rank,:]
            new_B = U[:,:target_rank] @ torch.diag(sqrt_S)

            print(f"- Layer {layer_idx}, SVD rank {target_rank}/{original_rank}, "
                  f"{100*torch.sum(S[:target_rank]**2)/torch.sum(S**2):.4f}% of variance explained:")
        else:
            # Just bake in scale factor without compression
            sqrt_scale = math.sqrt(scale_factor)
            new_A = sqrt_scale * A.to(torch.float32)
            new_B = sqrt_scale * B.to(torch.float32)

            print(f"- Layer {layer_idx}:")

        new_A = new_A.to(old_dtype)
        new_B = new_B.to(old_dtype)

        # Distribute to target keys
        target_keys = generate_model_weight_keys(layer_idx, args)
        for target_key in target_keys:
            a_key = generate_lora_key(layer_idx, target_key, 'A', args)
            b_key = generate_lora_key(layer_idx, target_key, 'B', args)
            lora_state_dict[a_key] = new_A.clone()
            lora_state_dict[b_key] = new_B.clone()

            # Reconstruct original keys for output
            original_a_key = f"base_model.model.model.layers.{layer_idx}.control_A.weight"
            original_b_key = f"base_model.model.model.layers.{layer_idx}.control_B.weight"
            print(f"-- '{original_a_key}' -> '{a_key}'")
            print(f"-- '{original_b_key}' -> '{b_key}'")

    print(f"Done (total tensors: {len(control_state_dict)} -> {len(lora_state_dict)})")

    save_adapter_weights(lora_state_dict, output_path)