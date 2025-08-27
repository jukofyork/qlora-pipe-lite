#!/usr/bin/env python3
"""
This script converts Control Adapters to an approximate additive LoRA by computing
the multiplicative effect on actual model weights, then approximating the delta via
an SVD decomposition.

Usage: python control_adapter_to_additive_lora.py base_model_path control_adapter_path output_path
"""

from pathlib import Path
import argparse
import torch

from control_adapter_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Control Adapters to Additive LoRA format via SVD")
    parser.add_argument("base_model_path", type=str, help="Path to the base model directory")
    parser.add_argument("control_adapter_path", type=str, help="Path to the Control Adapter directory")
    parser.add_argument("output_path", type=str, help="Path to the output LoRA directory")
    parser.add_argument("--rank", type=int, help="Override rank for SVD truncation (default: use original rank)")
    parser.add_argument("--no-gpu", action="store_true", help="Use CPU for SVD computation")
    add_model_args(parser)

    args = parser.parse_args()
    svd_device = "cpu" if args.no_gpu or not torch.cuda.is_available() else "cuda"
    control_adapter_path = Path(args.control_adapter_path)
    base_model_path = Path(args.base_model_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    scale_factor, target_rank, _ = copy_and_patch_adapter_config(control_adapter_path, output_path, args)

    control_keys, control_state_dict = load_control_adapter_weights(control_adapter_path)

    model_weights = load_model_weights(base_model_path)

    lora_state_dict = {}

    layer_data = parse_control_adapter_keys(control_state_dict)

    print(f"Converting {len(control_keys)} control adapter tensors (device='{svd_device}', scale={scale_factor:.4f}):")

    for layer_idx in sorted(layer_data.keys()):
        if 'A' not in layer_data[layer_idx] or 'B' not in layer_data[layer_idx]:
            continue

        A = layer_data[layer_idx]['A']
        B = layer_data[layer_idx]['B']
        old_type = A.dtype

        # Apply the original alpha/rank scaling factor
        lora_delta = scale_factor * (B.to(torch.float32) @ A.to(torch.float32))

        target_keys = generate_model_weight_keys(layer_idx, args)

        for target_key in target_keys:
            if target_key not in model_weights:
                continue

            weight = model_weights[target_key].to(torch.float32)
            multiplicative_effect = lora_delta @ weight

            effect_gpu = multiplicative_effect.to(svd_device)
            U, S, Vt = torch.linalg.svd(effect_gpu, full_matrices=False)
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
            print(f"- Layer {layer_idx}, SVD rank {target_rank}/{len(S)}, "
                  f"{100*torch.sum(S[:target_rank]**2)/torch.sum(S**2):.4f}% of variance explained:")
            print(f"-- '{base_key}.control_A.weight' -> '{a_key}'")
            print(f"-- '{base_key}.control_B.weight' -> '{b_key}'")

    print(f"Done (total tensors: {len(control_state_dict)} -> {len(lora_state_dict)})")

    save_adapter_weights(lora_state_dict, output_path)