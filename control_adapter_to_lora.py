#!/usr/bin/env python3
"""
Convert Control Adapters to an approximate additive LoRA by computing the
multiplicative effect on model weights, then approximating the delta via SVD.

USAGE:
    python control_adapter_to_lora.py --base /path/to/base_model --adapter /path/to/control_adapter --output /path/to/out_dir
                                      [--rank N] [--inverse] [--no-gpu] [--cohere | --mixtral N]
"""

from pathlib import Path
import argparse
import safetensors
import safetensors.torch
import torch

from training.control_adapters import (
    apply_control_adapter_transform,
    load_control_adapter_weights,
    parse_control_adapter_keys,
    copy_and_patch_adapter_config,
    generate_model_weight_keys,
    generate_lora_key,
    load_model_weights,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Control Adapters to Additive LoRA format via SVD")
    parser.add_argument("--base", required=True, type=str, help="Path to the base model directory")
    parser.add_argument("--adapter", required=True, type=str, help="Path to the Control Adapter directory")
    parser.add_argument("--output", required=True, type=str, help="Path to the output LoRA directory")
    parser.add_argument("--rank", type=int, help="Override rank for SVD truncation (default: use original rank)")
    parser.add_argument("--no-gpu", action="store_true", help="Use CPU for SVD computation")
    parser.add_argument("--inverse", action="store_true", help="Use exact inverse delta: (I + W)^{-1} - I")
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

    target_rank = copy_and_patch_adapter_config(control_adapter_path, output_path, args)

    control_keys, control_state_dict = load_control_adapter_weights(control_adapter_path)
    model_weights = load_model_weights(base_model_path)

    lora_state_dict = {}

    layer_data = parse_control_adapter_keys(control_state_dict)

    print()
    print(f"Converting {len(control_keys)//2} layers ({len(control_keys)} tensors) (device='{device}'):")

    for layer_idx in sorted(layer_data.keys()):
        if 'Q' not in layer_data[layer_idx] or 'S' not in layer_data[layer_idx]:
            continue

        Q = layer_data[layer_idx]['Q']
        S = layer_data[layer_idx]['S']

        with torch.no_grad():
            old_type = Q.dtype

            # Choose forward or exact inverse delta via flag
            lora_delta = apply_control_adapter_transform(Q, S, device=device, inverse=args.inverse)

            target_keys = generate_model_weight_keys(layer_idx, args)

            for target_key in target_keys:
                if target_key not in model_weights:
                    continue

                weight = model_weights[target_key].to(device=device, dtype=torch.float32)
                multiplicative_effect = lora_delta @ weight

                U, svals, Vt = torch.linalg.svd(multiplicative_effect, full_matrices=False)
                U, svals, Vt = U.cpu(), svals.cpu(), Vt.cpu()

                # Truncate to target rank
                if target_rank > len(svals):
                    raise ValueError(f"Requested rank {target_rank} exceeds maximum available rank {len(svals)} from SVD")
                sqrt_svals = torch.sqrt(svals[:target_rank])
                A_approx = torch.diag(sqrt_svals) @ Vt[:target_rank,:]
                B_approx = U[:,:target_rank] @ torch.diag(sqrt_svals)

                A_approx = A_approx.to(old_type)
                B_approx = B_approx.to(old_type)

                a_key = generate_lora_key(layer_idx, target_key, 'A', args)
                b_key = generate_lora_key(layer_idx, target_key, 'B', args)
                lora_state_dict[a_key] = A_approx
                lora_state_dict[b_key] = B_approx

                base_key = f"base_model.model.model.layers.{layer_idx}"
                print()
                print(f"- Layer {layer_idx}, SVD rank {target_rank}/{len(svals)}, "
                      f"{100 * torch.sum(svals[:target_rank] ** 2) / torch.sum(svals ** 2):.1f}% of variance explained:")
                print(f"  -- '{base_key}.control_Q' + '{base_key}.control_S'")
                print(f"  -> '{a_key}' + '{b_key}'")

    print()
    print(f"Done (total tensors: {len(control_state_dict)} -> {len(lora_state_dict)})")

    safetensors.torch.save_file(lora_state_dict, output_path / 'adapter_model.safetensors')
    print()
    print(f"Converted LoRA adapter saved to: '{output_path}'")