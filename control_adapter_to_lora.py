#!/usr/bin/env python3
"""
Convert Control Adapters to an exact additive LoRA.

We exploit the low-rank structure:
  DeltaW = (Q diag(lambda) Q^T) @ W = (Q diag(lambda)) @ (Q^T W)
So LoRA B = Q diag(lambda), LoRA A = Q^T W, retaining the original adapter rank.

USAGE:
    python control_adapter_to_lora.py --base /path/to/base_model --adapter /path/to/control_adapter --output /path/to/out_dir
                                      [--inverse] [--cohere | --mixtral N]
"""

from pathlib import Path
import argparse
import safetensors
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Control Adapters to exact Additive LoRA (no SVD)")
    parser.add_argument("--base", required=True, type=str, help="Path to the base model directory")
    parser.add_argument("--adapter", required=True, type=str, help="Path to the Control Adapter directory")
    parser.add_argument("--output", required=True, type=str, help="Path to the output LoRA directory")
    parser.add_argument("--no-gpu", action="store_true", help="Use CPU for computation")
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

    copy_and_patch_adapter_config(control_adapter_path, output_path, args)

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
            m, r = Q.shape

            # Move to device as float32 for compute
            Q_d = Q.to(device=device, dtype=torch.float32)
            S_d = S.to(device=device, dtype=torch.float32)

            # Forward or exact inverse delta coefficients
            # forward:  lambda = exp(S) - 1
            # inverse:  lambda' = exp(-S) - 1 = -lambda/(1+lambda)
            coeff = torch.expm1(-S_d) if args.inverse else torch.expm1(S_d)  # [r]

            # Common B factor per layer: B = Q diag(coeff)  => scale columns of Q by coeff
            B_common = Q_d * coeff.unsqueeze(0)  # [m, r]

            target_keys = generate_model_weight_keys(layer_idx, args)

            for target_key in target_keys:
                if target_key not in model_weights:
                    continue

                # Load base weight and compute A = Q^T W
                W = model_weights[target_key].to(device=device, dtype=torch.float32)  # [m, n]
                A = Q_d.T @ W  # [r, n]

                # Cast back to original dtype and CPU for saving
                A_out = A.to(dtype=old_type, device='cpu')
                B_out = B_common.to(dtype=old_type, device='cpu')

                a_key = generate_lora_key(layer_idx, target_key, 'A', args)
                b_key = generate_lora_key(layer_idx, target_key, 'B', args)
                lora_state_dict[a_key] = A_out
                lora_state_dict[b_key] = B_out

                base_key = f"base_model.model.model.layers.{layer_idx}"
                print()
                print(f"- Layer {layer_idx}, retained rank {r}")
                print(f"  -- '{base_key}.control_Q' + '{base_key}.control_S'")
                print(f"  -> '{a_key}' + '{b_key}'")

    print()
    print(f"Done (total tensors: {len(control_state_dict)} -> {len(lora_state_dict)})")

    safetensors.torch.save_file(lora_state_dict, output_path / 'adapter_model.safetensors')
    print()
    print(f"Converted LoRA adapter saved to: '{output_path}'")