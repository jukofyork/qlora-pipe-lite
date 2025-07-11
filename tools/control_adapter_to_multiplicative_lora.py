#!/usr/bin/env python3
"""
This script converts Control Adapters to a multiplicative LoRA by distributing the
multiplicative effect to specific linear layers within each transformer block.

Usage: python control_adapter_to_multiplicative_lora.py control_adapter_path output_path [--cohere | --mixtral N]
"""

from pathlib import Path
import argparse
from control_adapter_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Control Adapters to Multiplicative LoRA format")
    parser.add_argument("control_adapter_path", type=str, help="The path to the Control Adapter directory.")
    parser.add_argument("output_path", type=str, help="The path to the LoRA directory.")
    add_model_args(parser)

    args = parser.parse_args()
    control_adapter_path = Path(args.control_adapter_path)
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy the config file and patch any fields required to turn it into a LoRA
    copy_and_patch_adapter_config(control_adapter_path, output_path, args)

    control_keys, control_state_dict = load_control_adapter_weights(control_adapter_path)

    lora_state_dict = {}

    layer_data = parse_control_adapter_keys(control_state_dict)

    print(f"Converting {len(control_keys)} control adapter tensors:")
    for layer_idx in sorted(layer_data.keys()):
        for adapter_type in ['A', 'B']:  # Process A then B for each layer
            if adapter_type not in layer_data[layer_idx]:
                continue

            tensor = layer_data[layer_idx][adapter_type]
            target_keys = generate_model_weight_keys(layer_idx, args)
            for target_key in target_keys:
                new_key = generate_lora_key(layer_idx, target_key, adapter_type, args)
                lora_state_dict[new_key] = tensor.clone()

                # Reconstruct original key for output
                original_key = f"base_model.model.model.layers.{layer_idx}.control_{adapter_type}.weight"
                print(f"- '{original_key}' -> '{new_key}'")

    print(f"Done (total tensors: {len(control_state_dict)} -> {len(lora_state_dict)})")

    save_adapter_weights(lora_state_dict, output_path)