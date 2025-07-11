#!/usr/bin/env python3
"""
This script converts Control Adapters to standard LoRA format by distributing the multiplicative
effect to specific linear layers within each transformer block, enabling use with standard LoRA
merging tools.

Usage: python convert_control_adapter_to_lora.py input_path output_path [--cohere | --mixtral N]

---

Control Adapters work by applying a multiplicative transform to the decoder layer's output:

    decoder_output = decoder_output + (scale * B @ A @ decoder_output)

This multiplicative effect can be distributed to specific linear layers within each block:

1. LLAMA-TYPE MODELS - Target `mlp.down_proj` only:

   In Llama-style architectures, the MLP processes the post-attention residual:

   ```
   residual = hidden_states
   hidden_states = self.input_layernorm(hidden_states)
   hidden_states, _ = self.self_attn(...)  # attention
   hidden_states = residual + hidden_states

   residual = hidden_states
   hidden_states = self.post_attention_layernorm(hidden_states)
   hidden_states = self.mlp(hidden_states)  # down_proj is the final layer here
   hidden_states = residual + hidden_states
   ```

   Since down_proj is the final transformation before the residual addition, applying the
   multiplicative LoRA here achieves a similar effect, though not identical due to `o_proj`
   adding to the residual stream mid-layer.

2. COHERE MODELS - Target both `self_attn.o_proj` AND `mlp.down_proj`:

   Cohere runs attention and MLP in parallel, then sums both contributions:

   ```
   residual = hidden_states
   hidden_states = self.input_layernorm(hidden_states)
   hidden_states_attention, _ = self.self_attn(...)  # o_proj is final layer here
   hidden_states_mlp = self.mlp(...)                 # down_proj is final layer here
   hidden_states = residual + hidden_states_attention + hidden_states_mlp
   ```

   Since both paths contribute to the final residual, we need a multiplicative LoRA on
   both `o_proj` and `down_proj` to exactly capture the full Control Adapter effect.

3. MIXTRAL MODELS - Target all `block_sparse_moe.experts.{0..N-1}.w2`:

   In Mixtral's MoE structure, `w2` serves the same role as `down_proj`:

   ```
   current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
   current_hidden_states = self.w2(current_hidden_states)  # w2 ≡ down_proj
   ```

   Each expert's `w2` is the final linear transformation, so applying multiplicative LoRA
   to all expert `w2` layers achieves a similar effect to the Control Adapter, though not
   identical due to `o_proj` adding to the residual stream mid-layer.

    Due to the linearity of matrix operations, this works regardless of router weights or
    top-k selection:

       MoE(x) = sum of (weight_i * Expert_i(x))

    Applying the same multiplicative effect to each Expert_i preserves this weighted sum.
"""

from pathlib import Path
import argparse
import json
import re
import safetensors
import safetensors.torch
import shutil
import torch

def main():
    parser = argparse.ArgumentParser(description="Convert Control Adapters to Multiplicative LoRA format")
    parser.add_argument("input_path", type=str, help="The path to the Control Adapter directory.")
    parser.add_argument("output_path", type=str, help="The path to the LoRA directory.")

    # Model-specific options (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--cohere", action="store_true", help="Also target o_proj for Cohere models")
    model_group.add_argument("--mixtral", type=int, metavar="N", help="Target experts.{0..N-1}.w2 for Mixtral models")

    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Copy and modify adapter_config.json
    config_filename = 'adapter_config.json'
    if not (input_path / config_filename).exists():
        raise FileNotFoundError(f"{config_filename} not found in input directory")

    # Load, modify, and save config
    with open(input_path / config_filename, 'r') as f:
        config = json.load(f)

    # Update target_modules based on model type
    if args.mixtral:
        config['target_modules'] = ["w2"]
    elif args.cohere:
        config['target_modules'] = ["down_proj", "o_proj"]
    else:
        config['target_modules'] = ["down_proj"]

    with open(output_path / config_filename, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Updated and copied {config_filename}")

    # Find adapter model file
    adapter_filename = None
    for filename in ['adapter_model.safetensors', 'adapter_model.bin']:
        if (input_path / filename).exists():
            adapter_filename = filename
            break
    else:
        raise FileNotFoundError("No adapter_model.safetensors or adapter_model.bin found in input directory")

    # Load the adapter weights
    if Path(adapter_filename).suffix == '.safetensors':
        state_dict = safetensors.torch.load_file(input_path / adapter_filename)
    else:
        state_dict = torch.load(input_path / adapter_filename, map_location='cpu', weights_only=True)

    # Convert control adapters to multiplicative LoRA format
    new_state_dict = {}

    # Collect and sort control adapter keys by layer number
    control_keys = list(state_dict.items())

    # Sort control keys by layer number
    def extract_layer_num(item):
        key, _ = item
        match = re.search(r'layers\.(\d+)\.control_([AB])\.weight', key)
        if match:
            return (int(match.group(1)), match.group(2))  # Sort by layer number, then A/B
        return (float('inf'), '')  # Put unparseable keys at the end

    control_keys.sort(key=extract_layer_num)

    # Process control adapters in sorted order
    print(f"Converting {len(control_keys)} control adapter tensors:")
    for key, tensor in control_keys:
        # Extract layer index from the key
        # Pattern: base_model.model.model.layers.{layer_idx}.control_{A|B}.weight
        match = re.search(r'layers\.(\d+)\.control_([AB])\.weight', key)
        if match:
            layer_idx = match.group(1)
            adapter_type = match.group(2)

            base_key = f"base_model.model.model.layers.{layer_idx}"
            lora_suffix = f"lora_{adapter_type}.weight"

            if args.mixtral:
                # Target all experts' w2 for Mixtral models
                for expert_idx in range(args.mixtral):
                    new_key = f"{base_key}.block_sparse_moe.experts.{expert_idx}.w2.{lora_suffix}"
                    new_state_dict[new_key] = tensor.clone()
                    print(f"- '{key}' -> '{new_key}'")
            else:
                # Default: target mlp.down_proj
                down_proj_key = f"{base_key}.mlp.down_proj.{lora_suffix}"
                new_state_dict[down_proj_key] = tensor.clone()
                print(f"- '{key}' -> '{down_proj_key}'")

                if args.cohere:
                    # Also target self_attn.o_proj for Cohere models
                    o_proj_key = f"{base_key}.self_attn.o_proj.{lora_suffix}"
                    new_state_dict[o_proj_key] = tensor.clone()
                    print(f"- '{key}' -> '{o_proj_key}'")
        else:
            raise ValueError(f"Could not parse control adapter key: {key}")

    print(f"Done (total tensors: {len(state_dict)} -> {len(new_state_dict)})")

    # Save converted adapter weights
    if Path(adapter_filename).suffix == '.safetensors':
        safetensors.torch.save_file(new_state_dict, output_path / adapter_filename)
    else:
        torch.save(new_state_dict, output_path / adapter_filename)

    print(f"Converted adapter saved to: '{output_path}'")

if __name__ == "__main__":
    main()