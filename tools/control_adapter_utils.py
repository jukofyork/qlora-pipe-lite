# tools/control_adapter_utils.py
"""
Shared utilities for control adapter conversion scripts.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import argparse
import json
import re
import safetensors.torch
import torch

def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add model-specific argument group to parser."""
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--cohere", action="store_true", help="Also target o_proj for Cohere models")
    model_group.add_argument("--mixtral", type=int, metavar="N", help="Target experts.{0..N-1}.w2 for Mixtral models")

def copy_and_patch_adapter_config(input_path: Path, output_path: Path, args) -> int:
    """Copy adapter config, patch target_modules based on model type, and return rank."""
    config_file = input_path / 'adapter_config.json'
    if not config_file.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {input_path}")

    with open(config_file, 'r') as f:
        config = json.load(f)

    # Set target modules based on model type
    if args.mixtral:
        config['target_modules'] = ["w2"]
    elif args.cohere:
        config['target_modules'] = ["down_proj", "o_proj"]
    else:
        config['target_modules'] = ["down_proj"]

    with open(output_path / 'adapter_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print("Updated and copied adapter_config.json")

    return config['r']

def load_control_adapter_weights(adapter_path: Path) -> Tuple[List[Tuple[str, torch.Tensor]], Dict[str, torch.Tensor]]:
    """Load control adapter weights and return (sorted key-value pairs, full state dict)."""

    def _find_and_load_adapter_weights(adapter_path: Path) -> Tuple[Dict[str, torch.Tensor], str]:
        """Find and load control adapter weights, returning weights and filename."""
        for filename in ['adapter_model.safetensors', 'adapter_model.bin']:
            filepath = adapter_path / filename
            if filepath.exists():
                if filename.endswith('.safetensors'):
                    weights = safetensors.torch.load_file(filepath)
                else:
                    weights = torch.load(filepath, map_location='cpu', weights_only=True)
                return weights, filename

        raise FileNotFoundError("No adapter_model.safetensors or adapter_model.bin found in adapter directory")

    def _extract_layer_num(item):
        """Extract layer number and adapter type from control adapter key for sorting."""
        key, _ = item
        match = re.search(r'layers\.(\d+)\.control_([AB])\.weight', key)
        if match:
            return (int(match.group(1)), match.group(2))
        return (float('inf'), '')

    state_dict, _ = _find_and_load_adapter_weights(adapter_path)
    control_keys = list(state_dict.items())
    control_keys.sort(key=_extract_layer_num)
    return control_keys, state_dict

def parse_control_adapter_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[int, Dict[str, torch.Tensor]]:
    """Parse control adapter state dict into layer -> {A, B} mapping."""
    control_adapters = {}

    for key, tensor in state_dict.items():
        match = re.search(r'layers\.(\d+)\.control_([AB])\.weight', key)
        if match:
            layer_idx = int(match.group(1))
            adapter_type = match.group(2)

            if layer_idx not in control_adapters:
                control_adapters[layer_idx] = {}
            control_adapters[layer_idx][adapter_type] = tensor
        else:
            raise ValueError(f"Could not parse control adapter key: {key}")

    return control_adapters

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

def save_adapter_weights(new_state_dict: Dict, output_path: Path) -> None:
    output_file = output_path / 'adapter_model.safetensors'
    safetensors.torch.save_file(new_state_dict, output_file)
    print(f"Converted LoRA adapter saved to: '{output_path}'")