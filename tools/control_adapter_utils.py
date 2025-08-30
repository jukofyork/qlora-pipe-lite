# tools/control_adapter_utils.py
"""
Shared utilities for control adapter conversion scripts.
"""

from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
import argparse
import json
import re
import safetensors.torch
import torch

def load_adapter_config(adapter_path: Path) -> Dict[str, Any]:
    """Load adapter configuration."""
    config_path = adapter_path / 'adapter_config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")

    with open(config_path, 'r') as f:
        return json.load(f)

def get_control_adapter_type(config):
    """Validate and return control_adapter_type from config."""
    control_adapter_type = config.get('control_adapter_type', 'full')
    allowed_types = ['full', 'symmetrise', 'antisymmetrise']
    if control_adapter_type not in allowed_types:
        raise ValueError(f"Invalid control_adapter_type '{control_adapter_type}'. Must be one of: {allowed_types}")
    return control_adapter_type

def apply_control_adapter_transform(A, B, scale_factor, control_adapter_type, device=None):
    """Apply Control Adapter transformation with scaling and symmetry operations."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move to specified device and compute scaled composite matrix
    A_gpu = A.to(device=device, dtype=torch.float32)
    B_gpu = B.to(device=device, dtype=torch.float32)
    result = scale_factor * (B_gpu @ A_gpu)

    # Apply transformation based on control_adapter_type
    if control_adapter_type == "symmetrise":
        result = 0.5 * (result + result.transpose(-2, -1))
    elif control_adapter_type == "antisymmetrise":
        result = 0.5 * (result - result.transpose(-2, -1))

    return result

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