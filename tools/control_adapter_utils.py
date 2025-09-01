# tools/control_adapter_utils.py
"""
Shared utilities for control adapter conversion scripts.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any
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

def apply_control_adapter_transform(Q, lambda_vec, scale_factor, device=None):
    """Apply Control Adapter transformation: scale_factor * (Q @ diag(λ) @ Q^T)."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move to specified device and compute
    Q_gpu = Q.to(device=device, dtype=torch.float32)
    lambda_gpu = lambda_vec.to(device=device, dtype=torch.float32)

    # Compute Q @ diag(λ) @ Q^T
    # This is equivalent to: Q @ torch.diag(lambda_gpu) @ Q.T
    # But more memory efficient as: (Q * lambda_gpu.unsqueeze(0)) @ Q.T
    result = scale_factor * ((Q_gpu * lambda_gpu.unsqueeze(0)) @ Q_gpu.T)

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
        """Extract layer number and parameter type from control adapter key for sorting."""
        key, _ = item
        # Extract layer number and parameter type for sorting
        match = re.search(r'layers\.(\d+)\.control_(Q|lambda)', key)
        if match:
            layer_num = int(match.group(1))
            param_type = match.group(2)
            # Sort Q before lambda for consistent ordering
            return (layer_num, 0 if param_type == 'Q' else 1)
        return (float('inf'), 2)

    state_dict, _ = _find_and_load_adapter_weights(adapter_path)
    control_keys = list(state_dict.items())
    control_keys.sort(key=_extract_layer_num)
    return control_keys, state_dict

def parse_control_adapter_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[int, Dict[str, torch.Tensor]]:
    """Parse control adapter state dict into layer -> {Q, lambda} mapping."""
    control_adapters = {}

    for key, tensor in state_dict.items():
        match = re.search(r'layers\.(\d+)\.control_(Q|lambda)', key)
        if match:
            layer_idx = int(match.group(1))
            param_type = match.group(2)

            if layer_idx not in control_adapters:
                control_adapters[layer_idx] = {}
            control_adapters[layer_idx][param_type] = tensor
        else:
            raise ValueError(f"Could not parse control adapter key: {key}")

    return control_adapters