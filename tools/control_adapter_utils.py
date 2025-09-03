# tools/control_adapter_utils.py
"""
Shared utilities for control adapter conversion scripts.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import re
import safetensors.torch
import torch

def apply_control_adapter_transform(Q, S, device=None):
    """Apply forward delta: Q diag(λ) Q^T, with λ = exp(S) - 1.

    Returns:
        Tensor: Q diag(λ) Q^T on the specified device (float32)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move to specified device and compute
    Q_gpu = Q.to(device=device, dtype=torch.float32)
    S_gpu = S.to(device=device, dtype=torch.float32)

    # Compute λ = exp(S) - 1
    lambda_vec = torch.expm1(S_gpu)

    # Memory-efficient construct: (Q * λ.unsqueeze(0)) @ Q.T
    result = (Q_gpu * lambda_vec.unsqueeze(0)) @ Q_gpu.T

    return result

def apply_control_adapter_inverse_transform(Q, S, device=None):
    """Apply exact inverse delta: (I + Q diag(λ) Q^T)^{-1} - I, with λ = exp(S) - 1.
                                  = Q diag(λ') Q^T, with λ' = exp(-S) - 1.

    Returns:
        Tensor: Q diag(λ') Q^T on the specified device (float32)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move to specified device and compute
    Q_gpu = Q.to(device=device, dtype=torch.float32)
    S_gpu = S.to(device=device, dtype=torch.float32)

    # Exact inverse coefficient: λ' = -λ/(1+λ) = exp(-S) - 1
    lambda_inv_vec = torch.expm1(-S_gpu)

    # Memory-efficient construct: (Q * λ'.unsqueeze(0)) @ Q.T
    result = (Q_gpu * lambda_inv_vec.unsqueeze(0)) @ Q_gpu.T
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
        match = re.search(r'layers\.(\d+)\.control_(Q|S)', key)
        if match:
            layer_num = int(match.group(1))
            param_type = match.group(2)
            # Sort Q before S for consistent ordering
            return (layer_num, 0 if param_type == 'Q' else 1)
        return (float('inf'), 2)

    state_dict, _ = _find_and_load_adapter_weights(adapter_path)
    control_keys = list(state_dict.items())
    control_keys.sort(key=_extract_layer_num)
    return control_keys, state_dict

def parse_control_adapter_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[int, Dict[str, torch.Tensor]]:
    """Parse control adapter state dict into layer -> {Q, S} mapping."""
    control_adapters = {}

    for key, tensor in state_dict.items():
        match = re.search(r'layers\.(\d+)\.control_(Q|S)', key)
        if match:
            layer_idx = int(match.group(1))
            param_type = match.group(2)

            if layer_idx not in control_adapters:
                control_adapters[layer_idx] = {}
            control_adapters[layer_idx][param_type] = tensor
        else:
            raise ValueError(f"Could not parse control adapter key: {key}")

    return control_adapters