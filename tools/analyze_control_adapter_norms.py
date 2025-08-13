#!/usr/bin/env python3
"""
Minimal script to analyze Control Adapter norm constraints for convergence.
Prints statistics about ||W||_F, ||W||_2, and convergence requirements.

Usage: python analyze_control_adapter_norms.py control_adapter_path
"""

import argparse
import json
import math
from pathlib import Path
import re
import safetensors
import torch
from typing import Dict, Any

def load_control_adapter_weights(adapter_path: Path) -> Dict[str, torch.Tensor]:
    """Load Control Adapter weights from safetensors or .bin files."""
    state_dict = {}
    
    # Try safetensors first
    adapter_files = list(adapter_path.glob('adapter*.safetensors'))
    
    if adapter_files:
        for file_path in adapter_files:
            with safetensors.safe_open(file_path, framework='pt', device='cpu') as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)
    else:
        # Try .bin files
        adapter_files = list(adapter_path.glob('adapter*.bin'))
        if not adapter_files:
            # Try pytorch_model.bin as fallback
            adapter_files = list(adapter_path.glob('pytorch_model*.bin'))
        
        if not adapter_files:
            raise FileNotFoundError(f"No adapter*.safetensors, adapter*.bin, or pytorch_model*.bin files found in {adapter_path}")
        
        for file_path in adapter_files:
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=True)
            state_dict.update(checkpoint)
    
    return state_dict

def parse_control_adapter_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[int, Dict[str, torch.Tensor]]:
    """Parse control adapter keys and group by layer."""
    layer_data = {}
    
    for key, tensor in state_dict.items():
        # Match patterns like "base_model.model.model.layers.0.control_A.weight"
        match = re.search(r'layers\.(\d+)\.control_([AB])\.weight', key)
        if match:
            layer_idx = int(match.group(1))
            matrix_type = match.group(2)
            
            if layer_idx not in layer_data:
                layer_data[layer_idx] = {}
            layer_data[layer_idx][matrix_type] = tensor
    
    return layer_data

def load_adapter_config(adapter_path: Path) -> Dict[str, Any]:
    """Load adapter configuration."""
    config_path = adapter_path / 'adapter_config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)

def analyze_layer_norms(A: torch.Tensor, B: torch.Tensor, lora_scale: float, rank: int) -> Dict[str, float]:
    """Analyze norms for a single layer."""
    # Compute composite matrix W = lora_scale * (B @ A)
    W = lora_scale * (B.to(torch.float32) @ A.to(torch.float32))
    
    # Use SVD for accurate norm calculations
    # NOTE: We use SVD instead of torch.norm() or torch.linalg.matrix_norm() because
    # those functions gave inconsistent results that violated ‖W‖₂ ≤ ‖W‖_F.
    # Computing from singular values directly ensures mathematical correctness.
    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
    
    # Compute norms from singular values
    spectral_norm = S[0].item()  # Largest singular value
    frobenius_norm = torch.sqrt(torch.sum(S**2)).item()  # sqrt(sum of squared singular values)
    
    sqrt_rank = math.sqrt(rank)
    
    return {
        'frobenius_norm': frobenius_norm,
        'spectral_norm': spectral_norm,
        'sqrt_rank': sqrt_rank,
        'frobenius_over_sqrt_rank': frobenius_norm / sqrt_rank,
        'spectral_over_frobenius': spectral_norm / frobenius_norm,
        'converges': spectral_norm < 1.0,
        'meets_guideline': frobenius_norm < 0.25 * sqrt_rank,
        'optimal_spectral': spectral_norm < 0.25
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Control Adapter norm constraints")
    parser.add_argument("control_adapter_path", type=str, help="Path to the Control Adapter directory")
    args = parser.parse_args()
    
    adapter_path = Path(args.control_adapter_path)
    
    # Load configuration and weights
    config = load_adapter_config(adapter_path)
    state_dict = load_control_adapter_weights(adapter_path)
    
    # Extract parameters
    lora_alpha = config['lora_alpha']
    lora_rank = config['r']
    lora_scale = lora_alpha / lora_rank
    
    print(f"Control Adapter Analysis")
    print(f"========================")
    print(f"Path: {adapter_path}")
    print(f"LoRA Alpha: {lora_alpha}")
    print(f"LoRA Rank: {lora_rank}")
    print(f"LoRA Scale: {lora_scale:.4f}")
    print("Target ‖W‖₂  < 0.25 (keeps 2nd order Neumann truncation error O(‖s·AB‖₂³) ≤ 1-2%)")
    print(f"Target ‖W‖_F < {0.25 * math.sqrt(lora_rank):.3f} (cheaper monitoring targe0t: 0.25·√r)")
    print()
    
    # Parse layers
    layer_data = parse_control_adapter_keys(state_dict)
    
    if not layer_data:
        print("No control adapter layers found!")
        exit(1)
    
    # Analyze each layer
    results = []
    print(f"{'Layer':<6} {'‖W‖₂':<8} {'‖W‖_F':<8} {'‖W‖_F/√r':<10} {'‖W‖₂/‖W‖_F':<12} {'Target':<9} {'Stable':<9}")
    print("-" * 76)
    
    for layer_idx in sorted(layer_data.keys()):
        if 'A' not in layer_data[layer_idx] or 'B' not in layer_data[layer_idx]:
            continue
            
        A = layer_data[layer_idx]['A']
        B = layer_data[layer_idx]['B']
        
        stats = analyze_layer_norms(A, B, lora_scale, lora_rank)
        results.append(stats)
        
        print(f"{layer_idx:<6} {stats['spectral_norm']:<8.3f} {stats['frobenius_norm']:<8.3f} "
              f"{stats['frobenius_over_sqrt_rank']:<10.3f} {stats['spectral_over_frobenius']:<12.3f} "
              f"{'✓' if stats['optimal_spectral'] else '✗':<9} {'✓' if stats['converges'] else '✗':<9}")
    
    # Summary statistics
    if results:
        frobenius_norms = [r['frobenius_norm'] for r in results]
        spectral_norms = [r['spectral_norm'] for r in results]
        frobenius_ratios = [r['frobenius_over_sqrt_rank'] for r in results]
        
        converging_layers = sum(1 for r in results if r['converges'])
        optimal_layers = sum(1 for r in results if r['optimal_spectral'])
        
        print()
        print("Summary Statistics")
        print("==================")
        print(f"Total layers: {len(results)}")
        print(f"Stable layers (‖W‖₂ < 1): {converging_layers}/{len(results)} ({100*converging_layers/len(results):.1f}%)")
        print(f"Optimal layers (‖W‖₂ < 0.25): {optimal_layers}/{len(results)} ({100*optimal_layers/len(results):.1f}%)")
        print()
        print(f"‖W‖_F  - Min: {min(frobenius_norms):.3f}, Max: {max(frobenius_norms):.3f}, Mean: {sum(frobenius_norms)/len(frobenius_norms):.3f}")
        print(f"‖W‖₂   - Min: {min(spectral_norms):.3f}, Max: {max(spectral_norms):.3f}, Mean: {sum(spectral_norms)/len(spectral_norms):.3f}")
        print(f"‖W‖_F/√r - Min: {min(frobenius_ratios):.3f}, Max: {max(frobenius_ratios):.3f}, Mean: {sum(frobenius_ratios)/len(frobenius_ratios):.3f}")
        
        if max(spectral_norms) >= 1.0:
            print()
            print("⚠️  WARNING: Some layers have ‖W‖₂ ≥ 1, which may cause convergence issues!")
        
        if min(spectral_norms) >= 0.25:
            print("⚠️  WARNING: All layers exceed the optimal ‖W‖₂ < 0.25 target for best approximation!")
        elif max(spectral_norms) >= 0.25:
            print("⚠️  WARNING: Some layers exceed the optimal ‖W‖₂ < 0.25 target for best approximation!")