#!/usr/bin/env python3
"""
Script to analyze Control Adapter matrix norms and statistical properties.
Computes spectral norm (||W||_2), nuclear norm (||W||_*), and Frobenius norm (||W||_F)
along with their ratios for each layer.

Usage: python analyze_control_adapters.py control_adapter_path
"""

from pathlib import Path
import argparse
import torch

from control_adapter_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Control Adapter norm constraints")
    parser.add_argument("control_adapter_path", type=str, help="Path to the Control Adapter directory")
    args = parser.parse_args()

    adapter_path = Path(args.control_adapter_path)

    # Load configuration and weights
    adapter_config = load_adapter_config(adapter_path)
    _, state_dict = load_control_adapter_weights(adapter_path)

    # Extract parameters
    lora_alpha = adapter_config['lora_alpha']
    lora_rank = adapter_config['r']
    lora_scale = lora_alpha / lora_rank

    print(f"Control Adapter Analysis")
    print(f"========================")
    print(f"Path: {adapter_path}")
    print(f"LoRA Alpha: {lora_alpha}")
    print(f"LoRA Rank: {lora_rank}")
    print(f"LoRA Scale: {lora_scale:.4f}")
    print()

    # Parse layers using utils function
    layer_data = parse_control_adapter_keys(state_dict)

    if not layer_data:
        print("No control adapter layers found!")
        exit(1)

    # Analyze each layer
    results = []
    print(f"{'Layer':<6} {'‖W‖_2':<8} {'‖W‖_*':<8} {'‖W‖_F':<8} {'‖W‖_2/‖W‖_*':<12} {'‖W‖_2/‖W‖_F':<12}")
    print("-" * 58)

    for layer_idx in sorted(layer_data.keys()):
        if 'lambda' not in layer_data[layer_idx]:
            continue

        lambda_vec = layer_data[layer_idx]['lambda'].to(torch.float32)

        # For Control Adapters: W = scale * Q @ diag(λ) @ Q^T
        # Since Q is orthogonal and W is symmetric with eigenvalues λ:
        # - Spectral norm = scale * max(|λ_i|)
        # - Frobenius norm = scale * ||λ||_2
        # - Nuclear norm = scale * ||λ||_1
        spectral_norm = lora_scale * torch.max(torch.abs(lambda_vec)).item()
        frobenius_norm = lora_scale * torch.norm(lambda_vec).item()
        nuclear_norm = lora_scale * torch.sum(torch.abs(lambda_vec)).item()

        # Add guards for division by zero
        spectral_over_nuclear = spectral_norm / nuclear_norm if nuclear_norm > 1e-10 else float('inf')
        spectral_over_frobenius = spectral_norm / frobenius_norm if frobenius_norm > 1e-10 else float('inf')

        stats = {
            'spectral_norm': spectral_norm,
            'nuclear_norm': nuclear_norm,
            'frobenius_norm': frobenius_norm,
            'spectral_over_nuclear': spectral_over_nuclear,
            'spectral_over_frobenius': spectral_over_frobenius
        }
        results.append(stats)

        print(f"{layer_idx:<6} "
              f"{stats['spectral_norm']:<8.2f} {stats['nuclear_norm']:<8.2f} {stats['frobenius_norm']:<8.2f} "
              f"{stats['spectral_over_nuclear']:<12.3f} {stats['spectral_over_frobenius']:<12.3f}")

    # Summary statistics
    if results:
        spectral_norms = [r['spectral_norm'] for r in results]
        nuclear_norms = [r['nuclear_norm'] for r in results]
        frobenius_norms = [r['frobenius_norm'] for r in results]

        print()
        print("Summary Statistics")
        print("==================")
        print(f"Total layers: {len(results)}")
        print()
        print(f"‖W‖_2 --> Min: {min(spectral_norms):.3f}, Max: {max(spectral_norms):.3f}, "
              f"Mean: {sum(spectral_norms)/len(spectral_norms):.3f}")
        print(f"‖W‖_* --> Min: {min(nuclear_norms):.3f}, Max: {max(nuclear_norms):.3f}, "
              f"Mean: {sum(nuclear_norms)/len(nuclear_norms):.3f}")
        print(f"‖W‖_F --> Min: {min(frobenius_norms):.3f}, Max: {max(frobenius_norms):.3f}, "
              f"Mean: {sum(frobenius_norms)/len(frobenius_norms):.3f}")