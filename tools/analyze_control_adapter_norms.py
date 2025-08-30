#!/usr/bin/env python3
"""
Minimal script to analyze Control Adapter norm constraints for convergence.
Prints statistics about ||W||_F, ||W||_2, and convergence requirements.

Usage: python analyze_control_adapter_norms.py control_adapter_path
"""

from pathlib import Path
from typing import Dict, Any
import argparse
import json
import math
import re
import torch

from control_adapter_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Control Adapter norm constraints")
    parser.add_argument("control_adapter_path", type=str, help="Path to the Control Adapter directory")
    parser.add_argument("--no-gpu", action="store_true", help="Use CPU for SVD computation")
    args = parser.parse_args()

    # Move to GPU if available for faster computation
    device = "cpu" if args.no_gpu or not torch.cuda.is_available() else "cuda"

    adapter_path = Path(args.control_adapter_path)

    # Load configuration and weights
    adapter_config = load_adapter_config(adapter_path)
    _, state_dict = load_control_adapter_weights(adapter_path)  # Use utils function, take state_dict

    # Validate and get the control_adapter_type
    control_adapter_type = get_control_adapter_type(adapter_config)

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
    print(f"{'Layer':<6} {'‖W‖_2':<8} {'‖W‖_*':<8} {'‖W‖_F':<8} {'‖W‖_2/‖W‖_*':<12} {'‖W‖_2/‖W‖_F':<12} {'‖W‖_F/√r':<10}")
    print("-" * 68)

    for layer_idx in sorted(layer_data.keys()):
        if 'A' not in layer_data[layer_idx] or 'B' not in layer_data[layer_idx]:
            continue

        A = layer_data[layer_idx]['A']
        B = layer_data[layer_idx]['B']
        W = apply_control_adapter_transform(A, B, lora_scale, control_adapter_type, device)

        # Use SVD for accurate norm calculations
        # NOTE: We use SVD instead of torch.norm() or torch.linalg.matrix_norm() because
        # those functions gave inconsistent results that violated ‖W‖_2 ≤ ‖W‖_F.
        # Computing from singular values directly ensures mathematical correctness.
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)

        # Compute norms from singular values (move back to CPU for final values)
        spectral_norm = S[0].item()  # Largest singular value
        nuclear_norm = torch.sum(S).item()  # Sum of all singular values
        frobenius_norm = torch.sqrt(torch.sum(S ** 2)).item()  # sqrt(sum of squared singular values)

        sqrt_rank = math.sqrt(lora_rank)

        stats = {
            'spectral_norm': spectral_norm,
            'nuclear_norm': nuclear_norm,
            'frobenius_norm': frobenius_norm,
            'sqrt_rank': sqrt_rank,
            'frobenius_over_sqrt_rank': frobenius_norm / sqrt_rank,
            'spectral_over_nuclear': spectral_norm / nuclear_norm,
            'spectral_over_frobenius': spectral_norm / frobenius_norm
        }
        results.append(stats)

        print(f"{layer_idx:<6} {stats['spectral_norm']:<8.2f} {stats['nuclear_norm']:<8.2f} "
              f"{stats['frobenius_norm']:<8.2f} {stats['spectral_over_nuclear']:<12.3f} "
              f"{stats['spectral_over_frobenius']:<12.3f} {stats['frobenius_over_sqrt_rank']:<10.3f}")

    # Summary statistics
    if results:
        spectral_norms = [r['spectral_norm'] for r in results]
        nuclear_norms = [r['nuclear_norm'] for r in results]
        frobenius_norms = [r['frobenius_norm'] for r in results]
        frobenius_ratios = [r['frobenius_over_sqrt_rank'] for r in results]

        print()
        print("Summary Statistics")
        print("==================")
        print(f"Total layers: {len(results)}")
        print()
        print(f"‖W‖_2    : Min: {min(spectral_norms):.3f}, Max: {max(spectral_norms):.3f}, "
              f"Mean: {sum(spectral_norms)/len(spectral_norms):.3f}")
        print(f"‖W‖_*    : Min: {min(nuclear_norms):.3f}, Max: {max(nuclear_norms):.3f}, "
              f"Mean: {sum(nuclear_norms)/len(nuclear_norms):.3f}")
        print(f"‖W‖_F    : Min: {min(frobenius_norms):.3f}, Max: {max(frobenius_norms):.3f}, "
              f"Mean: {sum(frobenius_norms)/len(frobenius_norms):.3f}")
        print(f"‖W‖_F/√r : Min: {min(frobenius_ratios):.3f}, Max: {max(frobenius_ratios):.3f}, "
              f"Mean: {sum(frobenius_ratios)/len(frobenius_ratios):.3f}")