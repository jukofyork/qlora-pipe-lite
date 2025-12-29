#!/usr/bin/env python3
"""
Analyze Control Adapter matrix norms and convergence properties.

OVERVIEW:
    Analyzes Control Adapter matrices W = s·(B @ A) where:
    • A ∈ ℝ^{r×H} is the down-projection (control_A.weight)
    • B ∈ ℝ^{H×r} is the up-projection (control_B.weight)
    • s = lora_alpha / r is the LoRA scaling factor

    Computes matrix norms via SVD to verify convergence requirements for
    the Neumann series approximation used during training.

USAGE:
    python analyze_control_adapters.py --adapter /path/to/adapter [--no-gpu]

OUTPUT COLUMNS:
    ‖W‖₂                : Spectral norm (largest singular value)
                          • Must be < 1 for Neumann series convergence
                          • Target < 0.25 for optimal 1st-order approximation

    ‖W‖_F               : Frobenius norm (‖singular values‖₂)
                          • Cheaper to monitor during training
                          • Target < 0.25·√r

    ‖W‖_F/√r            : Normalized Frobenius norm for rank-independent comparison

    ‖W‖₂/‖W‖_F          : Ratio showing singular value distribution
                          • = 1/√r for uniform distribution
                          • = 1 for rank-1 matrix (all energy in one direction)

    erank(W)            : Effective rank = (‖W‖_*)²/(‖W‖_F)² ∈ [1, r]
                          • High ≈ balanced singular values
                          • Low ≈ rank collapse

    κ(W)                : Condition number = σ_max/σ_min
                          • Good: < 100, Poor: > 1000

    Target              : ✓ if ‖W‖₂ < 0.25 (optimal approximation)

    Stable              : ✓ if ‖W‖₂ < 1 (convergence guaranteed)

SUMMARY STATISTICS:
    Shows min/max/mean across all layers, plus convergence status.
"""

from pathlib import Path
from typing import Dict
import argparse
import json
import math
import re
import safetensors.torch
import torch

def load_control_adapter_weights(adapter_path: Path) -> Dict[str, torch.Tensor]:
    """Load Control Adapter weights from safetensors or .bin files."""
    # Try adapter_model.safetensors first
    st_path = adapter_path / 'adapter_model.safetensors'
    if st_path.exists():
        return safetensors.torch.load_file(st_path, device='cpu')

    # Try adapter_model.bin
    bin_path = adapter_path / 'adapter_model.bin'
    if bin_path.exists():
        return torch.load(bin_path, map_location='cpu', weights_only=True)

    raise FileNotFoundError(f"No adapter_model.safetensors or adapter_model.bin found in {adapter_path}")

def parse_control_adapter_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[int, Dict[str, torch.Tensor]]:
    """Parse control adapter state dict into layer -> {A, B} mapping."""
    control_adapters = {}

    for key, tensor in state_dict.items():
        match = re.search(r'layers\.(\d+)\.control_(A|B)', key)
        if match:
            layer_idx = int(match.group(1))
            param_type = match.group(2)

            if layer_idx not in control_adapters:
                control_adapters[layer_idx] = {}
            control_adapters[layer_idx][param_type] = tensor

    return control_adapters

def load_adapter_config(adapter_path: Path) -> Dict:
    """Load adapter configuration."""
    config_path = adapter_path / 'adapter_config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"adapter_config.json not found in {adapter_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    if 'lora_alpha' not in config or 'r' not in config:
        raise ValueError("adapter_config.json must contain 'lora_alpha' and 'r'")

    return config

def analyze_layer_norms(A: torch.Tensor, B: torch.Tensor, lora_scale: float, rank: int, device: str) -> Dict[str, float]:
    """Analyze norms for a single Control Adapter layer.

    Args:
        A: control_A.weight tensor [adapter_rank, hidden_size]
        B: control_B.weight tensor [hidden_size, adapter_rank]
        lora_scale: lora_alpha / rank scaling factor
        rank: adapter rank
        device: device for computation ('cpu' or 'cuda')

    Returns:
        Dictionary of statistics including norms, effective rank, condition number, etc.
    """
    A_gpu = A.to(device=device, dtype=torch.float32)
    B_gpu = B.to(device=device, dtype=torch.float32)

    # Compute composite matrix W = lora_scale * (B @ A)
    # B is [hidden_size, adapter_rank], A is [adapter_rank, hidden_size]
    # Result is [hidden_size, hidden_size]
    W = lora_scale * (B_gpu @ A_gpu)

    # Use SVD for accurate norm calculations
    # full_matrices=False gives min(M, N) singular values
    _, S, _ = torch.linalg.svd(W, full_matrices=False)

    # Truncate to rank (in case of numerical artifacts)
    k = min(rank, S.numel())
    assert k > 0, f"No singular values computed (rank={rank}, S.numel()={S.numel()})"

    S_truncated = S[:k]

    # Compute norms from singular values
    spectral_norm = S_truncated[0].item()  # ‖W‖₂ = σ_max
    min_singular_value = S_truncated[-1].item()  # σ_min (within rank)
    nuclear_norm = torch.sum(S_truncated).item()  # ‖W‖_* = Σσ_i
    frobenius_norm = torch.norm(S_truncated).item()  # ‖W‖_F = ‖σ‖₂

    # Effective rank: erank = (‖W‖_*)² / (‖W‖_F)²
    effective_rank = (nuclear_norm ** 2) / (frobenius_norm ** 2) if frobenius_norm > 1e-10 else 0.0

    # Condition number: κ = σ_max / σ_min
    condition_number = spectral_norm / min_singular_value if min_singular_value > 1e-10 else float('inf')

    sqrt_rank = math.sqrt(rank)

    # Guard division by frobenius_norm
    spectral_over_frobenius = spectral_norm / frobenius_norm if frobenius_norm > 1e-10 else 0.0

    return {
        'spectral_norm': spectral_norm,
        'frobenius_norm': frobenius_norm,
        'nuclear_norm': nuclear_norm,
        'effective_rank': effective_rank,
        'condition_number': condition_number,
        'sqrt_rank': sqrt_rank,
        'frobenius_over_sqrt_rank': frobenius_norm / sqrt_rank,
        'spectral_over_frobenius': spectral_over_frobenius,
        'converges': spectral_norm < 1.0,
        'meets_guideline': frobenius_norm < 0.25 * sqrt_rank,
        'optimal_spectral': spectral_norm < 0.25
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Control Adapter norm constraints")
    parser.add_argument("--adapter", required=True, type=str, help="Path to the Control Adapter directory")
    parser.add_argument("--no-gpu", action="store_true", help="Use CPU for SVD computation")
    args = parser.parse_args()

    # Device selection
    device = "cpu" if args.no_gpu or not torch.cuda.is_available() else "cuda"

    adapter_path = Path(args.adapter)

    # Load configuration and weights
    config = load_adapter_config(adapter_path)
    state_dict = load_control_adapter_weights(adapter_path)

    lora_alpha = config['lora_alpha']
    lora_rank = config['r']
    lora_scale = lora_alpha / lora_rank

    print("Control Adapter Analysis")
    print("========================")
    print(f"Path       : '{adapter_path}'")
    print(f"Device     : '{device}'")
    print(f"LoRA Alpha : {lora_alpha}")
    print(f"LoRA Rank  : {lora_rank}")
    print(f"LoRA Scale : {lora_scale:.4f}")
    print()
    print("Convergence Requirements:")
    print(f"  • ‖W‖₂ < 1.00   (Neumann series converges)")
    print(f"  • ‖W‖₂ < 0.25   (optimal 1st-order approximation: error ≤ 1-2%)")
    print(f"  • ‖W‖_F < {0.25 * math.sqrt(lora_rank):.3f}  (monitoring target: 0.25·√r)")
    print()

    # Parse layers
    layer_data = parse_control_adapter_keys(state_dict)

    if not layer_data:
        print("No control adapter layers found!")
        raise SystemExit(1)

    # Analyze each layer
    results = []
    print(f"{'Layer':<6} {'‖W‖₂':<8} {'‖W‖_F':<8} {'‖W‖_F/√r':<10} {'‖W‖₂/‖W‖_F':<12} "
          f"{'erank(W)':<10} {'κ(W)':<8} {'Target':<8} {'Stable':<8}")
    print("-" * 84)

    for layer_idx in sorted(layer_data.keys()):
        if 'A' not in layer_data[layer_idx] or 'B' not in layer_data[layer_idx]:
            continue

        A = layer_data[layer_idx]['A']
        B = layer_data[layer_idx]['B']

        stats = analyze_layer_norms(A, B, lora_scale, lora_rank, device)
        results.append(stats)

        print(f"{layer_idx:<6} "
              f"{stats['spectral_norm']:<8.3f} "
              f"{stats['frobenius_norm']:<8.3f} "
              f"{stats['frobenius_over_sqrt_rank']:<10.3f} "
              f"{stats['spectral_over_frobenius']:<12.3f} "
              f"{stats['effective_rank']:<10.1f} "
              f"{stats['condition_number']:<8.1f} "
              f"{'✓' if stats['optimal_spectral'] else '✗':<8} "
              f"{'✓' if stats['converges'] else '✗':<8}")

    # Summary statistics
    if results:
        spectral_norms = [r['spectral_norm'] for r in results]
        frobenius_norms = [r['frobenius_norm'] for r in results]
        frobenius_ratios = [r['frobenius_over_sqrt_rank'] for r in results]
        effective_ranks = [r['effective_rank'] for r in results]
        condition_numbers = [r['condition_number'] for r in results]

        converging_layers = sum(1 for r in results if r['converges'])
        optimal_layers = sum(1 for r in results if r['optimal_spectral'])

        print()
        layer_count_digits = len(str(len(results)))
        print(f"Summary statistics for {len(results)} layers (min / max / mean):")
        print("=" * (58 + layer_count_digits))
        print(f"Spectral norm (‖W‖₂)          : {min(spectral_norms):.3f} / {max(spectral_norms):.3f} / {sum(spectral_norms)/len(spectral_norms):.3f}")
        print(f"Frobenius norm (‖W‖_F)        : {min(frobenius_norms):.3f} / {max(frobenius_norms):.3f} / {sum(frobenius_norms)/len(frobenius_norms):.3f}")
        print(f"Normalized Frobenius (‖W‖_F/√r) : {min(frobenius_ratios):.3f} / {max(frobenius_ratios):.3f} / {sum(frobenius_ratios)/len(frobenius_ratios):.3f}")
        print(f"Effective rank (erank(W))     : {min(effective_ranks):.1f} / {max(effective_ranks):.1f} / {sum(effective_ranks)/len(effective_ranks):.1f}")
        print(f"Condition number (κ(W))       : {min(condition_numbers):.1f} / {max(condition_numbers):.1f} / {sum(condition_numbers)/len(condition_numbers):.1f}")
        print()
        print(f"Stable layers (‖W‖₂ < 1):      {converging_layers}/{len(results)} ({100*converging_layers/len(results):.1f}%)")
        print(f"Optimal layers (‖W‖₂ < 0.25):  {optimal_layers}/{len(results)} ({100*optimal_layers/len(results):.1f}%)")

        if max(spectral_norms) >= 1.0:
            print()
            print("⚠️  WARNING: Some layers have ‖W‖₂ ≥ 1, which may cause Neumann series divergence!")

        if min(spectral_norms) >= 0.25:
            print("⚠️  WARNING: All layers exceed the optimal ‖W‖₂ < 0.25 target!")
        elif max(spectral_norms) >= 0.25:
            print("⚠️  WARNING: Some layers exceed the optimal ‖W‖₂ < 0.25 target!")