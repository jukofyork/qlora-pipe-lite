#!/usr/bin/env python3
"""
Analyze Control Adapter matrix norms and statistical properties.

OVERVIEW:
    Analyzes Control Adapter matrices W = Q @ diag(λ) @ Q^T where:
    • Q ∈ ℝ^{H×r} should have semi-orthogonal columns (enforced by training regularizer)
    • λ ∈ ℝ^r contains per-direction eigenvalues

    Computes true matrix norms via SVD and compares to λ-based approximations
    (which are exact only when Q is perfectly orthogonal).

USAGE:
    python analyze_control_adapters.py --adapter /path/to/adapter [--no-gpu]

OUTPUT COLUMNS:
    ‖Q^TQ-I_r‖_F²       : Squared orthogonality error of Q
                          • 0 = perfect, max = r(r-1) = 240 for rank-16
                          • Good: < 1.0, Excellent: < 0.5, Poor: > 2.0

    ‖W‖_F, ‖W‖_2, ‖W‖_* : True matrix norms from SVD (truncated to rank r)

    erank(W)            : Effective rank = (‖W‖_*)²/(‖W‖_F)² ∈ [1, r]
                          • Measures singular value distribution flatness
                          • Good: Close to r (e.g., 15+ out of 16), Poor: << r (rank collapse)

    κ(W)                : Condition number = σ_max/σ_min
                          • Good: < 100, Poor: > 1000, ∞ = rank deficient

    ‖Ŵ‖_2, ‖Ŵ‖_*        : λ-based approximations assuming perfect orthogonality:
                          • ‖Ŵ‖_2 = max(|λ|), ‖Ŵ‖_* = sum(|λ|)

    ε_‖Ŵ‖_2, ε_‖Ŵ‖_*    : Signed approximation errors = (approx - true) / true
                          • Positive = overestimation (expected with non-orthogonal Q)
                          • Good: < 5%, Poor: > 20%

SUMMARY STATISTICS:
    Shows mean [min, max] across all layers.
"""

from pathlib import Path
import argparse
import torch

from training.control_adapters import (
    apply_control_adapter_transform,
    load_control_adapter_weights,
    parse_control_adapter_keys,
)
from utils.utils import format_percentage

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Control Adapter norm constraints")
    parser.add_argument("--adapter", required=True, type=str, help="Path to the Control Adapter directory")
    parser.add_argument("--no-gpu", action="store_true", help="Use CPU for SVD computation")
    args = parser.parse_args()

    # Device selection logic
    device = "cpu" if args.no_gpu or not torch.cuda.is_available() else "cuda"

    adapter_path = Path(args.adapter)

    # Load weights
    _, state_dict = load_control_adapter_weights(adapter_path)

    print("Control Adapter Analysis")
    print("========================")
    print(f"Path   : '{adapter_path}'")
    print(f"Device : '{device}'")
    print()

    # Parse layers
    layer_data = parse_control_adapter_keys(state_dict)

    if not layer_data:
        print("No control adapter layers found!")
        raise SystemExit(1)

    # Analyze each layer
    results = []
    print(f"{'Layer':<6} {'‖QᵀQ-Iᵣ‖_F²':<12} {'‖W‖_F':<8} {'‖W‖_2':<8} {'‖W‖_*':<8} "
          f"{'erank(W)':<10} {'κ(W)':<8} "
          f"{'‖Ŵ‖_2':<8} {'‖Ŵ‖_*':<8} {'ε_‖Ŵ‖_2':<10} {'ε_‖Ŵ‖_*':<10}")
    print("-" * 104)

    for layer_idx in sorted(layer_data.keys()):
        # Check that both Q and S exist for this layer
        if 'Q' not in layer_data[layer_idx] or 'S' not in layer_data[layer_idx]:
            continue

        Q = layer_data[layer_idx]['Q'].to(torch.float32)
        S = layer_data[layer_idx]['S'].to(torch.float32)

        rank = int(S.numel())
        assert Q.ndim == 2 and Q.shape[1] == rank, f"Shape mismatch: Q has {Q.shape[1]} cols, S has {rank} entries"

        with torch.no_grad():
            # Compute the full matrix W = Q @ diag(λ) @ Q^T, with λ = exp(S) - 1
            W = apply_control_adapter_transform(Q, S, device=device, inverse=False)

            # Run SVD to get true singular values
            svals = torch.linalg.svdvals(W).cpu()

            # Truncate to the actual rank of the control adapter
            svals = svals[:rank]

            # True norms from SVD (using truncated singular values)
            spectral_norm = svals[0].item()  # ||W||_2 = max singular value
            min_singular_value = svals[-1].item()  # minimum singular value within rank
            nuclear_norm = torch.sum(svals).item()  # ||W||_* = sum of singular values
            frobenius_norm = torch.norm(svals).item()  # ||W||_F = ||singular values||_2

            # Effective rank
            effective_rank = (nuclear_norm ** 2) / (frobenius_norm ** 2) if frobenius_norm > 1e-10 else 0.0

            # Condition number
            condition_number = spectral_norm / min_singular_value if min_singular_value > 1e-10 else float('inf')

            # Approximated norms (assuming perfect orthogonality): λ = exp(S) - 1
            lambda_vec = torch.expm1(S)
            spectral_norm_approx = torch.max(torch.abs(lambda_vec)).item()
            nuclear_norm_approx = torch.sum(torch.abs(lambda_vec)).item()

            # Signed ratio errors (for % formatting) - shows direction of bias
            spectral_error_ratio = (spectral_norm_approx - spectral_norm) / abs(spectral_norm) if abs(spectral_norm) > 1e-10 else 0.0
            nuclear_error_ratio = (nuclear_norm_approx - nuclear_norm) / abs(nuclear_norm) if abs(nuclear_norm) > 1e-10 else 0.0

            # Orthogonality measure: ||Q^T @ Q - I_r||_F^2
            QTQ = Q.t() @ Q
            I_r = torch.eye(Q.size(1), dtype=Q.dtype, device=Q.device)
            orthogonality_error = (torch.norm(QTQ - I_r, p='fro') ** 2).item()

        stats = {
            'spectral_norm': spectral_norm,
            'nuclear_norm': nuclear_norm,
            'frobenius_norm': frobenius_norm,
            'effective_rank': effective_rank,
            'condition_number': condition_number,
            'spectral_norm_approx': spectral_norm_approx,
            'nuclear_norm_approx': nuclear_norm_approx,
            'spectral_error_ratio': spectral_error_ratio,
            'nuclear_error_ratio': nuclear_error_ratio,
            'orthogonality_error': orthogonality_error
        }
        results.append(stats)

        print(f"{layer_idx:<6} "
              f"{stats['orthogonality_error']:<12.3f} "
              f"{stats['frobenius_norm']:<8.3f} "
              f"{stats['spectral_norm']:<8.3f} "
              f"{stats['nuclear_norm']:<8.3f} "
              f"{stats['effective_rank']:<10.1f} "
              f"{stats['condition_number']:<8.1f} "
              f"{stats['spectral_norm_approx']:<8.3f} "
              f"{stats['nuclear_norm_approx']:<8.3f} "
              f"{format_percentage(stats['spectral_error_ratio'], 3):<10} "
              f"{format_percentage(stats['nuclear_error_ratio'], 3):<10}")

    # Summary statistics
    if results:
        spectral_norms = [r['spectral_norm'] for r in results]
        nuclear_norms = [r['nuclear_norm'] for r in results]
        frobenius_norms = [r['frobenius_norm'] for r in results]
        effective_ranks = [r['effective_rank'] for r in results]
        condition_numbers = [r['condition_number'] for r in results]
        orthogonality_errors = [r['orthogonality_error'] for r in results]
        spectral_error_ratios = [r['spectral_error_ratio'] for r in results]
        nuclear_error_ratios = [r['nuclear_error_ratio'] for r in results]
        print()

        layer_count_digits = len(str(len(results)))
        print(f"Summary statistics for {len(results)} layers (mean [min, max]):")
        print("=" * (49 + layer_count_digits))
        print(f"Orthogonality (‖QᵀQ-Iᵣ‖_F²) : {sum(orthogonality_errors)/len(orthogonality_errors):.3f} "
              f"[{min(orthogonality_errors):.3f}, {max(orthogonality_errors):.3f}]")
        print(f"True Frobenius norm (‖W‖_F) : {sum(frobenius_norms)/len(frobenius_norms):.3f} "
              f"[{min(frobenius_norms):.3f}, {max(frobenius_norms):.3f}]")
        print(f"True spectral norm (‖W‖_2)  : {sum(spectral_norms)/len(spectral_norms):.3f} "
              f"[{min(spectral_norms):.3f}, {max(spectral_norms):.3f}]")
        print(f"True nuclear norm (‖W‖_*)   : {sum(nuclear_norms)/len(nuclear_norms):.3f} "
              f"[{min(nuclear_norms):.3f}, {max(nuclear_norms):.3f}]")
        print(f"Effective rank (erank(W))   : {sum(effective_ranks)/len(effective_ranks):.1f} "
              f"[{min(effective_ranks):.1f}, {max(effective_ranks):.1f}]")
        print(f"Condition number (κ(W))     : {sum(condition_numbers)/len(condition_numbers):.1f} "
              f"[{min(condition_numbers):.1f}, {max(condition_numbers):.1f}]")
        print(f"Spectral norm error (ε_2)   : {format_percentage(sum(spectral_error_ratios)/len(spectral_error_ratios))} "
              f"[{format_percentage(min(spectral_error_ratios))}, {format_percentage(max(spectral_error_ratios))}]")
        print(f"Nuclear norm error (ε_*)    : {format_percentage(sum(nuclear_error_ratios)/len(nuclear_error_ratios))} "
              f"[{format_percentage(min(nuclear_error_ratios))}, {format_percentage(max(nuclear_error_ratios))}]")