#!/usr/bin/env python3
"""
Script to analyze Control Adapter matrix norms and statistical properties.

OVERVIEW:
    Analyzes Control Adapter matrices W = scale * Q @ diag(λ) @ Q^T where:
    • Q ∈ ℝ^{H×r} should have orthonormal columns (enforced by training regularizer)
    • λ ∈ ℝ^r contains per-direction eigenvalues
    • scale = lora_alpha / r

    Computes true matrix norms via SVD and compares to λ-based approximations
    (which are exact only when Q is perfectly orthogonal).

USAGE:
    python analyze_control_adapters.py control_adapter_path [--no-gpu]

OUTPUT COLUMNS:
    ‖Q^TQ-I_r‖_F²       : Squared orthogonality error of Q
                          • 0 = perfect, max = r(r-1) = 240 for rank-16
                          • Good: < 1.0, Excellent: < 0.5, Poor: > 2.0

    ‖W‖_F, ‖W‖_2, ‖W‖_* : True matrix norms from SVD (truncated to rank r)
                          • ‖W‖_2 should be << 1 for stable Neumann approximation

    erank(W)            : Effective rank = (‖W‖_*)²/(‖W‖_F)² ∈ [1, r]
                          • Measures singular value distribution flatness
                          • Good: Close to r (e.g., 15+ out of 16), Poor: << r (rank collapse)

    κ(W)                : Condition number = σ_max/σ_min
                          • Good: < 100, Poor: > 1000, ∞ = rank deficient

    ε_inv               : Neumann inverse error bound = ‖W‖_2²/(1-‖W‖_2) for ‖W‖_2 < 1
                          • Error in (I+W)^(-1) ≈ I-W approximation used for class -1 samples
                          • Good: < 1-5%, ∞ when ‖W‖_2 ≥ 1 (approximation invalid)
                          • Ideally maintain ‖W‖_2 < 0.2-0.3 for good inverse approximation


    ‖Ŵ‖_2, ‖Ŵ‖_*        : λ-based approximations assuming perfect orthogonality:
                          • ‖Ŵ‖_2 = scale * max(|λ|), ‖Ŵ‖_* = scale * sum(|λ|)

    ε_‖Ŵ‖_2, ε_‖Ŵ‖_*    : Signed approximation errors = (approx - true) / true
                          • Positive = overestimation (expected with non-orthogonal Q)
                          • Good: < 5%, Poor: > 20%

SUMMARY STATISTICS:
    Shows mean [min, max] across layers to identify outliers and overall health.

DIAGNOSTIC GUIDE:
    High orthogonality error   → Increase control_adapter_gamma (≤ 0.5)
    Low effective rank         → Check for rank collapse, review training dynamics
    High condition number      → Reduce learning rate or rank
    High Neumann error         → ‖W‖_2 too large, improve orthogonality or reduce scale
    Large approximation errors → Use SVD values instead of λ-based estimates
"""

from pathlib import Path
import argparse
import torch

from control_adapter_utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Control Adapter norm constraints")
    parser.add_argument("control_adapter_path", type=str, help="Path to the Control Adapter directory")
    parser.add_argument("--no-gpu", action="store_true", help="Use CPU for SVD computation")
    args = parser.parse_args()

    # Device selection logic from control_adapter_to_lora.py
    device = "cpu" if args.no_gpu or not torch.cuda.is_available() else "cuda"

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
    print(f"Path   : '{adapter_path}'")
    print(f"Alpha  : {lora_alpha}")
    print(f"Rank   : {lora_rank}")
    print(f"Scale  : {lora_scale:.4f}".rstrip('0').rstrip('.'))
    print(f"Device : '{device}'")
    print()

    # Parse layers using utils function
    layer_data = parse_control_adapter_keys(state_dict)

    if not layer_data:
        print("No control adapter layers found!")
        exit(1)

    # Analyze each layer
    results = []
    print(f"{'Layer':<6} {'‖QᵀQ-Iᵣ‖_F²':<12} {'‖W‖_F':<8} {'‖W‖_2':<8} {'‖W‖_*':<8} "
          f"{'erank(W)':<10} {'κ(W)':<8} {'ε_inv':<8} "
          f"{'‖Ŵ‖_2':<8} {'‖Ŵ‖_*':<8} {'ε_‖Ŵ‖_2':<10} {'ε_‖Ŵ‖_*':<10}")
    print("-" * 112)

    for layer_idx in sorted(layer_data.keys()):
        # Check that both Q and lambda exist for this layer
        if 'lambda' not in layer_data[layer_idx] or 'Q' not in layer_data[layer_idx]:
            continue

        lambda_vec = layer_data[layer_idx]['lambda'].to(torch.float32)
        Q = layer_data[layer_idx]['Q'].to(torch.float32)

        # Compute the full matrix W = scale * Q @ diag(λ) @ Q^T
        W = apply_control_adapter_transform(Q, lambda_vec, lora_scale, device)

        # Run SVD to get true singular values
        S = torch.linalg.svdvals(W)

        # Move singular values back to CPU for statistics
        S = S.cpu()

        # Truncate to the actual rank of the control adapter
        S_truncated = S[:lora_rank]

        # True norms from SVD (using truncated singular values)
        spectral_norm = S_truncated[0].item()  # ||W||_2 = max singular value
        min_singular_value = S_truncated[-1].item()  # minimum singular value within rank
        nuclear_norm = torch.sum(S_truncated).item()  # ||W||_* = sum of singular values
        frobenius_norm = torch.norm(S_truncated).item()  # ||W||_F = ||singular values||_2

        # Effective rank
        effective_rank = (nuclear_norm ** 2) / (frobenius_norm ** 2) if frobenius_norm > 1e-10 else 0.0

        # Condition number
        condition_number = spectral_norm / min_singular_value if min_singular_value > 1e-10 else float('inf')

        # Neumann inverse approximation error bound:
        #   ε = ‖W‖₂² / (1 − ‖W‖₂), valid for ‖W‖₂ < 1; otherwise ∞
        if spectral_norm < 1.0 - 1e-12:
            neumann_error = (spectral_norm ** 2) / (1.0 - spectral_norm)
        else:
            neumann_error = float('inf')

        # Approximated norms (assuming perfect orthogonality)
        spectral_norm_approx = lora_scale * torch.max(torch.abs(lambda_vec)).item()
        nuclear_norm_approx = lora_scale * torch.sum(torch.abs(lambda_vec)).item()

        # Signed ratio errors (for % formatting) - shows direction of bias
        spectral_error_ratio = (spectral_norm_approx - spectral_norm) / abs(spectral_norm) if abs(spectral_norm) > 1e-10 else 0.0
        nuclear_error_ratio = (nuclear_norm_approx - nuclear_norm) / abs(nuclear_norm) if abs(nuclear_norm) > 1e-10 else 0.0

        # Orthogonality measure: ||Q^T @ Q - I_r||_F^2
        QTQ = Q.t() @ Q
        I_r = torch.eye(Q.size(1), dtype=Q.dtype)
        orthogonality_error = (torch.norm(QTQ - I_r, p='fro') ** 2).item()

        stats = {
            'spectral_norm': spectral_norm,
            'nuclear_norm': nuclear_norm,
            'frobenius_norm': frobenius_norm,
            'effective_rank': effective_rank,
            'condition_number': condition_number,
            'neumann_error': neumann_error,
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
              f"{stats['neumann_error']:<8.1%} "
              f"{stats['spectral_norm_approx']:<8.3f} "
              f"{stats['nuclear_norm_approx']:<8.3f} "
              f"{stats['spectral_error_ratio']:<10.1%} "
              f"{stats['nuclear_error_ratio']:<10.1%}")

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
        neumann_errors = [r['neumann_error'] for r in results]
        print()

        layer_count_digits = len(str(len(results)))
        print(f"Summary statistics for {len(results)} layers (mean [min, max]):")
        print("=" * (49 + layer_count_digits))
        print(f"Orthogonality (‖QᵀQ-Iᵣ‖_F²)  : {sum(orthogonality_errors)/len(orthogonality_errors):.3f} "
              f"[{min(orthogonality_errors):.3f}, {max(orthogonality_errors):.3f}]")
        print(f"True Frobenius norm (‖W‖_F)  : {sum(frobenius_norms)/len(frobenius_norms):.3f} "
              f"[{min(frobenius_norms):.3f}, {max(frobenius_norms):.3f}]")
        print(f"True spectral norm (‖W‖_2)   : {sum(spectral_norms)/len(spectral_norms):.3f} "
              f"[{min(spectral_norms):.3f}, {max(spectral_norms):.3f}]")
        print(f"True nuclear norm (‖W‖_*)    : {sum(nuclear_norms)/len(nuclear_norms):.3f} "
              f"[{min(nuclear_norms):.3f}, {max(nuclear_norms):.3f}]")
        print(f"Effective rank (erank(W))    : {sum(effective_ranks)/len(effective_ranks):.1f} "
              f"[{min(effective_ranks):.1f}, {max(effective_ranks):.1f}]")
        print(f"Condition number (κ(W))      : {sum(condition_numbers)/len(condition_numbers):.1f} "
              f"[{min(condition_numbers):.1f}, {max(condition_numbers):.1f}]")
        print(f"Neumann inv error (ε_inv)    : {sum(neumann_errors)/len(neumann_errors):.1%} "
              f"[{min(neumann_errors):.1%}, {max(neumann_errors):.1%}]")
        print(f"Spectral norm error (ε_2)    : {sum(spectral_error_ratios)/len(spectral_error_ratios):.1%} "
              f"[{min(spectral_error_ratios):.1%}, {max(spectral_error_ratios):.1%}]")
        print(f"Nuclear norm error (ε_*)     : {sum(nuclear_error_ratios)/len(nuclear_error_ratios):.1%} "
              f"[{min(nuclear_error_ratios):.1%}, {max(nuclear_error_ratios):.1%}]")