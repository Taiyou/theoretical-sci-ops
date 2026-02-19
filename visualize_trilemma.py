"""
Visualization Module for the Scientific Operations Trilemma
=============================================================

Generates publication-quality figures for the trilemma analysis:
1. 3D Trilemma Surface (Pareto frontier in S-Q-I space)
2. Ternary Diagram (time allocation trade-offs)
3. Phase Diagram (trilemma severity by domain and AI capability)
4. Epistemic Uncertainty Principle visualization
5. Goodhart Drift dynamics
6. Sprint/Reflect cycle optimization landscape
7. Counterexample boundary map
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, art3d
import mpl_toolkits.mplot3d as mplot3d
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Optional

from trilemma_proof import (
    TrilemmaParams, speed, quality, intelligibility,
    prove_trilemma, epistemic_uncertainty, error_amplification,
    understanding_fraction_vs_alpha, find_optimal_allocation,
    DOMAIN_PROFILES, analyze_domain
)

# Style configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = {
    'speed': '#E74C3C',
    'quality': '#2ECC71',
    'intelligibility': '#3498DB',
    'trilemma': '#9B59B6',
    'drift': '#E67E22',
    'rework': '#E74C3C',
    'sprint': '#F39C12',
    'reflect': '#1ABC9C',
}


# =============================================================================
# 1. 3D Trilemma Surface (Pareto Frontier)
# =============================================================================

def plot_trilemma_3d(params: TrilemmaParams = None, n_points: int = 80,
                     save_path: str = None) -> plt.Figure:
    """
    Visualize the Pareto frontier of S, Q, I in 3D space.

    Shows the feasible region (colored surface) and the impossible region
    (beyond the surface). Optimal allocations for different strategies are
    marked on the surface.
    """
    if params is None:
        params = TrilemmaParams(alpha=1.0)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    T = params.T
    S_vals, Q_vals, I_vals = [], [], []
    colors = []

    # Sample the feasible region
    for i in range(n_points):
        for j in range(n_points - i):
            te = max(T * (i + 0.5) / n_points, 0.01)
            tv = max(T * (j + 0.5) / n_points, 0.01)
            tu = T - te - tv
            if tu <= 0.01:
                continue

            S = speed(te, params)
            Q = quality(tv, params)
            I = intelligibility(tu, params)

            # Normalize S for visualization
            S_norm = S / speed(0.01, params)  # Normalize to [0, ~1]
            S_norm = min(S_norm, 1.0)

            S_vals.append(S_norm)
            Q_vals.append(Q)
            I_vals.append(I)

            # Color by F = S*Q*I (product metric)
            colors.append(S_norm * Q * I)

    S_vals = np.array(S_vals)
    Q_vals = np.array(Q_vals)
    I_vals = np.array(I_vals)
    colors = np.array(colors)

    # Scatter plot of feasible points
    sc = ax.scatter(S_vals, Q_vals, I_vals, c=colors, cmap='viridis',
                    alpha=0.4, s=8, edgecolors='none')

    # Mark the three extreme strategies
    strategies = {
        'S+Q\n(sacrifice: I)': (0.3 * T, 0.6 * T, 0.1 * T),
        'S+I\n(sacrifice: Q)': (0.3 * T, 0.1 * T, 0.6 * T),
        'Q+I\n(sacrifice: S)': (0.1 * T, 0.45 * T, 0.45 * T),
    }

    S_max = speed(0.01, params)
    for label, (te, tv, tu) in strategies.items():
        S_n = min(speed(te, params) / S_max, 1.0)
        Q_n = quality(tv, params)
        I_n = intelligibility(tu, params)
        ax.scatter([S_n], [Q_n], [I_n], s=200, marker='*', zorder=10,
                   edgecolors='black', linewidths=1.5)
        ax.text(S_n, Q_n, I_n + 0.05, label, fontsize=9, ha='center',
                fontweight='bold')

    # Mark optimal (balanced) point
    opt = find_optimal_allocation(params)
    S_opt_n = min(opt['S'] / S_max, 1.0)
    ax.scatter([S_opt_n], [opt['Q']], [opt['I']], s=300, marker='D',
               color='gold', edgecolors='black', linewidths=2, zorder=10)
    ax.text(S_opt_n, opt['Q'], opt['I'] + 0.07, 'Optimal\n(F=S·Q·I max)',
            fontsize=9, ha='center', fontweight='bold', color='darkgoldenrod')

    ax.set_xlabel('Speed (S)', fontsize=13, labelpad=10, color=COLORS['speed'])
    ax.set_ylabel('Quality (Q)', fontsize=13, labelpad=10, color=COLORS['quality'])
    ax.set_zlabel('Intelligibility (I)', fontsize=13, labelpad=10,
                  color=COLORS['intelligibility'])
    ax.set_title(f'SciOps Trilemma: Pareto Frontier (α={params.alpha})',
                 fontsize=15, fontweight='bold', pad=20)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.5, label='F = S·Q·I')

    ax.view_init(elev=25, azim=135)

    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# 2. Ternary Allocation Diagram
# =============================================================================

def _ternary_to_cartesian(a, b, c):
    """Convert ternary coordinates (a, b, c) to 2D cartesian."""
    total = a + b + c
    a, b, c = a / total, b / total, c / total
    x = 0.5 * (2 * b + c)
    y = (np.sqrt(3) / 2) * c
    return x, y


def plot_ternary_allocation(params: TrilemmaParams = None, n_grid: int = 100,
                            save_path: str = None) -> plt.Figure:
    """
    Ternary diagram showing time allocation trade-offs.

    Each point represents an allocation (τ_exec, τ_verify, τ_understand).
    Color represents the product metric F = S·Q·I.
    """
    if params is None:
        params = TrilemmaParams(alpha=1.0)

    fig, ax = plt.subplots(figsize=(10, 9))

    T = params.T
    x_vals, y_vals, f_vals = [], [], []

    for i in range(1, n_grid):
        for j in range(1, n_grid - i):
            k = n_grid - i - j
            if k <= 0:
                continue

            te = T * i / n_grid
            tv = T * j / n_grid
            tu = T * k / n_grid

            S = speed(te, params)
            Q = quality(tv, params)
            I = intelligibility(tu, params)
            F = S * Q * I

            # Normalize S for comparable magnitude
            S_max = speed(T * 1 / n_grid, params)
            F_norm = (S / S_max) * Q * I

            x, y = _ternary_to_cartesian(i, j, k)
            x_vals.append(x)
            y_vals.append(y)
            f_vals.append(F_norm)

    x_vals = np.array(x_vals)
    y_vals = np.array(y_vals)
    f_vals = np.array(f_vals)

    sc = ax.scatter(x_vals, y_vals, c=f_vals, cmap='magma', s=15,
                    edgecolors='none', alpha=0.8)

    # Draw triangle
    triangle = plt.Polygon(
        [_ternary_to_cartesian(1, 0, 0),
         _ternary_to_cartesian(0, 1, 0),
         _ternary_to_cartesian(0, 0, 1)],
        fill=False, edgecolor='black', linewidth=2
    )
    ax.add_patch(triangle)

    # Labels at vertices
    offset = 0.05
    ax.text(*_ternary_to_cartesian(1.15, -0.05, -0.05),
            'τ_exec\n(Speed)', ha='center', fontsize=12,
            fontweight='bold', color=COLORS['speed'])
    ax.text(*_ternary_to_cartesian(-0.05, 1.15, -0.05),
            'τ_verify\n(Quality)', ha='center', fontsize=12,
            fontweight='bold', color=COLORS['quality'])
    ax.text(*_ternary_to_cartesian(-0.05, -0.05, 1.15),
            'τ_understand\n(Intelligibility)', ha='center', fontsize=12,
            fontweight='bold', color=COLORS['intelligibility'])

    # Mark optimal point
    opt = find_optimal_allocation(params)
    x_opt, y_opt = _ternary_to_cartesian(
        opt['fractions']['exec'],
        opt['fractions']['verify'],
        opt['fractions']['understand']
    )
    ax.scatter([x_opt], [y_opt], s=200, marker='*', color='gold',
               edgecolors='black', linewidths=2, zorder=10)
    ax.annotate('Optimal', (x_opt, y_opt), fontsize=11, fontweight='bold',
                xytext=(x_opt + 0.08, y_opt + 0.05),
                arrowprops=dict(arrowstyle='->', color='black'))

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.05)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Time Allocation Ternary Diagram (α={params.alpha})',
                 fontsize=15, fontweight='bold')

    plt.colorbar(sc, ax=ax, label='Normalized F = S·Q·I', shrink=0.7)

    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# 3. Phase Diagram: Trilemma Severity
# =============================================================================

def plot_phase_diagram(save_path: str = None) -> plt.Figure:
    """
    Phase diagram showing where the trilemma holds vs. is relaxed,
    as a function of AI capability α and complexity-productivity ratio C/A.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: C_exponent vs A_exponent, colored by trilemma severity
    ax = axes[0]
    C_exps = np.linspace(1.0, 4.0, 30)
    A_exps = np.linspace(1.0, 4.0, 30)
    severity = np.zeros((len(C_exps), len(A_exps)))

    for i, c_exp in enumerate(C_exps):
        for j, a_exp in enumerate(A_exps):
            p = TrilemmaParams(alpha=1.0, C_exponent=c_exp, A_exponent=a_exp)
            alpha_range = np.linspace(0.1, 2.0, 30)
            proof = prove_trilemma(p, alpha_range)
            severity[i, j] = np.mean(proof['trilemma_holds'])

    im = ax.imshow(severity, origin='lower', cmap='RdYlGn_r',
                   extent=[A_exps[0], A_exps[-1], C_exps[0], C_exps[-1]],
                   aspect='auto', vmin=0, vmax=1)

    # Mark domain profiles
    for key, domain in DOMAIN_PROFILES.items():
        ax.scatter(domain.A_exponent, domain.C_exponent, s=120,
                   edgecolors='black', linewidths=2, zorder=10,
                   marker='o', color='white')
        ax.annotate(domain.name.split('(')[0].strip(),
                    (domain.A_exponent, domain.C_exponent),
                    fontsize=7, ha='center', va='bottom',
                    xytext=(0, 8), textcoords='offset points',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                             alpha=0.8, edgecolor='gray'))

    # Diagonal line: C_exp = A_exp (trilemma boundary)
    ax.plot([1, 4], [1, 4], '--', color='black', linewidth=1.5, alpha=0.7)
    ax.text(3.5, 3.2, 'C=A\n(boundary)', fontsize=8, rotation=45,
            ha='center', color='black', alpha=0.7)

    ax.set_xlabel('A exponent (Productivity growth rate)', fontsize=12)
    ax.set_ylabel('C exponent (Complexity growth rate)', fontsize=12)
    ax.set_title('Trilemma Severity: Phase Diagram', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Fraction of α range where trilemma holds')

    # Right: α vs trilemma severity for each domain
    ax2 = axes[1]
    alpha_range = np.linspace(0.1, 2.0, 50)

    for key, domain in DOMAIN_PROFILES.items():
        p = TrilemmaParams(A_exponent=domain.A_exponent,
                           C_exponent=domain.C_exponent)
        proof = prove_trilemma(p, alpha_range)

        # Rolling average of trilemma condition
        window = 5
        condition_smooth = np.convolve(
            proof['trilemma_condition'],
            np.ones(window) / window, mode='valid'
        )
        alpha_smooth = alpha_range[:len(condition_smooth)]

        ax2.plot(alpha_smooth, condition_smooth, linewidth=2, label=domain.name)

    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.text(0.15, 1.05, 'Trilemma threshold', fontsize=9, color='black', alpha=0.7)

    ax2.set_xlabel('AI Capability (α)', fontsize=12)
    ax2.set_ylabel('Trilemma Condition Ratio\n(dC/dα) / [(I·Q/S)·(dA/dα)]', fontsize=11)
    ax2.set_title('Trilemma Condition by Domain', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper left')
    ax2.set_ylim(0, 5)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# 4. Epistemic Uncertainty Principle
# =============================================================================

def plot_uncertainty_principle(params: TrilemmaParams = None,
                               save_path: str = None) -> plt.Figure:
    """
    Visualize the epistemic uncertainty principle: ΔS · ΔI ≥ κ(α).

    Shows how κ(α) grows with AI capability, analogous to but distinct from
    Heisenberg's uncertainty principle.
    """
    if params is None:
        params = TrilemmaParams()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    alpha_range = np.linspace(0.01, 2.5, 100)

    eup = epistemic_uncertainty(params, alpha_range)

    # Panel 1: κ(α) growth
    ax = axes[0]
    ax.plot(alpha_range, eup['kappa'], linewidth=2.5, color=COLORS['trilemma'])
    ax.fill_between(alpha_range, 0, eup['kappa'], alpha=0.15, color=COLORS['trilemma'])
    ax.set_xlabel('AI Capability (α)', fontsize=12)
    ax.set_ylabel('κ(α) = C(α) / [A(α) · μ]', fontsize=12)
    ax.set_title('Epistemic Uncertainty Bound κ(α)', fontsize=14, fontweight='bold')

    # Add Heisenberg analogy annotation
    ax.annotate('cf. Heisenberg: ΔxΔp ≥ ℏ/2\n(ℏ is constant)',
                xy=(0.5, 0.85), xycoords='axes fraction',
                fontsize=9, ha='center', style='italic',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    ax.annotate('κ(α) is NOT constant:\nit grows with AI capability',
                xy=(0.5, 0.7), xycoords='axes fraction',
                fontsize=9, ha='center', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Panel 2: S vs I trade-off curves at different α
    ax2 = axes[1]
    T = params.T
    for alpha_val in [0.1, 0.5, 1.0, 1.5, 2.0]:
        p = TrilemmaParams(
            alpha=alpha_val, T=T, lam=params.lam, mu=params.mu,
            A_scale=params.A_scale, A_exponent=params.A_exponent,
            C_scale=params.C_scale, C_exponent=params.C_exponent,
        )
        S_vals, I_vals = [], []
        # Fix Q at a moderate level, vary S-I trade-off
        tau_verify = 0.3 * T
        for frac in np.linspace(0.05, 0.95, 50):
            te = (T - tau_verify) * frac
            tu = (T - tau_verify) * (1 - frac)
            S_max = speed(0.01, p)
            S_vals.append(min(speed(te, p) / S_max, 1.0))
            I_vals.append(intelligibility(tu, p))

        ax2.plot(S_vals, I_vals, linewidth=2, label=f'α={alpha_val}')

    ax2.set_xlabel('Normalized Speed (S)', fontsize=12, color=COLORS['speed'])
    ax2.set_ylabel('Intelligibility (I)', fontsize=12, color=COLORS['intelligibility'])
    ax2.set_title('S-I Trade-off at Different α\n(Q fixed at moderate level)',
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)

    # Add arrow showing constraint tightening
    ax2.annotate('Constraint\ntightens', xy=(0.6, 0.3), xytext=(0.8, 0.5),
                 fontsize=10, fontweight='bold', color=COLORS['trilemma'],
                 arrowprops=dict(arrowstyle='->', color=COLORS['trilemma'],
                                linewidth=2))

    # Panel 3: Comparison with Heisenberg structure
    ax3 = axes[2]
    # Schematic: two curves showing the structural analogy
    x = np.linspace(0.1, 5, 100)

    # Heisenberg-like: constant bound
    heisenberg_bound = 0.5 / x
    ax3.plot(x, heisenberg_bound, '--', color='gray', linewidth=2,
             label='Heisenberg: ΔxΔp ≥ ℏ/2\n(constant bound)')
    ax3.fill_between(x, 0, heisenberg_bound, alpha=0.1, color='gray')

    # Epistemic: growing bound (at α=2)
    epistemic_bound = 1.5 / x  # Larger bound
    ax3.plot(x, epistemic_bound, '-', color=COLORS['trilemma'], linewidth=2.5,
             label='Epistemic: ΔS·ΔI ≥ κ(α)\n(growing bound)')
    ax3.fill_between(x, 0, epistemic_bound, alpha=0.1, color=COLORS['trilemma'])

    ax3.set_xlabel('ΔS (or Δx)', fontsize=12)
    ax3.set_ylabel('ΔI (or Δp)', fontsize=12)
    ax3.set_title('Structural Analogy:\nHeisenberg vs. Epistemic UP',
                  fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.set_xlim(0, 5)
    ax3.set_ylim(0, 3)

    # Add key difference annotation
    ax3.text(3.5, 2.2, 'Key difference:\nκ(α) grows with α\n→ constraint worsens\n   as AI improves',
             fontsize=9, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# 5. Understanding Fraction vs α (Prediction 3)
# =============================================================================

def plot_prediction3(save_path: str = None) -> plt.Figure:
    """
    Visualize Prediction 3: Optimal τ_understand/T increases with α.

    "The better AI becomes, the MORE time humans should spend understanding."
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    alpha_range = np.linspace(0.01, 2.5, 80)

    # Left: Time allocation fractions
    ax = axes[0]
    params = TrilemmaParams()
    fracs = understanding_fraction_vs_alpha(params, alpha_range)

    ax.stackplot(alpha_range,
                 fracs['f_exec'], fracs['f_verify'], fracs['f_understand'],
                 labels=['τ_exec / T (Execution)', 'τ_verify / T (Verification)',
                         'τ_understand / T (Understanding)'],
                 colors=[COLORS['speed'], COLORS['quality'], COLORS['intelligibility']],
                 alpha=0.7)

    ax.set_xlabel('AI Capability (α)', fontsize=12)
    ax.set_ylabel('Optimal Time Fraction', fontsize=12)
    ax.set_title('Optimal Time Allocation vs. AI Capability',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='center right', fontsize=9)
    ax.set_ylim(0, 1)

    # Highlight the counterintuitive result
    ax.annotate('Counter-intuitive:\nunderstanding fraction\nINCREASES with α',
                xy=(1.8, fracs['f_understand'][int(1.8 / 2.5 * len(alpha_range))]),
                xytext=(0.5, 0.8), textcoords='axes fraction',
                fontsize=10, fontweight='bold', color=COLORS['intelligibility'],
                arrowprops=dict(arrowstyle='->', color=COLORS['intelligibility'],
                               linewidth=2))

    # Right: Understanding fraction alone, multiple C/A ratios
    ax2 = axes[1]
    for C_exp, label in [(1.5, 'C/A low (formal science)'),
                          (2.5, 'C/A moderate (data science)'),
                          (3.5, 'C/A high (experimental)')]:
        p = TrilemmaParams(C_exponent=C_exp)
        fracs = understanding_fraction_vs_alpha(p, alpha_range)
        ax2.plot(alpha_range, fracs['f_understand'], linewidth=2.5, label=label)

    ax2.set_xlabel('AI Capability (α)', fontsize=12)
    ax2.set_ylabel('Optimal Understanding Fraction (τ_understand / T)', fontsize=12)
    ax2.set_title('Prediction 3: Understanding Time Grows with α',
                  fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)

    ax2.axhline(y=1/3, color='gray', linestyle=':', alpha=0.5)
    ax2.text(0.1, 1/3 + 0.02, 'Equal allocation (1/3)', fontsize=8, color='gray')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# 6. Goodhart Drift and Error Amplification
# =============================================================================

def plot_error_amplification(params: TrilemmaParams = None,
                              save_path: str = None) -> plt.Figure:
    """
    Visualize the superlinear amplification of management errors:
    L(α) = S(α) · δ(α) · Δt(α) ∝ α^(superlinear)
    """
    if params is None:
        params = TrilemmaParams()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    alpha_range = np.linspace(0.01, 2.5, 100)

    amp = error_amplification(params, alpha_range)

    # Panel 1: Individual components
    ax = axes[0]
    ax.plot(alpha_range, amp['S'] / amp['S'][0], linewidth=2,
            color=COLORS['speed'], label='S(α) / S(0)')
    ax.plot(alpha_range, amp['delta'] / amp['delta'][0], linewidth=2,
            color=COLORS['drift'], label='δ(α) / δ(0)')
    ax.plot(alpha_range, amp['dt'] / amp['dt'][0], linewidth=2,
            color='gray', label='Δt(α) / Δt(0)')
    ax.set_xlabel('AI Capability (α)', fontsize=12)
    ax.set_ylabel('Normalized Value', fontsize=12)
    ax.set_title('Error Components', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)

    # Panel 2: Total loss (superlinear growth)
    ax2 = axes[1]
    L_norm = amp['L'] / amp['L'][0]
    ax2.plot(alpha_range, L_norm, linewidth=3, color=COLORS['rework'],
             label='L(α) = S·δ·Δt')

    # Linear reference
    linear_ref = 1 + (L_norm[-1] - 1) * (alpha_range - alpha_range[0]) / (alpha_range[-1] - alpha_range[0])
    ax2.plot(alpha_range, linear_ref, '--', color='gray', linewidth=1.5,
             label='Linear reference')

    ax2.fill_between(alpha_range, linear_ref, L_norm,
                     where=L_norm > linear_ref, alpha=0.2, color=COLORS['rework'])

    ax2.set_xlabel('AI Capability (α)', fontsize=12)
    ax2.set_ylabel('Normalized Loss L(α) / L(0)', fontsize=12)
    ax2.set_title('Error Amplification:\nSuperlinear Growth', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.annotate('Superlinear gap:\n"faster systems have\nlarger rework costs"',
                 xy=(1.5, L_norm[int(1.5 / 2.5 * len(alpha_range))]),
                 xytext=(0.3, 0.75), textcoords='axes fraction',
                 fontsize=10, fontweight='bold', color=COLORS['rework'],
                 arrowprops=dict(arrowstyle='->', color=COLORS['rework'], linewidth=2))

    # Panel 3: Implication — net benefit of AI capability
    ax3 = axes[2]
    # Gross benefit: productivity gain
    gross_benefit = amp['S'] / amp['S'][0]
    # Net benefit: gross - rework loss
    net_benefit = gross_benefit - (L_norm - 1) * 0.3  # Scaled rework impact

    ax3.plot(alpha_range, gross_benefit, linewidth=2, color=COLORS['quality'],
             label='Gross benefit (productivity)')
    ax3.plot(alpha_range, L_norm * 0.3, linewidth=2, color=COLORS['rework'],
             label='Rework cost')
    ax3.plot(alpha_range, net_benefit, linewidth=3, color=COLORS['trilemma'],
             label='Net benefit')

    # Find the peak
    peak_idx = np.argmax(net_benefit)
    ax3.axvline(x=alpha_range[peak_idx], color=COLORS['trilemma'],
                linestyle=':', alpha=0.7)
    ax3.scatter([alpha_range[peak_idx]], [net_benefit[peak_idx]], s=150,
                color=COLORS['trilemma'], zorder=10, edgecolors='black')
    ax3.annotate(f'Optimal α ≈ {alpha_range[peak_idx]:.1f}',
                 xy=(alpha_range[peak_idx], net_benefit[peak_idx]),
                 xytext=(alpha_range[peak_idx] + 0.3, net_benefit[peak_idx] + 0.5),
                 fontsize=10, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', linewidth=1.5))

    ax3.set_xlabel('AI Capability (α)', fontsize=12)
    ax3.set_ylabel('Benefit / Cost (normalized)', fontsize=12)
    ax3.set_title('Net Benefit of AI Acceleration\n(after rework costs)',
                  fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# 7. Trilemma Triangle Summary (Publication Figure)
# =============================================================================

def plot_trilemma_triangle(save_path: str = None) -> plt.Figure:
    """
    The canonical trilemma visualization: an equilateral triangle with
    S, Q, I at vertices, showing the three positioning choices and
    the impossibility region.
    """
    fig, ax = plt.subplots(figsize=(10, 9))

    # Triangle vertices
    vertices = np.array([
        [0.5, np.sqrt(3) / 2],   # Top: Intelligibility
        [0.0, 0.0],               # Bottom-left: Speed
        [1.0, 0.0],               # Bottom-right: Quality
    ])

    # Draw triangle
    triangle = plt.Polygon(vertices, fill=False, edgecolor='black',
                            linewidth=3)
    ax.add_patch(triangle)

    # Fill with gradient
    from matplotlib.colors import LinearSegmentedColormap
    inner_triangle = plt.Polygon(vertices, fill=True, facecolor='lightyellow',
                                  edgecolor='none', alpha=0.3)
    ax.add_patch(inner_triangle)

    # Vertex labels
    labels = [
        ('Intelligibility (I)', vertices[0], (0, 15)),
        ('Speed (S)', vertices[1], (-15, -15)),
        ('Quality (Q)', vertices[2], (15, -15)),
    ]
    label_colors = [COLORS['intelligibility'], COLORS['speed'], COLORS['quality']]

    for (text, pos, offset), color in zip(labels, label_colors):
        ax.annotate(text, pos, fontsize=14, fontweight='bold', ha='center',
                    xytext=offset, textcoords='offset points', color=color)

    # Edge labels (trade-offs)
    edge_midpoints = [
        ((vertices[0] + vertices[1]) / 2, 'S+I\n(sacrifice: Q)', (-40, 0)),
        ((vertices[0] + vertices[2]) / 2, 'Q+I\n(sacrifice: S)', (40, 0)),
        ((vertices[1] + vertices[2]) / 2, 'S+Q\n(sacrifice: I)', (0, -25)),
    ]
    edge_colors = [COLORS['quality'], COLORS['speed'], COLORS['intelligibility']]

    for (pos, text, offset), color in zip(edge_midpoints, edge_colors):
        ax.annotate(text, pos, fontsize=10, ha='center', va='center',
                    xytext=offset, textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor=color, alpha=0.9, linewidth=1.5))

    # Central "impossible" region
    center = np.mean(vertices, axis=0)
    ax.scatter(*center, s=100, color='red', alpha=0.3, zorder=5)
    ax.annotate('S+Q+I\n(Impossible Region)',
                center, fontsize=11, ha='center', va='center',
                fontweight='bold', color='red', alpha=0.7,
                bbox=dict(boxstyle='round', facecolor='mistyrose',
                         edgecolor='red', alpha=0.5))

    # Arrows showing trade-off directions
    for i in range(3):
        start = center
        end = vertices[i]
        direction = end - start
        arrow_end = start + 0.35 * direction
        ax.annotate('', xy=arrow_end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color='gray',
                                   linewidth=1.5, alpha=0.5))

    # Title and subtitle
    ax.set_title('The SciOps Trilemma\nAI-Driven Scientific Pipeline Trade-offs',
                 fontsize=16, fontweight='bold', pad=30)

    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.2, 1.1)
    ax.set_aspect('equal')
    ax.axis('off')

    # Caption
    ax.text(0.5, -0.15,
            'Any two vertices can be prioritized, but the third must be sacrificed.\n'
            'The central region (simultaneous maximization) is inaccessible under '
            'the condition dC/dα ≥ (I·Q/S)·dA/dα.',
            ha='center', fontsize=10, style='italic', color='gray',
            transform=ax.transAxes)

    if save_path:
        fig.savefig(save_path)
    return fig


# =============================================================================
# 8. Complete Figure Set Generator
# =============================================================================

def generate_all_figures(output_dir: str = "figures",
                         params: TrilemmaParams = None) -> Dict:
    """Generate all publication figures."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    if params is None:
        params = TrilemmaParams(alpha=1.0)

    figures = {}

    print("Generating Figure 1: Trilemma Triangle...")
    figures['triangle'] = plot_trilemma_triangle(
        save_path=f"{output_dir}/fig1_trilemma_triangle.png")

    print("Generating Figure 2: 3D Pareto Surface...")
    figures['pareto_3d'] = plot_trilemma_3d(
        params, save_path=f"{output_dir}/fig2_pareto_3d.png")

    print("Generating Figure 3: Ternary Allocation...")
    figures['ternary'] = plot_ternary_allocation(
        params, save_path=f"{output_dir}/fig3_ternary_allocation.png")

    print("Generating Figure 4: Phase Diagram...")
    figures['phase'] = plot_phase_diagram(
        save_path=f"{output_dir}/fig4_phase_diagram.png")

    print("Generating Figure 5: Epistemic Uncertainty Principle...")
    figures['uncertainty'] = plot_uncertainty_principle(
        params, save_path=f"{output_dir}/fig5_uncertainty_principle.png")

    print("Generating Figure 6: Prediction 3 (Understanding Fraction)...")
    figures['prediction3'] = plot_prediction3(
        save_path=f"{output_dir}/fig6_prediction3.png")

    print("Generating Figure 7: Error Amplification...")
    figures['errors'] = plot_error_amplification(
        params, save_path=f"{output_dir}/fig7_error_amplification.png")

    print(f"\nAll figures saved to {output_dir}/")
    return figures


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    generate_all_figures()
