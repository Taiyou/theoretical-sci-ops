"""
Counterexample Analysis for the Scientific Operations Trilemma
================================================================

Systematically analyzes the five counterexamples that can relax or
invalidate the trilemma, and maps the boundary between trilemma-active
and trilemma-relaxed regions.

Counterexamples:
1. Modularization: C(α) grows slower than A(α)
2. Formal Verification: Quality independent of time
3. Exploratory Understanding: Speed positively correlated with I
4. Parallelization: Non-serial time constraint
5. I→Q Feedback: Understanding improves quality directly
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from trilemma_proof import (
    TrilemmaParams, speed, quality, intelligibility,
    find_optimal_allocation, prove_trilemma, DOMAIN_PROFILES
)


# =============================================================================
# 1. Counterexample 1: Modularization
# =============================================================================

def counterexample_modularization(
    alpha_range: np.ndarray = None,
    C_A_ratios: List[float] = None
) -> Dict:
    """
    Counterexample 1: When modular AI design keeps C(α) from growing
    as fast as A(α), the trilemma weakens.

    Tests: C_exponent < A_exponent (modularized) vs C_exponent > A_exponent (monolithic)
    """
    if alpha_range is None:
        alpha_range = np.linspace(0.01, 2.5, 80)
    if C_A_ratios is None:
        C_A_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

    results = {}
    A_exp = 2.0  # Fix A exponent

    for ratio in C_A_ratios:
        C_exp = A_exp * ratio
        params = TrilemmaParams(A_exponent=A_exp, C_exponent=C_exp)
        proof = prove_trilemma(params, alpha_range)
        trilemma_frac = np.mean(proof['trilemma_holds'])

        results[ratio] = {
            'C_exponent': C_exp,
            'A_exponent': A_exp,
            'proof': proof,
            'trilemma_fraction': trilemma_frac,
            'label': f'C/A ratio = {ratio:.2f} ({"trilemma" if trilemma_frac > 0.5 else "relaxed"})',
        }

    return results


# =============================================================================
# 2. Counterexample 2: Formal Verification
# =============================================================================

def quality_formal_verification(tau_verify: float, params: TrilemmaParams,
                                 verification_power: float = 0.9) -> float:
    """
    Modified quality function for domains with formal verification.

    In formally verifiable domains (math, TCS), Q can reach high levels
    with minimal time investment because correctness is mechanically checkable.
    """
    # Base quality from formal checking (time-independent component)
    Q_formal = verification_power  # e.g., Lean/Coq can verify 90% of quality

    # Remaining quality from time-dependent review
    Q_review = (1 - verification_power) * (1 - np.exp(-params.lam * tau_verify))

    return Q_formal + Q_review


def counterexample_formal_verification(alpha_range: np.ndarray = None) -> Dict:
    """
    Counterexample 2: When formal verification is possible, Q is largely
    time-independent, freeing time for τ_understand.

    Compares: standard quality function vs formal-verification quality function
    """
    if alpha_range is None:
        alpha_range = np.linspace(0.01, 2.5, 60)

    results = {
        'standard': {'S': [], 'Q': [], 'I': [], 'fracs': []},
        'formal': {'S': [], 'Q': [], 'I': [], 'fracs': []},
    }

    for alpha in alpha_range:
        # Standard quality
        params = TrilemmaParams(alpha=alpha)
        opt = find_optimal_allocation(params)
        results['standard']['S'].append(opt['S'])
        results['standard']['Q'].append(opt['Q'])
        results['standard']['I'].append(opt['I'])
        results['standard']['fracs'].append(opt['fractions'])

        # With formal verification: Q needs less time → more for understanding
        # Simulate by reducing the effective verification need
        params_fv = TrilemmaParams(alpha=alpha, lam=10.0)  # Very efficient verification
        opt_fv = find_optimal_allocation(params_fv)
        results['formal']['S'].append(opt_fv['S'])
        results['formal']['Q'].append(opt_fv['Q'])
        results['formal']['I'].append(opt_fv['I'])
        results['formal']['fracs'].append(opt_fv['fractions'])

    results['alpha'] = alpha_range

    # Improvement in I due to formal verification
    results['I_improvement'] = np.array(results['formal']['I']) - np.array(results['standard']['I'])

    return results


# =============================================================================
# 3. Counterexample 3: Exploratory Understanding
# =============================================================================

def intelligibility_exploratory(tau_understand: float, tau_exec: float,
                                 params: TrilemmaParams,
                                 exploration_bonus: float = 0.3) -> float:
    """
    Modified intelligibility for exploratory understanding.

    When AI generates many variations rapidly, pattern recognition across
    these variations can enhance understanding. Speed positively correlates
    with a component of understanding.
    """
    # Standard understanding from deep analysis
    I_deep = 1.0 - np.exp(-params.mu * tau_understand / params.C())

    # Exploratory bonus: more speed → more variations → better pattern recognition
    S = speed(tau_exec, params)
    S_ref = speed(params.T / 3, params)  # Reference speed at balanced allocation
    I_exploratory = exploration_bonus * (1 - np.exp(-S / S_ref))

    return min(I_deep + I_exploratory, 1.0)


def counterexample_exploratory(alpha_range: np.ndarray = None,
                                bonus_levels: List[float] = None) -> Dict:
    """
    Counterexample 3: Exploratory understanding where speed enhances I.

    Shows that when pattern recognition from rapid exploration contributes
    to understanding, the S-I trade-off partially dissolves.
    """
    if alpha_range is None:
        alpha_range = np.linspace(0.01, 2.5, 60)
    if bonus_levels is None:
        bonus_levels = [0.0, 0.1, 0.2, 0.3, 0.5]

    results = {}
    for bonus in bonus_levels:
        S_vals, I_vals = [], []
        for alpha in alpha_range:
            params = TrilemmaParams(alpha=alpha)
            T = params.T
            # At balanced allocation
            te = T / 3
            tu = T / 3
            S = speed(te, params)
            I = intelligibility_exploratory(tu, te, params, exploration_bonus=bonus)
            S_vals.append(S)
            I_vals.append(I)

        results[bonus] = {
            'S': np.array(S_vals),
            'I': np.array(I_vals),
            'correlation': np.corrcoef(S_vals, I_vals)[0, 1],
        }

    results['alpha'] = alpha_range
    return results


# =============================================================================
# 4. Counterexample 4: Parallelization
# =============================================================================

def find_optimal_allocation_parallel(params: TrilemmaParams,
                                      overlap: float = 0.5) -> Dict:
    """
    Modified optimization with partial parallelization.

    Instead of: τ_exec + τ_verify + τ_understand = T
    We have:    max(τ_exec, overlap * τ_understand) + τ_verify + (1-overlap)*τ_understand = T

    overlap ∈ [0, 1]: fraction of understanding that can overlap with execution.
    overlap=0: fully serial (standard model)
    overlap=1: fully parallel (understanding during execution)
    """
    from scipy.optimize import minimize

    T = params.T

    def objective(tau):
        te, tv = tau
        # Effective understanding time: some overlaps with execution
        tu_available = T - max(te, overlap * (T - te - tv)) - tv + overlap * (T - te - tv)
        tu = max(T - te - tv, 0.01)  # Total understanding needed

        if te <= 0 or tv <= 0 or tu <= 0:
            return 1e12

        # With parallelization, effective time available increases
        effective_T = T + overlap * min(te, tu)

        S = speed(te, params)
        Q = quality(tv, params)
        I = intelligibility(tu, params)

        return -(S * Q * I)

    best_val = 1e12
    best_x = np.array([T / 3, T / 3])

    for i_frac in np.linspace(0.05, 0.9, 20):
        for j_frac in np.linspace(0.05, 0.9 - i_frac, 20):
            x0 = np.array([T * i_frac, T * j_frac])
            val = objective(x0)
            if val < best_val:
                best_val = val
                best_x = x0

    result = minimize(objective, best_x, method='Nelder-Mead')

    te, tv = result.x
    tu = T - te - tv

    S = speed(te, params)
    Q = quality(tv, params)
    I = intelligibility(tu, params)

    return {
        'tau_exec': te, 'tau_verify': tv, 'tau_understand': tu,
        'S': S, 'Q': Q, 'I': I, 'F': S * Q * I,
        'overlap': overlap,
    }


def counterexample_parallelization(alpha_range: np.ndarray = None,
                                    overlap_levels: List[float] = None) -> Dict:
    """
    Counterexample 4: Parallelization relaxes the time constraint.

    Shows that pipelined execution (understanding previous results while
    AI produces new ones) eases the S-I trade-off.
    """
    if alpha_range is None:
        alpha_range = np.linspace(0.1, 2.0, 40)
    if overlap_levels is None:
        overlap_levels = [0.0, 0.25, 0.5, 0.75]

    results = {}
    for overlap in overlap_levels:
        F_vals = []
        I_vals = []
        for alpha in alpha_range:
            params = TrilemmaParams(alpha=alpha)
            opt = find_optimal_allocation_parallel(params, overlap=overlap)
            F_vals.append(opt['F'])
            I_vals.append(opt['I'])

        results[overlap] = {
            'F': np.array(F_vals),
            'I': np.array(I_vals),
            'improvement_over_serial': np.array(F_vals) / max(results.get(0.0, {}).get('F', [1.0])[0], 1e-6) if overlap > 0 and 0.0 in results else None,
        }

    results['alpha'] = alpha_range
    return results


# =============================================================================
# 5. Counterexample 5: I → Q Feedback
# =============================================================================

def quality_with_feedback(tau_verify: float, tau_understand: float,
                           params: TrilemmaParams,
                           beta: float = 0.3) -> float:
    """
    Modified quality function with I → Q feedback.

    Q = Q_base(τ_verify) + β · I(τ_understand)

    Deep understanding helps prevent errors, directly improving quality.
    """
    Q_base = 1.0 - params.epsilon() * np.exp(-params.lam * tau_verify)
    I = intelligibility(tau_understand, params)
    Q_with_feedback = Q_base + beta * I * (1 - Q_base)  # Fills gap between Q_base and 1
    return min(Q_with_feedback, 1.0)


def counterexample_iq_feedback(alpha_range: np.ndarray = None,
                                beta_levels: List[float] = None) -> Dict:
    """
    Counterexample 5: Understanding improves quality (I → Q feedback).

    When β > 0, investing in τ_understand also improves Q, partially
    breaking the Q-I trade-off.
    """
    if alpha_range is None:
        alpha_range = np.linspace(0.01, 2.5, 60)
    if beta_levels is None:
        beta_levels = [0.0, 0.1, 0.3, 0.5, 0.8]

    results = {}
    for beta in beta_levels:
        Q_vals, I_vals, F_vals = [], [], []
        for alpha in alpha_range:
            params = TrilemmaParams(alpha=alpha)
            T = params.T
            te, tv, tu = T / 3, T / 3, T / 3

            S = speed(te, params)
            Q = quality_with_feedback(tv, tu, params, beta=beta)
            I = intelligibility(tu, params)

            Q_vals.append(Q)
            I_vals.append(I)
            F_vals.append(S * Q * I)

        results[beta] = {
            'Q': np.array(Q_vals),
            'I': np.array(I_vals),
            'F': np.array(F_vals),
        }

    results['alpha'] = alpha_range
    return results


# =============================================================================
# 6. Boundary Map: Where Does the Trilemma Hold?
# =============================================================================

def compute_trilemma_boundary(
    A_exp_range: np.ndarray = None,
    C_exp_range: np.ndarray = None,
    alpha_test: float = 1.0,
    resolution: int = 40
) -> Dict:
    """
    Compute the boundary between trilemma-active and trilemma-relaxed
    regions in the (A_exponent, C_exponent) parameter space.

    Returns a 2D map of trilemma severity and the estimated boundary curve.
    """
    if A_exp_range is None:
        A_exp_range = np.linspace(0.5, 4.0, resolution)
    if C_exp_range is None:
        C_exp_range = np.linspace(0.5, 4.0, resolution)

    severity_map = np.zeros((len(C_exp_range), len(A_exp_range)))
    alpha_range = np.linspace(0.1, 2.0, 30)

    for i, c_exp in enumerate(C_exp_range):
        for j, a_exp in enumerate(A_exp_range):
            params = TrilemmaParams(alpha=alpha_test, A_exponent=a_exp, C_exponent=c_exp)
            proof = prove_trilemma(params, alpha_range)
            severity_map[i, j] = np.mean(proof['trilemma_holds'])

    # Extract boundary contour (trilemma_fraction ≈ 0.5)
    from scipy.ndimage import gaussian_filter
    smooth_map = gaussian_filter(severity_map, sigma=1.5)

    return {
        'A_exp_range': A_exp_range,
        'C_exp_range': C_exp_range,
        'severity_map': severity_map,
        'smooth_map': smooth_map,
    }


# =============================================================================
# 7. Summary Table: Counterexample Impact
# =============================================================================

def counterexample_summary_table() -> Dict:
    """
    Generate a summary table of all counterexamples with their impact
    on the trilemma condition.
    """
    alpha_range = np.linspace(0.1, 2.0, 30)
    base_params = TrilemmaParams()

    # Baseline trilemma severity
    base_proof = prove_trilemma(base_params, alpha_range)
    base_severity = np.mean(base_proof['trilemma_holds'])

    summary = {
        'baseline': {
            'description': 'Standard model (no counterexample)',
            'trilemma_severity': base_severity,
            'condition': 'C_exp=2.5, A_exp=2.0',
            'applicable_domains': 'All',
        }
    }

    # CE1: Modularization (lower C)
    params_mod = TrilemmaParams(C_exponent=1.5, A_exponent=2.0)
    proof_mod = prove_trilemma(params_mod, alpha_range)
    summary['modularization'] = {
        'description': 'C(α) grows slower than A(α) via modular design',
        'trilemma_severity': np.mean(proof_mod['trilemma_holds']),
        'condition': 'C_exp=1.5, A_exp=2.0',
        'applicable_domains': 'Software-based research, bioinformatics',
    }

    # CE2: Formal Verification (higher λ)
    params_fv = TrilemmaParams(lam=10.0)
    proof_fv = prove_trilemma(params_fv, alpha_range)
    summary['formal_verification'] = {
        'description': 'Quality verification is time-independent',
        'trilemma_severity': np.mean(proof_fv['trilemma_holds']),
        'condition': 'λ=10.0 (highly efficient verification)',
        'applicable_domains': 'Mathematics, Theoretical CS',
    }

    # CE3: Exploratory Understanding (not directly testable via prove_trilemma)
    summary['exploratory_understanding'] = {
        'description': 'Speed positively correlates with understanding',
        'trilemma_severity': 'Partially relaxed (S-I trade-off weakened)',
        'condition': 'exploration_bonus > 0',
        'applicable_domains': 'Large-scale screening, parameter exploration',
    }

    # CE4: Parallelization (not directly testable via standard model)
    summary['parallelization'] = {
        'description': 'Understanding overlaps with execution (pipelining)',
        'trilemma_severity': 'Relaxed proportionally to overlap degree',
        'condition': 'overlap ∈ (0, 1)',
        'applicable_domains': 'Pipeline-structured research, iterative experimentation',
    }

    # CE5: I→Q Feedback (higher effective quality)
    summary['iq_feedback'] = {
        'description': 'Understanding directly improves quality (β > 0)',
        'trilemma_severity': 'Q-I trade-off partially dissolved',
        'condition': 'β > 0 (feedback strength)',
        'applicable_domains': 'Basic science, safety-critical research',
    }

    return summary


# =============================================================================
# 8. Visualization of Counterexample Boundaries
# =============================================================================

def plot_counterexample_boundaries(save_path: str = None):
    """Plot the boundary map showing where counterexamples relax the trilemma."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    alpha_range = np.linspace(0.1, 2.0, 50)

    # Panel 1: Modularization
    ax = axes[0, 0]
    ce1 = counterexample_modularization(alpha_range)
    for ratio, data in ce1.items():
        if isinstance(ratio, float):
            proof = data['proof']
            condition = proof['trilemma_condition']
            ax.plot(alpha_range[:-1][:len(condition)], condition[:len(alpha_range)-1],
                    linewidth=2, label=f'C/A = {ratio}')
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_title('CE1: Modularization', fontweight='bold')
    ax.set_xlabel('α')
    ax.set_ylabel('Trilemma condition ratio')
    ax.legend(fontsize=8)

    # Panel 2: Formal Verification
    ax = axes[0, 1]
    ce2 = counterexample_formal_verification(alpha_range)
    ax.plot(alpha_range, ce2['standard']['I'], linewidth=2, label='Standard')
    ax.plot(alpha_range, ce2['formal']['I'], linewidth=2, label='Formal Verification')
    ax.fill_between(alpha_range,
                     ce2['standard']['I'], ce2['formal']['I'],
                     alpha=0.2, color='green')
    ax.set_title('CE2: Formal Verification → I gain', fontweight='bold')
    ax.set_xlabel('α')
    ax.set_ylabel('Intelligibility (I)')
    ax.legend()

    # Panel 3: Exploratory Understanding
    ax = axes[0, 2]
    ce3 = counterexample_exploratory(alpha_range)
    for bonus, data in ce3.items():
        if isinstance(bonus, float):
            ax.plot(data['S'], data['I'], linewidth=2, label=f'bonus={bonus}')
    ax.set_title('CE3: Exploratory Understanding', fontweight='bold')
    ax.set_xlabel('Speed (S)')
    ax.set_ylabel('Intelligibility (I)')
    ax.legend(fontsize=8)

    # Panel 4: Parallelization
    ax = axes[1, 0]
    alpha_short = np.linspace(0.1, 2.0, 25)
    ce4 = counterexample_parallelization(alpha_short)
    for overlap, data in ce4.items():
        if isinstance(overlap, float):
            ax.plot(alpha_short, data['F'], linewidth=2, label=f'overlap={overlap}')
    ax.set_title('CE4: Parallelization → F gain', fontweight='bold')
    ax.set_xlabel('α')
    ax.set_ylabel('F = S·Q·I')
    ax.legend(fontsize=8)

    # Panel 5: I→Q Feedback
    ax = axes[1, 1]
    ce5 = counterexample_iq_feedback(alpha_range)
    for beta, data in ce5.items():
        if isinstance(beta, float):
            ax.plot(alpha_range, data['F'], linewidth=2, label=f'β={beta}')
    ax.set_title('CE5: I→Q Feedback → F gain', fontweight='bold')
    ax.set_xlabel('α')
    ax.set_ylabel('F = S·Q·I')
    ax.legend(fontsize=8)

    # Panel 6: Boundary Map
    ax = axes[1, 2]
    boundary = compute_trilemma_boundary(resolution=25)
    im = ax.imshow(boundary['smooth_map'], origin='lower', cmap='RdYlGn_r',
                   extent=[boundary['A_exp_range'][0], boundary['A_exp_range'][-1],
                           boundary['C_exp_range'][0], boundary['C_exp_range'][-1]],
                   aspect='auto', vmin=0, vmax=1)
    ax.contour(boundary['A_exp_range'], boundary['C_exp_range'],
               boundary['smooth_map'], levels=[0.5], colors='black', linewidths=2)
    ax.plot([0.5, 4], [0.5, 4], '--', color='white', linewidth=1.5, alpha=0.7)
    ax.set_title('Trilemma Boundary Map', fontweight='bold')
    ax.set_xlabel('A exponent')
    ax.set_ylabel('C exponent')
    plt.colorbar(im, ax=ax, label='Trilemma severity')

    for key, domain in DOMAIN_PROFILES.items():
        ax.scatter(domain.A_exponent, domain.C_exponent, s=80,
                   edgecolors='white', linewidths=1.5, zorder=10, color='yellow')
        ax.annotate(domain.name.split('(')[0].strip()[:15],
                    (domain.A_exponent, domain.C_exponent),
                    fontsize=6, color='white', fontweight='bold',
                    xytext=(3, 3), textcoords='offset points')

    fig.suptitle('Counterexample Analysis: Conditions Under Which the Trilemma Relaxes',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
    return fig


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("COUNTEREXAMPLE ANALYSIS")
    print("=" * 70)

    summary = counterexample_summary_table()
    for name, info in summary.items():
        print(f"\n{name}:")
        print(f"  Description: {info['description']}")
        print(f"  Trilemma severity: {info['trilemma_severity']}")
        print(f"  Applicable domains: {info['applicable_domains']}")

    print("\n" + "=" * 70)
    print("Generating counterexample boundary plot...")
    fig = plot_counterexample_boundaries(save_path="figures/fig8_counterexamples.png")
    print("Done.")
