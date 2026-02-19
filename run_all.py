#!/usr/bin/env python3
"""
SciOps Trilemma: Complete Analysis Runner
==========================================

Runs all analyses and generates all figures for the paper:
1. Theoretical proof of the trilemma
2. Epistemic uncertainty principle derivation
3. Pipeline simulations (standard, hierarchical, intermittent)
4. Counterexample analysis
5. Publication-quality visualizations

Usage:
    python run_all.py                    # Run everything
    python run_all.py --proof-only       # Only theoretical proof
    python run_all.py --sim-only         # Only simulations
    python run_all.py --viz-only         # Only visualizations
    python run_all.py --quick            # Quick run (fewer data points)
"""

import os
import sys
import time
import argparse
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_proof(quick: bool = False):
    """Run the theoretical proof analysis."""
    from trilemma_proof import (
        TrilemmaParams, prove_trilemma, epistemic_uncertainty,
        error_amplification, understanding_fraction_vs_alpha,
        trilemma_condition_check, find_optimal_allocation,
        DOMAIN_PROFILES, analyze_domain
    )

    print("\n" + "=" * 70)
    print("PART 1: THEORETICAL PROOF OF THE TRILEMMA")
    print("=" * 70)

    params = TrilemmaParams(alpha=1.0)
    n_alpha = 30 if quick else 80
    alpha_range = np.linspace(0.01, 2.5, n_alpha)

    # 1.1 Optimal allocation at various α
    print("\n--- 1.1 Optimal Time Allocation ---")
    print(f"{'α':>6} {'τ_exec':>8} {'τ_verify':>8} {'τ_under':>8} {'S':>8} {'Q':>8} {'I':>8} {'F':>8}")
    print("-" * 70)
    for alpha in [0.1, 0.5, 1.0, 1.5, 2.0]:
        p = TrilemmaParams(alpha=alpha)
        opt = find_optimal_allocation(p)
        print(f"{alpha:6.1f} {opt['tau_exec']:8.3f} {opt['tau_verify']:8.3f} "
              f"{opt['tau_understand']:8.3f} {opt['S']:8.2f} {opt['Q']:8.3f} "
              f"{opt['I']:8.3f} {opt['F']:8.2f}")

    # 1.2 Trilemma proof
    print("\n--- 1.2 Trilemma Proof ---")
    proof = prove_trilemma(params, alpha_range)
    trilemma_pct = np.mean(proof['trilemma_holds']) * 100
    print(f"Trilemma holds for {trilemma_pct:.1f}% of α ∈ [0.01, 2.5]")

    # Check where all three improve simultaneously
    all_improve = (~proof['trilemma_holds']).sum()
    print(f"All three metrics improve simultaneously: {all_improve} / {len(proof['trilemma_holds'])} points")

    # 1.3 Trilemma condition
    print("\n--- 1.3 Trilemma Condition: dC/dα ≥ (I·Q/S)·dA/dα ---")
    for alpha in [0.1, 0.5, 1.0, 1.5, 2.0]:
        p = TrilemmaParams(alpha=alpha)
        cond = trilemma_condition_check(p)
        status = "HOLDS" if cond['condition_holds'] else "RELAXED"
        print(f"  α={alpha:.1f}: ratio={cond['ratio']:.2f} [{status}]")

    # 1.4 Epistemic Uncertainty Principle
    print("\n--- 1.4 Epistemic Uncertainty Principle ---")
    eup = epistemic_uncertainty(params, alpha_range)
    print(f"κ(α=0.01) = {eup['kappa'][0]:.4f}")
    print(f"κ(α=2.50) = {eup['kappa'][-1]:.4f}")
    print(f"κ ratio (α=2.5 / α=0.01): {eup['kappa'][-1] / eup['kappa'][0]:.2f}x")
    print(f"  → The uncertainty bound grows {eup['kappa'][-1] / eup['kappa'][0]:.1f}× "
          f"as AI capability increases from 0 to 2.5")

    # 1.5 Prediction 3: Understanding fraction
    print("\n--- 1.5 Prediction 3: Understanding Fraction vs. α ---")
    fracs = understanding_fraction_vs_alpha(params, alpha_range)
    print(f"τ_understand/T at α=0.01: {fracs['f_understand'][0]:.3f}")
    print(f"τ_understand/T at α=2.50: {fracs['f_understand'][-1]:.3f}")
    if fracs['f_understand'][-1] > fracs['f_understand'][0]:
        print("  → CONFIRMED: Understanding fraction INCREASES with AI capability")
        print("    'The better AI becomes, the MORE time humans should spend understanding.'")
    else:
        print("  → NOT CONFIRMED at these parameter values")

    # 1.6 Error amplification
    print("\n--- 1.6 Error Amplification (Information-Theoretic) ---")
    amp = error_amplification(params, alpha_range)
    L_ratio = amp['L'][-1] / amp['L'][0]
    S_ratio = amp['S'][-1] / amp['S'][0]
    print(f"Speed increase (α=0→2.5): {S_ratio:.1f}×")
    print(f"Loss increase (α=0→2.5):  {L_ratio:.1f}×")
    print(f"Superlinearity: L grows {L_ratio / S_ratio:.1f}× faster than speed alone")

    # 1.7 Domain analysis
    print("\n--- 1.7 Domain-Specific Analysis ---")
    domain_alpha = np.linspace(0.1, 2.0, 20 if quick else 40)
    for key, domain in DOMAIN_PROFILES.items():
        result = analyze_domain(domain, domain_alpha)
        print(f"  {domain.name}: trilemma holds {result['trilemma_fraction']*100:.0f}% "
              f"[severity: {domain.trilemma_severity}]")

    return proof, eup, fracs, amp


def run_simulations(quick: bool = False):
    """Run pipeline simulations."""
    from simulation_engine import (
        PipelineSimulation, HierarchicalPipeline,
        IntermittentDeceleration, optimize_sprint_cycle,
        run_strategy_comparison
    )
    from trilemma_proof import TrilemmaParams

    print("\n" + "=" * 70)
    print("PART 2: PIPELINE SIMULATIONS")
    print("=" * 70)

    n_steps = 50 if quick else 200

    # 2.1 Strategy comparison at different α
    print("\n--- 2.1 Strategy Comparison ---")
    alpha_vals = [0.1, 0.5, 1.0, 1.5, 2.0]
    strategies = ["speed_quality", "speed_understanding", "quality_understanding", "balanced"]

    print(f"{'Strategy':<25} {'α':>5} {'Units':>7} {'TrueQ':>7} {'Gap':>7} {'Drift':>7} {'Rework':>7}")
    print("-" * 75)

    for alpha in alpha_vals:
        params = TrilemmaParams(alpha=alpha)
        for strategy in strategies:
            sim = PipelineSimulation(params, n_steps=n_steps)
            result = sim.run_standard(strategy)
            print(f"{strategy:<25} {alpha:5.1f} {result['total_units']:7d} "
                  f"{result['mean_true_quality']:7.3f} {result['quality_gap']:7.3f} "
                  f"{result['final_goodhart_drift']:7.3f} {result['total_rework_cost']:7.1f}")

        # Hierarchical
        sim_h = HierarchicalPipeline(params, n_steps=n_steps)
        result_h = sim_h.run()
        print(f"{'hierarchical':<25} {alpha:5.1f} {result_h['total_units']:7d} "
              f"{result_h['mean_true_quality']:7.3f} {result_h['quality_gap']:7.3f} "
              f"{result_h['final_goodhart_drift']:7.3f} {result_h['total_rework_cost']:7.1f}")

        # Intermittent
        sim_i = IntermittentDeceleration(params, n_steps=n_steps)
        result_i = sim_i.run()
        print(f"{'intermittent_decel':<25} {alpha:5.1f} {result_i['total_units']:7d} "
              f"{result_i['mean_true_quality']:7.3f} {result_i['quality_gap']:7.3f} "
              f"{result_i['final_goodhart_drift']:7.3f} {result_i['total_rework_cost']:7.1f}")
        print()

    # 2.2 Sprint cycle optimization
    if not quick:
        print("\n--- 2.2 Sprint Cycle Optimization ---")
        params = TrilemmaParams()
        sprint_results = optimize_sprint_cycle(
            params,
            sprint_range=range(5, 30, 5),
            reflect_range=range(2, 15, 3),
            alpha_range=np.array([0.5, 1.0, 2.0]),
            n_steps=100
        )

        # Find optimal for each α
        for alpha in [0.5, 1.0, 2.0]:
            alpha_results = [r for r in sprint_results if r['alpha'] == alpha]
            best = max(alpha_results, key=lambda r: r['effective_output'])
            print(f"  α={alpha}: optimal sprint={best['sprint_length']}, "
                  f"reflect={best['reflect_length']} "
                  f"(reflect ratio={best['reflect_ratio']:.2f}, "
                  f"eff output={best['effective_output']:.1f})")

    return True


def run_counterexamples(quick: bool = False):
    """Run counterexample analysis."""
    from counterexamples import (
        counterexample_modularization,
        counterexample_formal_verification,
        counterexample_exploratory,
        counterexample_iq_feedback,
        counterexample_summary_table,
    )

    print("\n" + "=" * 70)
    print("PART 3: COUNTEREXAMPLE ANALYSIS")
    print("=" * 70)

    alpha_range = np.linspace(0.1, 2.0, 25 if quick else 50)

    # Summary table
    print("\n--- Counterexample Summary ---")
    summary = counterexample_summary_table()
    for name, info in summary.items():
        severity = info['trilemma_severity']
        if isinstance(severity, float):
            severity = f"{severity:.1%}"
        print(f"\n  {name}:")
        print(f"    {info['description']}")
        print(f"    Trilemma severity: {severity}")
        print(f"    Domains: {info['applicable_domains']}")

    # CE1: Modularization
    print("\n--- CE1: Modularization ---")
    ce1 = counterexample_modularization(alpha_range)
    for ratio, data in ce1.items():
        if isinstance(ratio, float):
            print(f"  C/A ratio={ratio:.2f}: trilemma holds {data['trilemma_fraction']:.1%}")

    # CE2: Formal Verification
    print("\n--- CE2: Formal Verification ---")
    ce2 = counterexample_formal_verification(alpha_range)
    avg_I_gain = np.mean(ce2['I_improvement'])
    print(f"  Average I improvement from formal verification: {avg_I_gain:.3f}")

    # CE3: Exploratory Understanding
    print("\n--- CE3: Exploratory Understanding ---")
    ce3 = counterexample_exploratory(alpha_range)
    for bonus, data in ce3.items():
        if isinstance(bonus, float):
            print(f"  bonus={bonus}: S-I correlation = {data['correlation']:.3f}")

    return summary


def run_visualizations(quick: bool = False):
    """Generate all figures."""
    from visualize_trilemma import generate_all_figures, plot_trilemma_triangle
    from counterexamples import plot_counterexample_boundaries
    from trilemma_proof import TrilemmaParams

    print("\n" + "=" * 70)
    print("PART 4: GENERATING VISUALIZATIONS")
    print("=" * 70)

    os.makedirs("figures", exist_ok=True)

    params = TrilemmaParams(alpha=1.0)
    figures = generate_all_figures(output_dir="figures", params=params)

    print("\nGenerating Figure 8: Counterexample Boundaries...")
    fig8 = plot_counterexample_boundaries(save_path="figures/fig8_counterexamples.png")
    figures['counterexamples'] = fig8

    print(f"\nTotal figures generated: {len(figures)}")
    print("Saved to: figures/")

    return figures


def main():
    parser = argparse.ArgumentParser(description="SciOps Trilemma Analysis")
    parser.add_argument('--proof-only', action='store_true', help='Only run proof')
    parser.add_argument('--sim-only', action='store_true', help='Only run simulations')
    parser.add_argument('--viz-only', action='store_true', help='Only run visualizations')
    parser.add_argument('--counterexamples-only', action='store_true', help='Only run counterexamples')
    parser.add_argument('--quick', action='store_true', help='Quick run with fewer points')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization generation')
    args = parser.parse_args()

    start = time.time()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          SciOps Trilemma: Complete Analysis Suite           ║")
    print("║                                                            ║")
    print("║  Speed · Quality · Intelligibility                         ║")
    print("║  'You can optimize any two, but not all three.'            ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    run_all = not any([args.proof_only, args.sim_only, args.viz_only,
                       args.counterexamples_only])

    if run_all or args.proof_only:
        run_proof(quick=args.quick)

    if run_all or args.sim_only:
        run_simulations(quick=args.quick)

    if run_all or args.counterexamples_only:
        run_counterexamples(quick=args.quick)

    if (run_all or args.viz_only) and not args.no_viz:
        run_visualizations(quick=args.quick)

    elapsed = time.time() - start
    print(f"\n{'=' * 70}")
    print(f"COMPLETE. Total time: {elapsed:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
