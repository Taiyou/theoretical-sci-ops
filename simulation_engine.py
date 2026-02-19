"""
Simulation Engine for the Scientific Operations Trilemma
=========================================================

Implements the pipeline simulation model with:
- Approach 1: Hierarchical Separation (Speed Layer / Understanding Layer / Integration Layer)
- Approach 4: Intermittent Deceleration (Sprint/Reflect cycles)
- Goodhart Effect drift modeling
- Rework cost amplification

Each simulation runs a discrete-time pipeline where research units flow through
stages, and the trilemma trade-offs manifest in measurable outcomes.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from trilemma_proof import TrilemmaParams, speed, quality, intelligibility


# =============================================================================
# 1. Research Unit
# =============================================================================

@dataclass
class ResearchUnit:
    """A single unit of research flowing through the pipeline."""
    id: int
    true_quality: float = 0.0       # Ground-truth quality (unknown to agents)
    measured_quality: float = 0.0    # Proxy-measured quality (subject to Goodhart)
    understood: bool = False          # Whether a human has understood this result
    understanding_depth: float = 0.0  # Depth of understanding [0, 1]
    reworked: bool = False
    rework_cost: float = 0.0
    production_step: int = 0


# =============================================================================
# 2. Goodhart Drift Model
# =============================================================================

class GoodhartDrift:
    """
    Models the divergence between proxy metrics and true quality.

    As the pipeline optimizes for proxy metrics, the mapping between
    proxy and true quality drifts. The drift accumulates over time
    unless corrected by human understanding (the Integration Layer).
    """

    def __init__(self, drift_rate: float = 0.01, correction_strength: float = 0.5):
        self.drift_rate = drift_rate
        self.correction_strength = correction_strength
        self.accumulated_drift = 0.0
        self.drift_history = [0.0]

    def step(self, n_units_produced: int, n_units_understood: int):
        """
        Update drift for one time step.

        Drift increases with production volume (more optimization pressure)
        and decreases with understanding (human correction).
        """
        # Drift accumulates proportionally to production without understanding
        drift_increase = self.drift_rate * n_units_produced
        drift_correction = self.correction_strength * n_units_understood

        self.accumulated_drift += drift_increase - drift_correction
        self.accumulated_drift = max(self.accumulated_drift, 0.0)
        self.drift_history.append(self.accumulated_drift)

    def proxy_to_true(self, proxy_quality: float) -> float:
        """Convert proxy quality to true quality given current drift."""
        # As drift increases, proxy quality overestimates true quality
        discount = np.exp(-self.accumulated_drift)
        return proxy_quality * discount + (1 - discount) * 0.3  # Regress to noise

    def reset_drift(self, fraction: float = 0.0):
        """Partially reset drift (e.g., after a reflection phase)."""
        self.accumulated_drift *= (1 - fraction)


# =============================================================================
# 3. Base Pipeline Simulation
# =============================================================================

class PipelineSimulation:
    """
    Base simulation of a scientific pipeline operating under the trilemma.

    At each time step:
    1. AI produces research units (speed depends on α and allocation)
    2. Quality is assessed (proxy metric, subject to Goodhart drift)
    3. Understanding is attempted for some fraction of outputs
    4. Rework occurs when true quality falls below threshold
    """

    def __init__(self, params: TrilemmaParams, n_steps: int = 200,
                 rework_threshold: float = 0.5, rng_seed: int = 42):
        self.params = params
        self.n_steps = n_steps
        self.rework_threshold = rework_threshold
        self.rng = np.random.RandomState(rng_seed)
        self.goodhart = GoodhartDrift()

        # Tracking
        self.units_produced: List[ResearchUnit] = []
        self.step_metrics: List[Dict] = []
        self.total_rework_cost = 0.0
        self.unit_counter = 0

    def produce_units(self, tau_exec: float, step: int) -> List[ResearchUnit]:
        """Produce research units during the execution phase."""
        S = speed(tau_exec, self.params)
        n_units = max(1, int(S * tau_exec))  # Number of units produced

        units = []
        for _ in range(n_units):
            self.unit_counter += 1

            # True quality: drawn from distribution, degraded by Goodhart drift
            base_quality = self.rng.beta(3, 1.5)  # Skewed toward higher quality
            true_q = self.goodhart.proxy_to_true(base_quality)

            # Measured (proxy) quality: what the pipeline "sees"
            proxy_q = base_quality + self.rng.normal(0, 0.05)
            proxy_q = np.clip(proxy_q, 0, 1)

            unit = ResearchUnit(
                id=self.unit_counter,
                true_quality=true_q,
                measured_quality=proxy_q,
                production_step=step
            )
            units.append(unit)

        return units

    def verify_units(self, units: List[ResearchUnit], tau_verify: float):
        """Quality verification phase."""
        Q = quality(tau_verify, self.params)
        for unit in units:
            # Verification catches some errors
            if self.rng.random() < Q:
                # Verified: measured quality gets closer to true quality
                unit.measured_quality = 0.7 * unit.true_quality + 0.3 * unit.measured_quality
            # Else: proxy quality remains potentially misleading

    def understand_units(self, units: List[ResearchUnit], tau_understand: float,
                         fraction: float = 1.0):
        """Human understanding phase. Only a fraction of units may be understood."""
        I_capacity = intelligibility(tau_understand, self.params)
        n_to_understand = max(1, int(len(units) * fraction))

        understood_count = 0
        for unit in units[:n_to_understand]:
            if self.rng.random() < I_capacity:
                unit.understood = True
                unit.understanding_depth = I_capacity
                understood_count += 1

        return understood_count

    def check_rework(self, units: List[ResearchUnit]) -> float:
        """Check for rework needs and compute rework costs."""
        total_cost = 0.0
        for unit in units:
            if unit.true_quality < self.rework_threshold and not unit.understood:
                # Low true quality AND not understood = rework needed
                cost = speed(0.1, self.params) * 0.1  # Proportional to speed
                unit.reworked = True
                unit.rework_cost = cost
                total_cost += cost
        return total_cost

    def run_standard(self, strategy: str = "balanced") -> Dict:
        """
        Run simulation with a standard (non-hierarchical) pipeline.

        Strategies:
        - "speed_quality": prioritize speed + quality (sacrifice understanding)
        - "speed_understanding": prioritize speed + understanding (sacrifice quality)
        - "quality_understanding": prioritize quality + understanding (sacrifice speed)
        - "balanced": equal allocation
        """
        T = self.params.T
        allocations = {
            "speed_quality":         (0.3 * T, 0.6 * T, 0.1 * T),
            "speed_understanding":   (0.3 * T, 0.1 * T, 0.6 * T),
            "quality_understanding": (0.1 * T, 0.45 * T, 0.45 * T),
            "balanced":              (T / 3, T / 3, T / 3),
        }
        tau_exec, tau_verify, tau_understand = allocations[strategy]

        for step in range(self.n_steps):
            units = self.produce_units(tau_exec, step)
            self.verify_units(units, tau_verify)
            n_understood = self.understand_units(units, tau_understand)
            rework_cost = self.check_rework(units)

            self.goodhart.step(len(units), n_understood)
            self.total_rework_cost += rework_cost
            self.units_produced.extend(units)

            self.step_metrics.append({
                'step': step,
                'n_produced': len(units),
                'n_understood': n_understood,
                'mean_true_quality': np.mean([u.true_quality for u in units]),
                'mean_proxy_quality': np.mean([u.measured_quality for u in units]),
                'rework_cost': rework_cost,
                'goodhart_drift': self.goodhart.accumulated_drift,
                'understanding_rate': n_understood / max(len(units), 1),
            })

        return self._compile_results(strategy)

    def _compile_results(self, strategy: str) -> Dict:
        """Compile simulation results."""
        metrics = self.step_metrics
        units = self.units_produced

        return {
            'strategy': strategy,
            'alpha': self.params.alpha,
            'n_steps': self.n_steps,
            'total_units': len(units),
            'total_understood': sum(1 for u in units if u.understood),
            'total_rework_cost': self.total_rework_cost,
            'mean_true_quality': np.mean([u.true_quality for u in units]),
            'mean_proxy_quality': np.mean([u.measured_quality for u in units]),
            'final_goodhart_drift': self.goodhart.accumulated_drift,
            'quality_gap': np.mean([u.measured_quality - u.true_quality for u in units]),
            'step_metrics': metrics,
            'goodhart_history': self.goodhart.drift_history,
        }


# =============================================================================
# 4. Hierarchical Separation (Approach 1)
# =============================================================================

class HierarchicalPipeline(PipelineSimulation):
    """
    Implements the hierarchical separation approach:
    - Speed Layer: AI processes all units at maximum speed
    - Understanding Layer: Important units are routed for deep human understanding
    - Integration Layer: Understanding feeds back to correct Goodhart drift

    Key parameter: importance_threshold determines routing between layers.
    """

    def __init__(self, params: TrilemmaParams, n_steps: int = 200,
                 importance_threshold: float = 0.7,
                 feedback_frequency: int = 20,
                 **kwargs):
        super().__init__(params, n_steps, **kwargs)
        self.importance_threshold = importance_threshold
        self.feedback_frequency = feedback_frequency

    def route_to_understanding(self, units: List[ResearchUnit]) -> Tuple[List, List]:
        """Route units: high-importance to understanding layer, rest to fast path."""
        important = [u for u in units if u.measured_quality >= self.importance_threshold]
        fast_path = [u for u in units if u.measured_quality < self.importance_threshold]
        return important, fast_path

    def run(self) -> Dict:
        """Run the hierarchical pipeline."""
        T = self.params.T

        # Speed Layer gets most of the time
        tau_exec_speed = 0.2 * T
        tau_verify_speed = 0.6 * T  # AI verification
        tau_understand_speed = 0.2 * T  # Minimal understanding

        # Understanding Layer allocations (applied only to important units)
        tau_understand_deep = 0.8 * T

        for step in range(self.n_steps):
            # Speed Layer: produce and quickly verify
            units = self.produce_units(tau_exec_speed, step)
            self.verify_units(units, tau_verify_speed)

            # Route
            important, fast_path = self.route_to_understanding(units)

            # Fast path: minimal understanding
            n_understood_fast = self.understand_units(fast_path, tau_understand_speed, fraction=0.1)

            # Understanding Layer: deep understanding of important units
            n_understood_deep = self.understand_units(important, tau_understand_deep, fraction=1.0)

            # Integration Layer: periodic feedback
            total_understood = n_understood_fast + n_understood_deep
            if step % self.feedback_frequency == 0 and step > 0:
                # Correct Goodhart drift based on understanding
                correction = min(total_understood / max(len(units), 1), 1.0)
                self.goodhart.reset_drift(correction * 0.5)

            rework_cost = self.check_rework(units)
            self.goodhart.step(len(units), total_understood)
            self.total_rework_cost += rework_cost
            self.units_produced.extend(units)

            self.step_metrics.append({
                'step': step,
                'n_produced': len(units),
                'n_important': len(important),
                'n_fast_path': len(fast_path),
                'n_understood': total_understood,
                'mean_true_quality': np.mean([u.true_quality for u in units]),
                'mean_proxy_quality': np.mean([u.measured_quality for u in units]),
                'rework_cost': rework_cost,
                'goodhart_drift': self.goodhart.accumulated_drift,
            })

        return self._compile_results("hierarchical")


# =============================================================================
# 5. Intermittent Deceleration (Approach 4)
# =============================================================================

class IntermittentDeceleration(PipelineSimulation):
    """
    Implements the sprint/reflect cycle:
    - Acceleration Phase: maximize speed + procedural quality
    - Reflection Phase: slow down, maximize understanding, correct Goodhart drift

    Key parameters:
    - sprint_length: number of steps in acceleration phase
    - reflect_length: number of steps in reflection phase
    """

    def __init__(self, params: TrilemmaParams, n_steps: int = 200,
                 sprint_length: int = 15, reflect_length: int = 5,
                 **kwargs):
        super().__init__(params, n_steps, **kwargs)
        self.sprint_length = sprint_length
        self.reflect_length = reflect_length

    def run(self) -> Dict:
        """Run the sprint/reflect pipeline."""
        T = self.params.T
        cycle_length = self.sprint_length + self.reflect_length

        for step in range(self.n_steps):
            phase_position = step % cycle_length
            is_sprint = phase_position < self.sprint_length

            if is_sprint:
                # Sprint: speed + quality, minimal understanding
                tau_exec = 0.3 * T
                tau_verify = 0.6 * T
                tau_understand = 0.1 * T
                understand_fraction = 0.1
            else:
                # Reflect: slow down, deep understanding
                tau_exec = 0.1 * T
                tau_verify = 0.2 * T
                tau_understand = 0.7 * T
                understand_fraction = 1.0

                # Correct Goodhart drift during reflection
                if phase_position == self.sprint_length:
                    self.goodhart.reset_drift(0.3)

            units = self.produce_units(tau_exec, step)
            self.verify_units(units, tau_verify)
            n_understood = self.understand_units(units, tau_understand, understand_fraction)
            rework_cost = self.check_rework(units)

            self.goodhart.step(len(units), n_understood)
            self.total_rework_cost += rework_cost
            self.units_produced.extend(units)

            self.step_metrics.append({
                'step': step,
                'n_produced': len(units),
                'n_understood': n_understood,
                'mean_true_quality': np.mean([u.true_quality for u in units]),
                'mean_proxy_quality': np.mean([u.measured_quality for u in units]),
                'rework_cost': rework_cost,
                'goodhart_drift': self.goodhart.accumulated_drift,
                'is_sprint': is_sprint,
            })

        return self._compile_results("intermittent_deceleration")


# =============================================================================
# 6. Sprint Cycle Optimization
# =============================================================================

def optimize_sprint_cycle(params: TrilemmaParams,
                          sprint_range: range = range(5, 50, 5),
                          reflect_range: range = range(2, 20, 2),
                          alpha_range: np.ndarray = None,
                          n_steps: int = 200) -> Dict:
    """
    Parametric search for optimal sprint/reflect cycle lengths.

    For each (sprint_length, reflect_length, α) combination, run a simulation
    and measure the effective output (quality-adjusted, understanding-weighted).

    Tests Prediction: Higher α requires more frequent reflection phases,
    but too-frequent reflection cancels speed benefits.
    """
    if alpha_range is None:
        alpha_range = np.array([0.1, 0.5, 1.0, 1.5, 2.0])

    results = []

    for alpha in alpha_range:
        for sprint_len in sprint_range:
            for reflect_len in reflect_range:
                p = TrilemmaParams(
                    alpha=alpha, T=params.T, lam=params.lam, mu=params.mu,
                    A_scale=params.A_scale, A_exponent=params.A_exponent,
                    C_scale=params.C_scale, C_exponent=params.C_exponent,
                    eps_base=params.eps_base, eps_reduction=params.eps_reduction
                )
                sim = IntermittentDeceleration(
                    p, n_steps=n_steps,
                    sprint_length=sprint_len,
                    reflect_length=reflect_len
                )
                result = sim.run()

                # Effective output: quality × understanding rate × production
                understood_fraction = result['total_understood'] / max(result['total_units'], 1)
                effective_output = (
                    result['total_units'] *
                    result['mean_true_quality'] *
                    (0.5 + 0.5 * understood_fraction)  # Weighted by understanding
                    - result['total_rework_cost']
                )

                results.append({
                    'alpha': alpha,
                    'sprint_length': sprint_len,
                    'reflect_length': reflect_len,
                    'total_units': result['total_units'],
                    'mean_true_quality': result['mean_true_quality'],
                    'understood_fraction': understood_fraction,
                    'final_drift': result['final_goodhart_drift'],
                    'total_rework': result['total_rework_cost'],
                    'effective_output': effective_output,
                    'reflect_ratio': reflect_len / (sprint_len + reflect_len),
                })

    return results


# =============================================================================
# 7. Comparative Analysis Across Strategies
# =============================================================================

def run_strategy_comparison(alpha_range: np.ndarray = None,
                            n_steps: int = 200,
                            n_seeds: int = 5) -> Dict:
    """
    Compare all strategies across different AI capability levels.

    Strategies compared:
    1. Speed + Quality (sacrifice understanding)
    2. Speed + Understanding (sacrifice quality)
    3. Quality + Understanding (sacrifice speed)
    4. Balanced
    5. Hierarchical Separation
    6. Intermittent Deceleration
    """
    if alpha_range is None:
        alpha_range = np.linspace(0.1, 2.0, 10)

    strategies = ["speed_quality", "speed_understanding",
                   "quality_understanding", "balanced"]

    all_results = {s: [] for s in strategies + ["hierarchical", "intermittent"]}

    for alpha in alpha_range:
        for seed in range(n_seeds):
            params = TrilemmaParams(alpha=alpha)

            # Standard strategies
            for strategy in strategies:
                sim = PipelineSimulation(params, n_steps=n_steps, rng_seed=seed + 100)
                result = sim.run_standard(strategy)
                all_results[strategy].append(result)

            # Hierarchical
            sim_h = HierarchicalPipeline(params, n_steps=n_steps, rng_seed=seed + 100)
            result_h = sim_h.run()
            all_results["hierarchical"].append(result_h)

            # Intermittent deceleration
            sim_i = IntermittentDeceleration(params, n_steps=n_steps, rng_seed=seed + 100)
            result_i = sim_i.run()
            all_results["intermittent"].append(result_i)

    return all_results


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SIMULATION ENGINE: Quick Test")
    print("=" * 70)

    params = TrilemmaParams(alpha=1.0)

    # Test standard pipeline
    print("\n--- Standard Pipeline (Balanced) ---")
    sim = PipelineSimulation(params, n_steps=100)
    result = sim.run_standard("balanced")
    print(f"  Units produced: {result['total_units']}")
    print(f"  Mean true quality: {result['mean_true_quality']:.3f}")
    print(f"  Mean proxy quality: {result['mean_proxy_quality']:.3f}")
    print(f"  Quality gap: {result['quality_gap']:.3f}")
    print(f"  Goodhart drift: {result['final_goodhart_drift']:.3f}")
    print(f"  Rework cost: {result['total_rework_cost']:.3f}")

    # Test hierarchical pipeline
    print("\n--- Hierarchical Pipeline ---")
    sim_h = HierarchicalPipeline(params, n_steps=100)
    result_h = sim_h.run()
    print(f"  Units produced: {result_h['total_units']}")
    print(f"  Mean true quality: {result_h['mean_true_quality']:.3f}")
    print(f"  Goodhart drift: {result_h['final_goodhart_drift']:.3f}")
    print(f"  Rework cost: {result_h['total_rework_cost']:.3f}")

    # Test intermittent deceleration
    print("\n--- Intermittent Deceleration ---")
    sim_i = IntermittentDeceleration(params, n_steps=100)
    result_i = sim_i.run()
    print(f"  Units produced: {result_i['total_units']}")
    print(f"  Mean true quality: {result_i['mean_true_quality']:.3f}")
    print(f"  Goodhart drift: {result_i['final_goodhart_drift']:.3f}")
    print(f"  Rework cost: {result_i['total_rework_cost']:.3f}")

    print("\n" + "=" * 70)
    print("SIMULATION TEST COMPLETE")
    print("=" * 70)
