"""
environment.py — Continuous Learning Environment Loop
========================================================

The main simulation loop that drives the brain-inspired system.
Streams input, updates neurons, applies plasticity rules, and
maintains the "live" evolving state. The system never stops learning.

Execution Flow (per timestep):
  1. Generate/receive input spikes (Poisson process or external source)
  2. Propagate spikes through sparse weight matrix → synaptic currents
  3. Update liquid dynamics (CfC modulation)
  4. Update membrane potentials (LIF dynamics + liquid modulation)
  5. Detect spikes (V > threshold)
  6. Apply STDP weight updates (local, no backprop)
  7. Apply reward-modulated STDP (if reward signal present)
  8. [Every 100 timesteps] Structural plasticity (prune + grow + CSR rebuild)
  9. [Every 50 timesteps] Homeostatic scaling
  10. [Every 500 timesteps] Metaplasticity adjustment
  11. Log metrics

Two modes:
  - UNSUPERVISED: Pure STDP, no reward signal
  - REWARD_MODULATED: External reward signal modulates plasticity
"""

import numpy as np
import time
import config as cfg
from brain_node import BrainNetwork
from synapse_manager import SynapseManager
from liquid_dynamics import LiquidTimeConstant
from homeostasis import HomeostasisController
from metaplasticity import MetaplasticityController


class EnvironmentLoop:
    """
    Continuous learning loop that drives the neuromorphic system.

    Maintains all components and orchestrates their interactions.
    """

    def __init__(self, mode: str = "unsupervised"):
        """
        Parameters
        ----------
        mode : str
            "unsupervised" or "reward_modulated"
        """
        self.mode = mode

        # ── Initialize all components ───────────────────────────────
        print("[ENV] Initializing seed network...")
        self.brain = BrainNetwork(cfg.N_NEURONS)
        self.synapses = SynapseManager(cfg.N_NEURONS)
        self.liquid = LiquidTimeConstant(cfg.N_NEURONS)
        self.homeostasis = HomeostasisController(cfg.N_NEURONS)
        self.metaplasticity = MetaplasticityController(cfg.N_NEURONS)

        # ── Input generation state ──────────────────────────────────
        self.rng = np.random.default_rng()

        # ── Reward signal (for reward-modulated mode) ───────────────
        self.reward_signal = 0.0

        # ── Metrics logging ─────────────────────────────────────────
        self.metrics_log = {
            "timestep": [],
            "num_spikes": [],
            "mean_firing_rate": [],
            "num_synapses": [],
            "mean_weight": [],
            "weight_std": [],
            "stability_cv": [],
            "liquid_mean_h": [],
            "a_plus": [],
            "a_minus": [],
            "elapsed_ms": [],
        }

        # ── Timestep counter ────────────────────────────────────────
        self.timestep = 0

        initial_stats = self.synapses.get_weight_stats()
        print(f"[ENV] Seed network ready: {cfg.N_NEURONS} neurons, "
              f"{initial_stats['nnz']} synapses "
              f"(density={initial_stats['density']:.4f})")

    def generate_input_spikes(self) -> np.ndarray:
        """
        Generate Poisson-distributed spike train for input neurons.

        The input rate is modulated by a sinusoidal signal to simulate
        changing environmental conditions (tests adaptability).

        Returns
        -------
        input_current : np.ndarray, shape (N_NEURONS,)
            External input current (only first N_INPUT_NEURONS receive input).
        """
        t = self.timestep * cfg.DT  # Current time in ms

        # Sinusoidal rate modulation
        rate = cfg.INPUT_RATE_HZ + cfg.INPUT_RATE_MODULATION * np.sin(
            2 * np.pi * t / cfg.INPUT_MODULATION_PERIOD
        )
        rate = max(rate, 0.1)  # Ensure non-negative rate

        # Poisson spike probability per timestep
        # P(spike) = rate_Hz * dt_seconds = rate * dt/1000
        prob = rate * cfg.DT / 1000.0

        # Generate spikes for input neurons only
        input_spikes = np.zeros(cfg.N_NEURONS)
        input_spikes[:cfg.N_INPUT_NEURONS] = (
            self.rng.random(cfg.N_INPUT_NEURONS) < prob
        ).astype(float)

        # Scale to current magnitude
        # Needs to be large enough that accumulated input can bridge
        # the V_REST→V_THRESHOLD gap (15 mV) within a few timesteps
        input_current = input_spikes * 150.0

        return input_current

    def step(self, external_input: np.ndarray = None, reward: float = None):
        """
        Advance the entire system by one timestep.

        Parameters
        ----------
        external_input : np.ndarray, optional
            External input current. If None, auto-generated Poisson input.
        reward : float, optional
            External reward signal for reward-modulated mode.
        """
        t_start = time.perf_counter()

        # ── 1. Get input ────────────────────────────────────────────
        if external_input is not None:
            input_current = external_input
        else:
            input_current = self.generate_input_spikes()

        # ── 2. Propagate previous spikes through network ────────────
        synaptic_current = self.synapses.propagate_spikes(self.brain.spikes)

        # Add external input
        total_current = synaptic_current + input_current

        # ── 3. Liquid dynamics modulation ───────────────────────────
        liquid_mod = self.liquid.step(total_current)

        # ── 4. Update neuron states (LIF + liquid modulation) ───────
        spikes = self.brain.step(total_current, liquid_modulation=liquid_mod)

        # ── 5. Apply STDP ───────────────────────────────────────────
        self.synapses.apply_stdp(self.brain)

        # ── 6. Apply reward-modulated STDP ──────────────────────────
        if self.mode == "reward_modulated":
            if reward is not None:
                self.reward_signal = reward
            else:
                # Decay existing reward signal
                self.reward_signal *= cfg.REWARD_DECAY

            self.synapses.apply_reward_stdp(self.brain, self.reward_signal)

        # ── 7. Structural plasticity (periodic) ─────────────────────
        if self.timestep > 0 and self.timestep % cfg.STRUCTURAL_PLASTICITY_INTERVAL == 0:
            self.synapses.structural_update(self.brain)

        # ── 8. Homeostasis (periodic) ───────────────────────────────
        if self.timestep > 0 and self.timestep % cfg.HOMEOSTASIS_INTERVAL == 0:
            self.homeostasis.apply(self.brain, self.synapses)

        # ── 9. Metaplasticity (periodic) ────────────────────────────
        if self.timestep > 0 and self.timestep % cfg.META_UPDATE_INTERVAL == 0:
            self.metaplasticity.update(self.brain, self.synapses)

        # ── 10. Log metrics ─────────────────────────────────────────
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        if self.timestep % cfg.LOG_INTERVAL == 0:
            self._log_metrics(elapsed_ms)

        self.timestep += 1

    def _log_metrics(self, elapsed_ms: float):
        """Record metrics for visualization and analysis."""
        brain_state = self.brain.get_state_summary()
        weight_stats = self.synapses.get_weight_stats()
        meta_state = self.metaplasticity.get_summary()
        liquid_state = self.liquid.get_state_summary()

        self.metrics_log["timestep"].append(self.timestep)
        self.metrics_log["num_spikes"].append(brain_state["num_spiking"])
        self.metrics_log["mean_firing_rate"].append(brain_state["mean_firing_rate"])
        self.metrics_log["num_synapses"].append(weight_stats["nnz"])
        self.metrics_log["mean_weight"].append(weight_stats["mean"])
        self.metrics_log["weight_std"].append(weight_stats["std"])
        self.metrics_log["stability_cv"].append(meta_state["stability_cv"])
        self.metrics_log["liquid_mean_h"].append(liquid_state["mean_h"])
        self.metrics_log["a_plus"].append(meta_state["a_plus"])
        self.metrics_log["a_minus"].append(meta_state["a_minus"])
        self.metrics_log["elapsed_ms"].append(elapsed_ms)

    def run(self, n_timesteps: int = cfg.DEFAULT_TIMESTEPS, verbose: bool = True):
        """
        Run the simulation for a given number of timesteps.

        Parameters
        ----------
        n_timesteps : int
            Number of timesteps to simulate.
        verbose : bool
            Print progress updates.
        """
        print(f"\n[ENV] Starting simulation: {n_timesteps} timesteps, mode={self.mode}")
        print("=" * 70)

        sim_start = time.perf_counter()

        for t in range(n_timesteps):
            self.step()

            # Progress reporting
            if verbose and t > 0 and t % 1000 == 0:
                brain_state = self.brain.get_state_summary()
                weight_stats = self.synapses.get_weight_stats()
                meta_state = self.metaplasticity.get_summary()
                elapsed = time.perf_counter() - sim_start

                print(f"\n  [t={t:>6d}]  elapsed={elapsed:.1f}s  "
                      f"rate={t/elapsed:.0f} steps/s")
                print(f"    Neurons: spikes={brain_state['num_spiking']:>3d}  "
                      f"mean_rate={brain_state['mean_firing_rate']:.2f} Hz  "
                      f"mean_V={brain_state['mean_V']:.1f} mV")
                print(f"    Synapses: nnz={weight_stats['nnz']:>5d}  "
                      f"pruned={weight_stats['pruned_total']:>4d}  "
                      f"grown={weight_stats['grown_total']:>4d}  "
                      f"mean_w={weight_stats['mean']:.4f}")
                print(f"    Meta: A+={meta_state['a_plus']:.5f}  "
                      f"A-={meta_state['a_minus']:.5f}  "
                      f"CV={meta_state['stability_cv']:.3f}  "
                      f"trend={meta_state['plasticity_trend']}")

        total_time = time.perf_counter() - sim_start
        print(f"\n{'=' * 70}")
        print(f"[ENV] Simulation complete: {n_timesteps} timesteps in {total_time:.2f}s")
        print(f"[ENV] Average: {n_timesteps / total_time:.0f} timesteps/s")

        # Final state
        final_weights = self.synapses.get_weight_stats()
        final_brain = self.brain.get_state_summary()
        print(f"\n[ENV] Final state:")
        print(f"  Neurons: mean_rate={final_brain['mean_firing_rate']:.2f} Hz")
        print(f"  Synapses: {final_weights['nnz']} active "
              f"(density={final_weights['density']:.4f})")
        print(f"  Homeostasis emergencies: {self.homeostasis.emergency_events}")

        return self.metrics_log
