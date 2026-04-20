"""
metaplasticity.py — Adaptive Learning Rate Controller
========================================================

The "learning about learning" layer. Adjusts plasticity parameters
based on environmental stability, implementing the BCM (Bienenstock-
Cooper-Munro) sliding threshold theory.

Biological Inspiration (Paper 2):
  ├── BCM Theory: The crossover point between LTP and LTD shifts
  │   based on the neuron's recent activity history
  ├── BDNF-mediated plasticity modulation
  │   Brain-Derived Neurotrophic Factor regulates synapse strength
  ├── Memory consolidation: short-term → long-term
  │   Synapses that remain stable become "crystallized" over time
  │   (reduced plasticity) — prevents catastrophic forgetting
  └── Stability-dependent plasticity:
      Stable environment → decrease learning rates (consolidate)
      Unstable environment → increase learning rates (explore)

Mechanism:
  1. Track firing rate variance over a stability window (500 timesteps)
  2. Compute stability index: CV = σ / μ (coefficient of variation)
  3. Stable (low CV) → reduce A+, A-, increase prune threshold
  4. Unstable (high CV) → increase A+, A-, decrease prune threshold
  5. Old synapses → reduced plasticity (crystallization)
"""

import numpy as np
from collections import deque
import config as cfg


class MetaplasticityController:
    """
    Dynamically adjusts learning parameters based on network stability.

    Tracks the coefficient of variation (CV) of firing rates over a
    sliding window to determine environmental stability.
    """

    def __init__(self, n_neurons: int = cfg.N_NEURONS):
        self.n = n_neurons

        # ── Stability tracking ──────────────────────────────────────
        # Store recent firing rate snapshots for CV computation
        self.rate_history = deque(maxlen=cfg.STABILITY_WINDOW)

        # ── Current plasticity parameters ───────────────────────────
        self.current_a_plus = cfg.A_PLUS
        self.current_a_minus = cfg.A_MINUS
        self.current_prune_threshold = cfg.PRUNE_THRESHOLD

        # ── Stability metrics log ───────────────────────────────────
        self.stability_log = []

    def update(self, brain, synapse_manager):
        """
        Assess stability and adjust plasticity parameters.

        Parameters
        ----------
        brain : BrainNetwork
        synapse_manager : SynapseManager
        """
        # Record current mean firing rate
        mean_rate = np.mean(brain.get_firing_rates())
        self.rate_history.append(mean_rate)

        if len(self.rate_history) < 10:
            return  # Not enough history to assess stability

        # ── Compute stability index (Coefficient of Variation) ──────
        rates = np.array(self.rate_history)
        mu = np.mean(rates)
        sigma = np.std(rates)

        if mu < 1e-8:
            cv = 0.0  # Network is silent, treat as "stable" edge case
        else:
            cv = sigma / mu

        self.stability_log.append(cv)

        # ── Adjust plasticity based on stability ────────────────────
        if cv < cfg.STABILITY_LOW_CV:
            # ── STABLE environment ──────────────────────────────────
            # Reduce learning rates → consolidate existing knowledge
            # Increase prune threshold → trim weak, keep strong
            self.current_a_plus *= (1.0 - cfg.META_LEARNING_RATE)
            self.current_a_minus *= (1.0 - cfg.META_LEARNING_RATE)
            self.current_prune_threshold *= (1.0 + cfg.META_LEARNING_RATE)

        elif cv > cfg.STABILITY_HIGH_CV:
            # ── UNSTABLE environment ────────────────────────────────
            # Increase learning rates → explore new connections
            # Decrease prune threshold → preserve more synapses
            self.current_a_plus *= (1.0 + cfg.META_LEARNING_RATE)
            self.current_a_minus *= (1.0 + cfg.META_LEARNING_RATE)
            self.current_prune_threshold *= (1.0 - cfg.META_LEARNING_RATE)

        # ── Clamp to reasonable bounds ──────────────────────────────
        self.current_a_plus = np.clip(self.current_a_plus, cfg.A_PLUS * 0.1, cfg.A_PLUS * 10.0)
        self.current_a_minus = np.clip(self.current_a_minus, cfg.A_MINUS * 10.0, cfg.A_MINUS * 0.1)
        self.current_prune_threshold = np.clip(
            self.current_prune_threshold, cfg.PRUNE_THRESHOLD * 0.1, cfg.PRUNE_THRESHOLD * 10.0
        )

        # ── Push updated parameters to synapse manager ──────────────
        synapse_manager.a_plus = self.current_a_plus
        synapse_manager.a_minus = self.current_a_minus

        # ── Apply synapse crystallization ───────────────────────────
        self._apply_crystallization(synapse_manager)

    def _apply_crystallization(self, synapse_manager):
        """
        Reduce plasticity for old, stable synapses.

        Synapses that have existed for > CRYSTALLIZATION_AGE timesteps
        get their plasticity modulation reduced. This models the
        biological transition from short-term to long-term memory:
        recently formed synapses are flexible, old ones are "hardwired".
        """
        age_coo = synapse_manager.synapse_age.tocoo()
        if age_coo.nnz == 0:
            return

        ages = age_coo.data
        old_synapse_fraction = np.mean(ages > cfg.CRYSTALLIZATION_AGE)

        # Overall plasticity modulation: weighted by how many synapses are old
        # More old synapses → lower global plasticity
        base_modulation = 1.0
        crystallized_modulation = cfg.CRYSTALLIZATION_FACTOR

        synapse_manager.plasticity_modulation = (
            base_modulation * (1.0 - old_synapse_fraction) +
            crystallized_modulation * old_synapse_fraction
        )

    def get_summary(self) -> dict:
        """Return summary for logging/visualization."""
        return {
            "a_plus": float(self.current_a_plus),
            "a_minus": float(self.current_a_minus),
            "prune_threshold": float(self.current_prune_threshold),
            "stability_cv": float(self.stability_log[-1]) if self.stability_log else 0.0,
            "plasticity_trend": "stable" if len(self.stability_log) > 0 and
                self.stability_log[-1] < cfg.STABILITY_LOW_CV else
                "exploring" if len(self.stability_log) > 0 and
                self.stability_log[-1] > cfg.STABILITY_HIGH_CV else "normal",
        }
