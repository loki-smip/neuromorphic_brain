"""
homeostasis.py — Homeostatic Scaling Controller
==================================================

Prevents neural "explosions" (runaway excitation / seizures) and
"death" (complete silence) by regulating firing rates toward a
biological target.

Biological Inspiration (Paper 2):
  ├── Excitatory/Inhibitory balance (glutamate vs GABA)
  │   The brain maintains ~80% excitatory / 20% inhibitory neuron ratio
  ├── Synaptic scaling — global adjustment of synaptic strengths
  │   When a neuron fires too much, its incoming weights are scaled DOWN
  │   When too quiet, scaled UP
  ├── GABAergic emergency inhibition
  │   Fast-spiking interneurons provide rapid global inhibition
  │   to prevent epileptic seizure-like runaway activity
  └── Neural efficiency principle (Paper 2, P-FIT)
      More intelligent systems show LESS activation during tasks

Mechanism:
  1. Track exponential moving average of each neuron's firing rate
  2. Compare to TARGET_FIRING_RATE (biological ~5 Hz for cortical neurons)
  3. Scale weights and/or thresholds to restore balance
  4. Emergency brake: if >30% neurons fire simultaneously → global inhibition
"""

import numpy as np
import config as cfg


class HomeostasisController:
    """
    Monitors and regulates network-wide firing activity.

    Two modes of operation:
      - GRADUAL: Slow synaptic scaling to nudge firing rates toward target
      - EMERGENCY: Rapid global inhibition pulse when activity is dangerous
    """

    def __init__(self, n_neurons: int = cfg.N_NEURONS):
        self.n = n_neurons

        # ── Rate tracking ───────────────────────────────────────────
        self.rate_history = []  # For visualization
        self.emergency_events = 0

    def apply(self, brain, synapse_manager):
        """
        Apply homeostatic regulation.

        Parameters
        ----------
        brain : BrainNetwork
        synapse_manager : SynapseManager
        """
        firing_rates = brain.get_firing_rates()
        mean_rate = np.mean(firing_rates)

        # ── Emergency check ─────────────────────────────────────────
        # If too many neurons fire simultaneously → GABAergic inhibition pulse
        spike_fraction = np.mean(brain.spikes.astype(float))
        if spike_fraction > cfg.EMERGENCY_THRESHOLD:
            self._emergency_inhibition(brain)
            return

        # ── Gradual homeostatic scaling ─────────────────────────────
        self._scale_weights(brain, synapse_manager, firing_rates)
        self._adjust_thresholds(brain, firing_rates)

        # Log
        self.rate_history.append(mean_rate)

    def _scale_weights(self, brain, synapse_manager, firing_rates):
        """
        Scale incoming weights for each neuron based on its firing rate.

        Neurons firing too fast → decrease incoming weights
        Neurons firing too slow → increase incoming weights

        This is biological synaptic scaling: the postsynaptic neuron
        adjusts ALL its incoming synapse strengths multiplicatively.
        """
        target = cfg.TARGET_FIRING_RATE
        w = synapse_manager.weights_lil

        for post_idx in range(self.n):
            rate = firing_rates[post_idx]
            if rate < 0.01 and target < 0.01:
                continue  # Both near zero, skip

            # Compute scaling factor
            if rate > target * 1.5:
                # Firing too fast → scale down
                scale = 1.0 - cfg.HOMEOSTASIS_GAIN
            elif rate < target * 0.5 and rate < target:
                # Firing too slow → scale up
                scale = 1.0 + cfg.HOMEOSTASIS_GAIN
            else:
                continue  # Within acceptable range

            # Apply to all incoming weights for this neuron
            col = w.getcol(post_idx)
            pre_indices = col.nonzero()[0]
            for pre_idx in pre_indices:
                w[pre_idx, post_idx] *= scale
                # Keep within bounds
                w[pre_idx, post_idx] = np.clip(
                    w[pre_idx, post_idx], cfg.W_MIN, cfg.W_MAX
                )

    def _adjust_thresholds(self, brain, firing_rates):
        """
        Adjust firing thresholds based on activity level.

        Complements weight scaling — provides faster response.
        High-firing neurons get higher thresholds (harder to fire).
        Low-firing neurons get lower thresholds (easier to fire).
        """
        target = cfg.TARGET_FIRING_RATE
        delta = cfg.HOMEOSTASIS_GAIN * 0.5  # Threshold adjustment (mV)

        too_high = firing_rates > target * 2.0
        too_low = (firing_rates < target * 0.3) & (firing_rates < target)

        brain.V_thresh[too_high] += delta
        brain.V_thresh[too_low] -= delta

        # Clamp thresholds to reasonable range
        brain.V_thresh = np.clip(brain.V_thresh, cfg.V_REST + 5.0, cfg.V_REST + 30.0)

    def _emergency_inhibition(self, brain):
        """
        GABAergic emergency brake.

        When >30% of neurons fire simultaneously (like a seizure),
        apply a strong hyperpolarizing current to ALL neurons.
        This mimics the fast-spiking inhibitory interneuron response.
        """
        brain.V += cfg.EMERGENCY_INHIBITION
        brain.V = np.maximum(brain.V, cfg.V_RESET - 10.0)  # Don't go absurdly low
        self.emergency_events += 1

    def get_summary(self) -> dict:
        """Return summary statistics."""
        return {
            "emergency_events": self.emergency_events,
            "recent_mean_rate": float(np.mean(self.rate_history[-10:]))
                if len(self.rate_history) > 0 else 0.0,
        }
