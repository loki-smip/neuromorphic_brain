"""
liquid_dynamics.py — Closed-form Continuous-Time (CfC) Neural Dynamics
========================================================================

Implements Liquid Neural Network principles as a modulation layer on top
of the LIF neuron dynamics. Each neuron gets an internal hidden state
that evolves continuously, modulating excitability.

Biological Inspiration (Paper 1, LTC Networks / C. elegans):
  - Time constant is NOT fixed — it's a function of the input
  - τ(x) = τ_base × σ(W_tau · x + b_tau)
  - Neurons "speed up" for salient inputs, "slow down" for noise
  - CfC approximation avoids expensive ODE solvers:
      h(t+dt) = σ_1 × f_1(x,h) + (1 - σ_1) × f_2(x,h)
    where σ_1 is a sigmoidal gate interpolating between two candidate states

Computational Tradeoff:
  CfC gives 5×–179× speedup over Neural-ODE (Paper 1)
  while preserving continuous-time dynamics. No ODE solver needed.

Output:
  A per-neuron modulation factor (gain) that scales synaptic input
  in the LIF update. Values near 1.0 = normal; >1 = amplify; <1 = dampen.
"""

import numpy as np
import config as cfg


class LiquidTimeConstant:
    """
    Continuous-time modulation layer using CfC (Closed-form Continuous-time).

    Each neuron has a hidden state h ∈ R that evolves based on input.
    The hidden state modulates the neuron's effective gain.
    """

    def __init__(self, n_neurons: int = cfg.N_NEURONS):
        self.n = n_neurons
        rng = np.random.default_rng(42)

        # ── Hidden state per neuron ─────────────────────────────────
        # Initialized near zero (neutral modulation)
        self.h = np.zeros(self.n)

        # ── Learnable parameters (fixed for now, could be adapted) ──
        # These define how input maps to time constant modulation
        # W_tau: input-to-tau mapping weights
        self.W_tau = rng.normal(0, 0.1, self.n)
        self.b_tau = rng.normal(0, 0.01, self.n)

        # W_h: input-to-hidden mapping (for CfC candidate computation)
        self.W_h1 = rng.normal(0, 0.1, self.n)  # Candidate 1 weights
        self.W_h2 = rng.normal(0, 0.1, self.n)  # Candidate 2 weights
        self.b_h1 = np.zeros(self.n)
        self.b_h2 = np.zeros(self.n)

        # Gate parameters
        self.W_gate = rng.normal(0, 0.1, self.n)
        self.b_gate = np.zeros(self.n)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(
            x >= 0,
            1.0 / (1.0 + np.exp(-x)),
            np.exp(x) / (1.0 + np.exp(x))
        )

    def step(self, synaptic_current: np.ndarray) -> np.ndarray:
        """
        Update hidden states and compute modulation gain.

        CfC Update Rule:
          1. Compute input-dependent time constant: τ(x) = τ_base × σ(W_tau · x + b_tau)
          2. Compute two candidate states: f1(x,h), f2(x,h)
          3. Compute gate: σ_1 = σ(W_gate · x + b_gate)
          4. Interpolate: h_new = σ_1 × f1 + (1 - σ_1) × f2
          5. Apply time constant: h = h + (h_new - h) × dt/τ
          6. Output modulation: gain = 1 + tanh(h)  (range: [0, 2])

        Parameters
        ----------
        synaptic_current : np.ndarray, shape (n,)
            Current input to each neuron.

        Returns
        -------
        modulation : np.ndarray, shape (n,)
            Per-neuron gain factor. 1.0 = no change.
        """
        x = synaptic_current
        dt = cfg.DT

        # ── 1. Input-dependent time constant ────────────────────────
        # τ(x) = τ_min + (τ_max - τ_min) × σ(W_tau · x + b_tau)
        # Salient inputs → small τ (fast response)
        # Noise → large τ (slow, filtering response)
        tau_gate = self._sigmoid(self.W_tau * x + self.b_tau)
        tau = cfg.LTC_TAU_MIN + (cfg.LTC_TAU_MAX - cfg.LTC_TAU_MIN) * tau_gate

        # ── 2. Compute candidate hidden states ──────────────────────
        # f1: "fast pathway" — more responsive to current input
        f1 = np.tanh(self.W_h1 * x + self.b_h1 + 0.5 * self.h)
        # f2: "slow pathway" — more influenced by history
        f2 = np.tanh(self.W_h2 * x + self.b_h2 + self.h)

        # ── 3. Sigmoidal gate for interpolation ─────────────────────
        gate = self._sigmoid(
            self.W_gate * x + self.b_gate + cfg.LTC_GATE_SCALE * self.h
        )

        # ── 4. CfC interpolation ───────────────────────────────────
        h_candidate = gate * f1 + (1.0 - gate) * f2

        # ── 5. Apply time constant (exponential smoothing) ──────────
        alpha = dt / tau
        alpha = np.clip(alpha, 0.0, 1.0)  # Ensure stability
        self.h = self.h + alpha * (h_candidate - self.h)

        # ── 6. Compute output modulation ────────────────────────────
        # Gain in range [0, 2]: 1.0 = neutral
        #   > 1.0: amplify (neuron is more excitable)
        #   < 1.0: dampen (neuron is less excitable)
        modulation = 1.0 + np.tanh(self.h)

        return modulation

    def get_state_summary(self) -> dict:
        """Return state summary for logging."""
        return {
            "mean_h": float(np.mean(self.h)),
            "std_h": float(np.std(self.h)),
            "mean_modulation": float(np.mean(1.0 + np.tanh(self.h))),
        }
