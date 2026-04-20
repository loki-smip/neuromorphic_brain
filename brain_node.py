"""
brain_node.py — Vectorized LIF Neuron Network
===============================================

Implements a population of Leaky Integrate-and-Fire (LIF) neurons as
parallel NumPy arrays (NOT individual objects) for CPU performance.

Biological Inspiration (Papers 1 & 2):
  ├── Membrane potential with exponential leak (τ_membrane decay)
  │   Models the RC circuit of the neuronal membrane (Hodgkin-Huxley simplified)
  ├── Spike generation when V > V_threshold (all-or-nothing action potential)
  ├── Absolute refractory period (2ms) — models Na+ channel inactivation
  ├── Adaptive threshold — increases after spiking, decays back
  │   Prevents pathological bursting; models slow Na+ inactivation
  └── Local plasticity state — last spike time, firing rate, eligibility trace

Computational Tradeoff:
  Using vectorized arrays instead of per-neuron objects gives ~100× speedup
  on CPU. All neurons update in a single NumPy operation per timestep.
"""

import numpy as np
import config as cfg


class BrainNetwork:
    """
    A population of LIF neurons stored as parallel arrays.

    All state variables are NumPy arrays of shape (N_NEURONS,).
    This enables fully vectorized updates — no Python loops over neurons.
    """

    def __init__(self, n_neurons: int = cfg.N_NEURONS):
        self.n = n_neurons

        # ── Membrane potential (mV) ──────────────────────────────────
        # Initialize at resting potential with small random perturbation
        # to break symmetry (prevents all neurons firing simultaneously)
        self.V = np.full(self.n, cfg.V_REST) + np.random.uniform(-2.0, 2.0, self.n)

        # ── Adaptive threshold (mV) ─────────────────────────────────
        # Starts at base threshold; rises after each spike, decays back
        self.V_thresh = np.full(self.n, cfg.V_THRESHOLD)

        # ── Refractory state (ms remaining) ─────────────────────────
        # When > 0, neuron cannot fire (absolute refractory period)
        self.refractory_timer = np.zeros(self.n)

        # ── Spike output (binary) ───────────────────────────────────
        self.spikes = np.zeros(self.n, dtype=bool)

        # ── Plasticity tracking ─────────────────────────────────────
        self.last_spike_time = np.full(self.n, -np.inf)  # Last spike timestamp
        self.firing_rate_estimate = np.zeros(self.n)       # Exponential moving avg

        # ── STDP trace variables ────────────────────────────────────
        # Pre-synaptic trace: increased when neuron fires, decays exponentially
        self.pre_trace = np.zeros(self.n)
        # Post-synaptic trace: same mechanism
        self.post_trace = np.zeros(self.n)

        # ── Eligibility traces for reward-modulated STDP ────────────
        # Combines pre/post timing; modulated by delayed reward signal
        self.eligibility_trace = np.zeros(self.n)

        # ── Timestep counter ────────────────────────────────────────
        self.current_time = 0.0

    def step(self, synaptic_current: np.ndarray, liquid_modulation: np.ndarray = None):
        """
        Advance all neurons by one timestep (DT milliseconds).

        Parameters
        ----------
        synaptic_current : np.ndarray, shape (n,)
            Total input current to each neuron from synaptic connections.
        liquid_modulation : np.ndarray, shape (n,), optional
            Multiplicative gain from LiquidTimeConstant module.
            Modulates neuron excitability (1.0 = no change).

        Returns
        -------
        spikes : np.ndarray, shape (n,), dtype=bool
            Which neurons fired this timestep.
        """
        dt = cfg.DT
        self.current_time += dt

        # ── 1. Decay refractory timers ──────────────────────────────
        self.refractory_timer = np.maximum(0.0, self.refractory_timer - dt)

        # Mask: neurons that are NOT in refractory period
        active_mask = self.refractory_timer <= 0.0

        # ── 2. LIF membrane potential update ────────────────────────
        # dV/dt = -(V - V_rest) / τ_membrane + I_syn / τ_membrane
        # Euler integration: V(t+dt) = V(t) + dV * dt
        #
        # Biological interpretation:
        #   - The (V_REST - V) term is the passive leak through ion channels
        #   - synaptic_current represents post-synaptic potentials (PSPs)
        leak = (cfg.V_REST - self.V) / cfg.TAU_MEMBRANE
        input_term = synaptic_current / cfg.TAU_MEMBRANE

        # Apply liquid modulation to gain (if available)
        if liquid_modulation is not None:
            input_term *= liquid_modulation

        dV = (leak + input_term) * dt

        # Only update non-refractory neurons
        self.V += dV * active_mask

        # ── 3. Spike detection ──────────────────────────────────────
        # All-or-nothing: if V exceeds threshold AND neuron is not refractory
        self.spikes = (self.V >= self.V_thresh) & active_mask

        # ── 4. Post-spike processing ────────────────────────────────
        spike_indices = np.where(self.spikes)[0]

        if len(spike_indices) > 0:
            # Reset membrane potential (hyperpolarization, like K+ channels opening)
            self.V[spike_indices] = cfg.V_RESET

            # Engage refractory period (Na+ channel inactivation)
            self.refractory_timer[spike_indices] = cfg.REFRACTORY_PERIOD

            # Adapt threshold upward (prevents bursting)
            self.V_thresh[spike_indices] += cfg.THRESHOLD_ADAPT_DELTA

            # Record spike time
            self.last_spike_time[spike_indices] = self.current_time

        # ── 5. Threshold decay back to baseline ─────────────────────
        # Exponential decay: mimics slow recovery of Na+ channel availability
        thresh_decay = (cfg.V_THRESHOLD - self.V_thresh) / cfg.THRESHOLD_ADAPT_TAU * dt
        self.V_thresh += thresh_decay

        # ── 6. Update STDP traces ───────────────────────────────────
        # Traces decay exponentially; spiking neurons get a boost
        self.pre_trace *= np.exp(-dt / cfg.TAU_PLUS)
        self.post_trace *= np.exp(-dt / cfg.TAU_MINUS)

        if len(spike_indices) > 0:
            self.pre_trace[spike_indices] += cfg.A_PLUS
            self.post_trace[spike_indices] += np.abs(cfg.A_MINUS)

        # ── 7. Update eligibility traces ────────────────────────────
        # Decays toward zero; updated by STDP engine based on pre/post correlation
        self.eligibility_trace *= np.exp(-dt / cfg.ELIGIBILITY_TRACE_TAU)

        # ── 8. Update firing rate estimate ──────────────────────────
        # Exponential moving average: rate_est += (spike - rate_est) / τ
        # Converts spike count to Hz: spike per ms → multiply by 1000
        spike_float = self.spikes.astype(float)
        alpha = dt / cfg.HOMEOSTASIS_TAU
        self.firing_rate_estimate += alpha * (
            spike_float * (1000.0 / dt) - self.firing_rate_estimate
        )

        return self.spikes

    def get_firing_rates(self) -> np.ndarray:
        """Return current firing rate estimates in Hz."""
        return self.firing_rate_estimate.copy()

    def get_state_summary(self) -> dict:
        """Return a summary dict for logging/visualization."""
        return {
            "mean_V": float(np.mean(self.V)),
            "std_V": float(np.std(self.V)),
            "mean_firing_rate": float(np.mean(self.firing_rate_estimate)),
            "max_firing_rate": float(np.max(self.firing_rate_estimate)),
            "num_spiking": int(np.sum(self.spikes)),
            "mean_threshold": float(np.mean(self.V_thresh)),
            "time": float(self.current_time),
        }
