"""
synapse_manager.py — Sparse Connectivity, STDP, and Structural Plasticity
============================================================================

The heart of the neuromorphic system. Manages all synaptic connections
as sparse matrices for CPU-efficient computation.

Biological Inspiration (Papers 1 & 2):
  ├── Sparse connectivity graph (scipy sparse matrices)
  │   Brain connectivity is ~1-10%, not fully connected
  ├── STDP Engine (Spike-Timing-Dependent Plasticity)
  │   ├── LTP: pre fires before post → strengthen (Paper 1, P-STDP)
  │   └── LTD: post fires before pre → weaken
  ├── Reward-Modulated STDP
  │   ├── Eligibility traces combine pre/post timing
  │   └── Delayed reward modulates weight change (dopaminergic)
  ├── Structural Plasticity (DEEP R rule, Paper 1)
  │   ├── PRUNE: remove weak/inactive synapses
  │   └── GROW: create synapses between correlated active neurons
  └── Weight Normalization
      └── Maintains stable total synaptic drive per neuron

Computational Tradeoff:
  - lil_matrix (List of Lists) for structural changes → O(1) insert/delete per row
  - csr_matrix (Compressed Sparse Row) for spike propagation → O(nnz) mat-vec multiply
  - Conversion cost: O(nnz), amortized over STRUCTURAL_PLASTICITY_INTERVAL
"""

import numpy as np
from scipy import sparse
import config as cfg


class SynapseManager:
    """
    Manages all synaptic connections between neurons.

    Storage strategy:
      - self.weights_lil: lil_matrix for dynamic modification (grow/prune)
      - self.weights_csr: csr_matrix for fast spike propagation (W @ spikes)
      - Conversion happens every STRUCTURAL_PLASTICITY_INTERVAL timesteps
    """

    def __init__(self, n_neurons: int = cfg.N_NEURONS):
        self.n = n_neurons

        # ── Initialize sparse connectivity ──────────────────────────
        # Random initial connectivity at cfg.INITIAL_CONNECTIVITY density
        self.weights_lil = sparse.lil_matrix((self.n, self.n), dtype=np.float64)
        self._initialize_random_connectivity()

        # Convert to CSR for fast computation
        self.weights_csr = self.weights_lil.tocsr()

        # ── Synapse age tracking ────────────────────────────────────
        # Tracks how long each synapse has existed (for crystallization)
        self.synapse_age = sparse.lil_matrix((self.n, self.n), dtype=np.float64)

        # ── Per-synapse plasticity modulation (metaplasticity interface) ──
        self.plasticity_modulation = 1.0  # Scalar; updated by MetaplasticityController

        # ── STDP parameters (can be modulated by metaplasticity) ────
        self.a_plus = cfg.A_PLUS
        self.a_minus = cfg.A_MINUS

        # ── Statistics ──────────────────────────────────────────────
        self.total_synapses_history = []
        self.pruned_count = 0
        self.grown_count = 0

    def _initialize_random_connectivity(self):
        """
        Create initial random sparse connectivity.
        No self-connections (diagonal = 0).
        Weights drawn from clipped normal distribution.
        """
        rng = np.random.default_rng()

        # Number of potential synapses (excluding self-connections)
        n_possible = self.n * (self.n - 1)
        n_synapses = int(n_possible * cfg.INITIAL_CONNECTIVITY)

        # Generate random pre-post pairs (no self-connections)
        pairs_created = 0
        while pairs_created < n_synapses:
            batch_size = min(n_synapses - pairs_created, 10000)
            pre = rng.integers(0, self.n, size=batch_size)
            post = rng.integers(0, self.n, size=batch_size)

            # Filter out self-connections
            valid = pre != post
            pre = pre[valid]
            post = post[valid]

            # Assign weights (clipped to [W_MIN, W_MAX])
            weights = rng.normal(cfg.INITIAL_WEIGHT_MEAN, cfg.INITIAL_WEIGHT_STD, len(pre))
            weights = np.clip(weights, cfg.W_MIN + 0.001, cfg.W_MAX)

            for i in range(len(pre)):
                self.weights_lil[pre[i], post[i]] = weights[i]

            pairs_created += len(pre)

    def propagate_spikes(self, spikes: np.ndarray) -> np.ndarray:
        """
        Compute synaptic currents from spike vector.

        W_csr @ spike_vector gives total weighted input to each neuron.
        This is the most performance-critical operation — CSR format
        gives O(nnz) complexity, where nnz << N².

        Parameters
        ----------
        spikes : np.ndarray, shape (n,), dtype=bool
            Binary spike vector from BrainNetwork.

        Returns
        -------
        currents : np.ndarray, shape (n,)
            Synaptic current arriving at each neuron.
        """
        spike_float = spikes.astype(np.float64)
        # Matrix-vector multiply: columns are pre-synaptic, rows are post-synaptic
        # W[post, pre] * spike[pre] -> current arriving at post
        # SYNAPTIC_GAIN scales weight values (0-1) to biologically meaningful currents
        currents = self.weights_csr.T.dot(spike_float) * cfg.SYNAPTIC_GAIN
        return currents

    def apply_stdp(self, brain):
        """
        Apply STDP weight updates based on current spike traces.

        STDP rule (from Paper 1, P-STDP):
          - When a POST-synaptic neuron fires:
              Δw = A+ × pre_trace[pre]  (LTP — potentiation)
          - When a PRE-synaptic neuron fires:
              Δw = A- × post_trace[post]  (LTD — depression)

        Parameters
        ----------
        brain : BrainNetwork
            The neuron population (provides traces and spike info).
        """
        spike_indices = np.where(brain.spikes)[0]
        if len(spike_indices) == 0:
            return

        # Work with LIL for efficient row/column access
        w = self.weights_lil

        for post_idx in spike_indices:
            # ── LTP: post fired → strengthen synapses from active pre neurons
            # Find all pre-synaptic neurons connected TO this post neuron
            # In our convention: w[pre, post] stores the weight
            col = w.getcol(post_idx)
            pre_indices = col.nonzero()[0]

            if len(pre_indices) > 0:
                # LTP: Δw = A+ × pre_trace[pre] × plasticity_modulation
                delta_w = self.a_plus * brain.pre_trace[pre_indices] * self.plasticity_modulation
                for i, pre_idx in enumerate(pre_indices):
                    new_w = w[pre_idx, post_idx] + delta_w[i]
                    w[pre_idx, post_idx] = np.clip(new_w, cfg.W_MIN, cfg.W_MAX)

        for pre_idx in spike_indices:
            # ── LTD: pre fired → weaken synapses TO post neurons with recent activity
            row = w.getrow(pre_idx)
            post_indices = row.nonzero()[1]

            if len(post_indices) > 0:
                # LTD: Δw = A- × post_trace[post] × plasticity_modulation
                delta_w = self.a_minus * brain.post_trace[post_indices] * self.plasticity_modulation
                for i, post_idx in enumerate(post_indices):
                    new_w = w[pre_idx, post_idx] + delta_w[i]
                    w[pre_idx, post_idx] = np.clip(new_w, cfg.W_MIN, cfg.W_MAX)

        # Update eligibility traces for reward-modulated STDP
        # Eligibility = running average of STDP-like correlations
        for idx in spike_indices:
            brain.eligibility_trace[idx] += (
                brain.pre_trace[idx] * self.a_plus +
                brain.post_trace[idx] * abs(self.a_minus)
            )

    def apply_reward_stdp(self, brain, reward_signal: float):
        """
        Apply reward-modulated STDP.

        Weight change = reward_signal × eligibility_trace
        This implements temporal credit assignment: the eligibility trace
        "remembers" which synapses were recently active, and the delayed
        reward decides whether to strengthen or weaken them.

        Biological basis: Dopaminergic modulation of synaptic plasticity (Paper 2)

        Parameters
        ----------
        brain : BrainNetwork
        reward_signal : float
            Positive = reward, negative = punishment, 0 = neutral.
        """
        if abs(reward_signal) < 1e-8:
            return

        w = self.weights_lil

        # For each neuron with nonzero eligibility, modulate its outgoing weights
        active_neurons = np.where(np.abs(brain.eligibility_trace) > 1e-6)[0]

        for pre_idx in active_neurons:
            row = w.getrow(pre_idx)
            post_indices = row.nonzero()[1]

            if len(post_indices) > 0:
                delta_w = (
                    cfg.REWARD_LEARNING_RATE *
                    reward_signal *
                    brain.eligibility_trace[pre_idx]
                )
                for post_idx in post_indices:
                    new_w = w[pre_idx, post_idx] + delta_w
                    w[pre_idx, post_idx] = np.clip(new_w, cfg.W_MIN, cfg.W_MAX)

    def structural_update(self, brain):
        """
        Perform structural plasticity: prune weak synapses, grow new ones.

        PRUNE (Paper 1, ETSM — removes up to 99% connections):
          - Remove synapses where |w| < PRUNE_THRESHOLD
          - These represent "forgotten" or unused connections

        GROW (Paper 1, DEEP R):
          - Create synapses between neurons that are both frequently active
          - Respects MAX_SYNAPSES_PER_NEURON constraint
          - New weights initialized at INITIAL_WEIGHT_MEAN

        Parameters
        ----------
        brain : BrainNetwork
        """
        self._prune_synapses()
        self._grow_synapses(brain)
        self._age_synapses()

        # Rebuild CSR after structural changes
        self.weights_csr = self.weights_lil.tocsr()

        # Record statistics
        self.total_synapses_history.append(self.weights_csr.nnz)

    def _prune_synapses(self):
        """Remove synapses with weight below threshold."""
        w = self.weights_lil
        count = 0

        for i in range(self.n):
            row = w.getrow(i)
            cols = row.nonzero()[1]
            for j in cols:
                if abs(w[i, j]) < cfg.PRUNE_THRESHOLD:
                    w[i, j] = 0.0
                    self.synapse_age[i, j] = 0.0
                    count += 1

        # Eliminate stored zeros
        w.tocsr()
        self.weights_lil = w.tolil()
        self.pruned_count += count

    def _grow_synapses(self, brain):
        """Grow new synapses between correlated active neurons."""
        firing_rates = brain.get_firing_rates()

        # Find neurons with above-average firing rate (active neurons)
        mean_rate = np.mean(firing_rates)
        if mean_rate < 0.1:
            return  # Network too quiet, nothing to correlate

        active_neurons = np.where(firing_rates > mean_rate)[0]
        if len(active_neurons) < 2:
            return

        rng = np.random.default_rng()
        w = self.weights_lil
        count = 0

        # Number of new synapses to attempt
        n_attempts = max(1, int(len(active_neurons) * cfg.GROWTH_RATE))

        for _ in range(n_attempts):
            # Pick two distinct active neurons
            pair = rng.choice(active_neurons, size=2, replace=False)
            pre, post = int(pair[0]), int(pair[1])

            # Skip if synapse already exists or self-connection
            if pre == post or w[pre, post] != 0.0:
                continue

            # Check fan-out limit for pre neuron
            if w.getrow(pre).nnz >= cfg.MAX_SYNAPSES_PER_NEURON:
                continue

            # Check fan-in limit for post neuron
            if w.getcol(post).nnz >= cfg.MAX_SYNAPSES_PER_NEURON:
                continue

            # Create new synapse with initial weight
            new_weight = rng.normal(cfg.INITIAL_WEIGHT_MEAN, cfg.INITIAL_WEIGHT_STD)
            new_weight = np.clip(new_weight, cfg.W_MIN + 0.001, cfg.W_MAX)
            w[pre, post] = new_weight
            count += 1

        self.grown_count += count

    def _age_synapses(self):
        """Increment age of all existing synapses by STRUCTURAL_PLASTICITY_INTERVAL."""
        # Only age synapses that exist
        w_coo = self.weights_lil.tocoo()
        for i, j in zip(w_coo.row, w_coo.col):
            self.synapse_age[i, j] += cfg.STRUCTURAL_PLASTICITY_INTERVAL

    def normalize_weights(self):
        """
        Normalize total synaptic input per neuron to maintain stable drive.

        For each post-synaptic neuron, scale all incoming weights so their
        sum equals a target value. This prevents runaway excitation from
        STDP-driven weight increases.

        Biological basis: Synaptic scaling (Paper 2, homeostatic mechanisms)
        """
        w = self.weights_lil
        target_sum = cfg.INITIAL_WEIGHT_MEAN * cfg.MAX_SYNAPSES_PER_NEURON * 0.5

        for post_idx in range(self.n):
            col = w.getcol(post_idx)
            col_sum = col.sum()
            if col_sum > 1e-8:
                scale = target_sum / col_sum
                # Don't scale too aggressively
                scale = np.clip(scale, 0.5, 2.0)
                pre_indices = col.nonzero()[0]
                for pre_idx in pre_indices:
                    w[pre_idx, post_idx] *= scale

        self.weights_csr = self.weights_lil.tocsr()

    def get_weight_stats(self) -> dict:
        """Return weight distribution statistics."""
        data = self.weights_csr.data
        if len(data) == 0:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "nnz": 0}
        return {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "nnz": int(self.weights_csr.nnz),
            "density": float(self.weights_csr.nnz / (self.n * self.n)),
            "pruned_total": self.pruned_count,
            "grown_total": self.grown_count,
        }

    def get_connectivity_matrix(self) -> sparse.csr_matrix:
        """Return the current weight matrix (CSR format) for visualization."""
        return self.weights_csr.copy()
