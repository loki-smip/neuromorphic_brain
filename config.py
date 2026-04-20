"""
config.py — Central Configuration for Brain-Inspired Neuromorphic System
=========================================================================

All biological and computational hyperparameters live here.
Values are derived from neuroscience literature:
  - LIF parameters from Hodgkin-Huxley simplification (Paper 2)
  - STDP windows from P-STDP real-time connectivity inference (Paper 1)
  - CfC dynamics from Liquid AI research (Paper 1, Section: CfC Breakthrough)
  - Homeostasis from cortical excitatory/inhibitory balance (Paper 2, glutamate/GABA)
  - Structural plasticity from DEEP R framework (Paper 1)
"""


# =============================================================================
# NETWORK TOPOLOGY
# =============================================================================
N_NEURONS = 256               # Seed network size (scalable)
N_INPUT_NEURONS = 32          # First N neurons receive external input
N_OUTPUT_NEURONS = 16         # Last N neurons are readout (for future tasks)
INITIAL_CONNECTIVITY = 0.05   # 5% initial random sparse connectivity
DT = 1.0                      # Timestep in milliseconds

# =============================================================================
# LIF NEURON MODEL (Leaky Integrate-and-Fire)
# Biological basis: Simplified Hodgkin-Huxley (Paper 2, Table 1)
# =============================================================================
V_REST = -70.0                # Resting membrane potential (mV)
V_THRESHOLD = -55.0           # Spike threshold (mV)
V_RESET = -75.0               # Post-spike reset potential (mV) — hyperpolarization
TAU_MEMBRANE = 10.0           # Membrane time constant (ms) — faster integration
REFRACTORY_PERIOD = 2.0       # Absolute refractory period (ms) — Na+ inactivation
THRESHOLD_ADAPT_DELTA = 0.5   # Threshold increase after each spike (mV)
THRESHOLD_ADAPT_TAU = 100.0   # Threshold recovery time constant (ms)

# =============================================================================
# STDP (Spike-Timing-Dependent Plasticity)
# Biological basis: Pre-synaptic STDP (P-STDP) from Paper 1
# LTP when pre fires before post; LTD when post fires before pre
# =============================================================================
A_PLUS = 0.01                 # LTP amplitude (potentiation strength)
A_MINUS = -0.012              # LTD amplitude (depression strength, slightly stronger)
TAU_PLUS = 20.0               # LTP time window (ms)
TAU_MINUS = 20.0              # LTD time window (ms)
W_MAX = 1.0                   # Maximum synaptic weight
W_MIN = 0.0                   # Minimum synaptic weight (excitatory only for now)

# =============================================================================
# REWARD-MODULATED STDP
# Biological basis: Dopaminergic neuromodulation, eligibility traces
# Paper 2: dopamine/serotonin oppositional balance
# =============================================================================
REWARD_DECAY = 0.99           # Reward signal exponential decay per timestep
ELIGIBILITY_TRACE_TAU = 25.0  # Eligibility trace time constant (ms)
REWARD_LEARNING_RATE = 0.005  # Modulation strength of reward on STDP

# =============================================================================
# STRUCTURAL PLASTICITY
# Biological basis: DEEP R rule (Paper 1), synaptic overproduction/pruning (Paper 1)
# =============================================================================
PRUNE_THRESHOLD = 0.01        # Synapses with |w| < this are pruned
GROWTH_RATE = 0.05            # Fraction of possible new synapses to grow per event
MAX_SYNAPSES_PER_NEURON = 50  # Fan-in/fan-out limit (biological constraint)
STRUCTURAL_PLASTICITY_INTERVAL = 100  # Timesteps between structural updates
INITIAL_WEIGHT_MEAN = 0.3     # Mean of initial random weights
INITIAL_WEIGHT_STD = 0.1      # Std dev of initial random weights

# Synaptic gain: scales weight values (0-1) to biologically meaningful current (mV)
# With ~12 synapses per neuron and weights ~0.3, total input per spike:
#   12 * 0.3 * SYNAPTIC_GAIN / TAU_MEMBRANE = target ~2-5 mV per correlated burst
SYNAPTIC_GAIN = 50.0          # Current scaling factor for synaptic weights

# =============================================================================
# LIQUID TIME-CONSTANT (LTC / CfC) DYNAMICS
# Biological basis: C. elegans nervous system, input-dependent time constants
# Paper 1: CfC closed-form approximation, 5×–179× speedup over Neural-ODE
# =============================================================================
LTC_TAU_BASE = 50.0           # Base time constant for liquid dynamics (ms)
LTC_TAU_MIN = 5.0             # Minimum tau (fast response to salient input)
LTC_TAU_MAX = 200.0           # Maximum tau (slow response for noise)
LTC_HIDDEN_DIM = 32           # Hidden state dimension per-neuron
LTC_GATE_SCALE = 1.0          # Sigmoid gate sharpness

# =============================================================================
# HOMEOSTASIS
# Biological basis: Excitatory/Inhibitory balance, GABAergic interneurons (Paper 2)
# Neural efficiency: smarter systems show LESS activation (Paper 2, P-FIT)
# =============================================================================
TARGET_FIRING_RATE = 5.0      # Target firing rate in Hz (cortical average)
HOMEOSTASIS_TAU = 1000.0      # Smoothing time constant for rate estimation (ms)
HOMEOSTASIS_INTERVAL = 50     # Timesteps between homeostatic adjustments
HOMEOSTASIS_GAIN = 0.01       # Scaling factor adjustment rate
EMERGENCY_THRESHOLD = 0.30    # If > 30% neurons fire → global inhibition pulse
EMERGENCY_INHIBITION = -10.0  # Membrane potential reduction during emergency (mV)

# =============================================================================
# METAPLASTICITY
# Biological basis: BCM theory, sliding threshold for LTP/LTD crossover
# Paper 2: BDNF-mediated plasticity modulation
# =============================================================================
META_LEARNING_RATE = 0.001    # Rate at which plasticity parameters are adjusted
STABILITY_WINDOW = 500        # Timesteps over which stability is assessed
META_UPDATE_INTERVAL = 500    # Timesteps between metaplasticity updates
STABILITY_LOW_CV = 0.3        # Below this CV → stable → reduce plasticity
STABILITY_HIGH_CV = 0.7       # Above this CV → unstable → increase plasticity
CRYSTALLIZATION_AGE = 5000    # Timesteps after which synapse becomes "crystallized"
CRYSTALLIZATION_FACTOR = 0.1  # Plasticity reduction factor for old synapses

# =============================================================================
# INPUT GENERATION (Poisson spike trains)
# =============================================================================
INPUT_RATE_HZ = 50.0          # Base Poisson firing rate for input neurons (Hz)
INPUT_RATE_MODULATION = 20.0  # Amplitude of sinusoidal rate modulation (Hz)
INPUT_MODULATION_PERIOD = 2000.0  # Period of input rate modulation (ms)

# =============================================================================
# SIMULATION
# =============================================================================
DEFAULT_TIMESTEPS = 10000     # Default simulation duration
LOG_INTERVAL = 100            # Timesteps between metric logging
VISUALIZATION_INTERVAL = 500  # Timesteps between visualization updates
