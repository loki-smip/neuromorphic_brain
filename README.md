# Brain-Inspired Self-Evolving Neuromorphic AI

> A CPU-efficient, self-organizing Spiking Neural Network with Liquid Neural Network dynamics, structural plasticity, and biologically-inspired learning rules. **No Transformers. No Backpropagation. No GPU required.**

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Theoretical Background](#theoretical-background)
  - [Spiking Neural Networks (SNNs)](#1-spiking-neural-networks-snns)
  - [Liquid Neural Networks (LNNs)](#2-liquid-neural-networks-lnns)
  - [STDP Learning](#3-spike-timing-dependent-plasticity-stdp)
  - [Structural Plasticity](#4-structural-plasticity)
  - [Homeostatic Regulation](#5-homeostatic-regulation)
  - [Metaplasticity](#6-metaplasticity--bcm-theory)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Module Documentation](#module-documentation)
  - [config.py](#configpy)
  - [brain_node.py](#brain_nodepy)
  - [synapse_manager.py](#synapse_managerpy)
  - [liquid_dynamics.py](#liquid_dynamicspy)
  - [homeostasis.py](#homeostasispy)
  - [metaplasticity.py](#metaplasticitypy)
  - [environment.py](#environmentpy)
  - [visualization.py](#visualizationpy)
  - [main.py](#mainpy)
- [Installation](#installation)
- [Usage](#usage)
- [Simulation Results](#simulation-results)
- [Performance](#performance)
- [Research Papers & References](#research-papers--references)
- [License](#license)

---

## Overview

This project implements a **self-evolving, brain-inspired AI model** that runs entirely on CPU. It is built from first principles of computational neuroscience, drawing from two comprehensive research analyses:

1. **"Architectures of Synthetic Cognition"** — covering Spiking Neural Networks, Liquid Neural Networks, and real-time structural evolution
2. **"The Neurobiological Architecture of Human Cognition"** — covering neural mechanics, structural connectivity, plasticity, and the biological basis of intelligence

Unlike conventional deep learning systems that use dense matrix multiplications on GPUs, this model operates through **sparse, event-driven, asynchronous** computation — mirroring how the biological brain achieves complex cognition on only ~20 watts of power.

The system features:
- A population of **256 Leaky Integrate-and-Fire (LIF) neurons** communicating via discrete spikes
- **~3,200 sparse synaptic connections** (5% density) stored as scipy sparse matrices
- **Six interacting biological mechanisms** that enable lifelong, unsupervised learning

---

## Key Features

| Feature | Implementation | Biological Basis |
|---------|---------------|-----------------|
| Spiking Neural Network | Vectorized LIF neurons (NumPy arrays) | Action potentials; Hodgkin-Huxley simplified |
| Liquid Dynamics | Closed-form Continuous-time (CfC) modulation | *C. elegans* nervous system; input-dependent time constants |
| STDP Learning | Pre/post spike-timing correlation | Long-Term Potentiation/Depression; Hebbian learning |
| Reward-Modulated STDP | Eligibility traces + delayed reward | Dopaminergic modulation of synaptic plasticity |
| Structural Plasticity | Dynamic synapse growth and pruning | Synaptic overproduction and developmental pruning |
| Homeostasis | Synaptic scaling + threshold adjustment + emergency brake | Excitatory/Inhibitory balance; GABAergic interneurons |
| Metaplasticity | BCM sliding threshold; crystallization | BDNF-mediated plasticity modulation; memory consolidation |
| CPU Optimization | Sparse matrices (CSR/LIL); vectorized operations | Computational parsimony; event-driven processing |

---

## Theoretical Background

### 1. Spiking Neural Networks (SNNs)

The biological brain processes information through **discrete, all-or-nothing electrical pulses** called action potentials (spikes). This is fundamentally different from the continuous-valued activations in standard artificial neural networks.

The **Leaky Integrate-and-Fire (LIF)** model describes how each neuron integrates incoming signals:

```
dV/dt = -(V - V_rest) / tau_membrane + I_syn / tau_membrane
```

When the membrane potential `V` exceeds a threshold `V_thresh`, the neuron emits a spike, resets to `V_reset`, and enters a refractory period during which it cannot fire again. This models the Na+ channel inactivation observed in real neurons.

**Why LIF over Hodgkin-Huxley?** The LIF model provides the optimal balance of biological plausibility and computational tractability for large-scale simulations. While Hodgkin-Huxley provides extreme biophysical accuracy, its computational cost makes it impractical for networks of hundreds of neurons running in real-time on CPU.

> *Reference: The evolution of spiking neuron models and the "performance gap" in handling long-range dependencies is discussed in Research Paper 1, Section "Biological Foundations and the Mechanics of Spiking Computation".*

### 2. Liquid Neural Networks (LNNs)

Developed by researchers at MIT, inspired by the compact nervous system of the nematode *C. elegans*, Liquid Neural Networks use **input-dependent time constants** that allow neurons to adapt their processing speed to the saliency of incoming data.

The **Liquid Time-Constant (LTC)** is mathematically modeled as:

```
tau(x) = tau_min + (tau_max - tau_min) * sigmoid(W_tau * x + b_tau)
```

This means the time constant is not fixed — it is a **function of the input**. Salient inputs produce small tau values (fast response), while noise produces large tau values (slow, filtering response).

The key breakthrough is the **Closed-form Continuous-time (CfC)** approximation, which replaces expensive ODE solvers with a closed-form interpolation:

```
h_new = gate * f_fast(x, h) + (1 - gate) * f_slow(x, h)
h = h + (dt / tau) * (h_new - h)
```

This achieves **5x-179x speedup** over Neural-ODE methods while preserving continuous-time dynamics.

> *Reference: Research Paper 1, Section "Liquid Neural Networks: Continuous-Time Dynamics and Real-Time Adaptation" and "The Closed-Form Breakthrough (CfC)".*

### 3. Spike-Timing-Dependent Plasticity (STDP)

STDP is the primary learning rule in this system, replacing backpropagation entirely. It is a **local, biologically plausible** learning rule based on the precise timing of pre- and post-synaptic spikes:

- **LTP (Long-Term Potentiation):** If a presynaptic neuron fires **before** the postsynaptic neuron, the synapse is **strengthened** (the pre neuron "caused" the post to fire).
- **LTD (Long-Term Depression):** If a presynaptic neuron fires **after** the postsynaptic neuron, the synapse is **weakened** (the pre neuron was "too late").

```
When post fires:  dw = +A_plus  * pre_trace[pre]   (LTP)
When pre fires:   dw = +A_minus * post_trace[post]  (LTD, A_minus < 0)
```

**Reward-Modulated STDP** extends this with eligibility traces that "remember" which synapses were recently co-active, allowing a delayed reward signal (analogous to dopamine) to decide whether to consolidate or erase the learned pattern.

> *Reference: Research Paper 1, Section "Synaptic Plasticity and Online Connectivity Inference" — covering P-STDP and sign-switching synapses. Research Paper 2, Section "Neuroplasticity: The Mechanism of Change and Memory" — covering LTP, LTD, and BDNF.*

### 4. Structural Plasticity

The human brain undergoes a protracted phase of **synaptic overproduction and subsequent pruning** during development. This system implements the same principle:

- **PRUNE:** Remove synapses where the absolute weight falls below a threshold (analogous to the ETSM method that removes up to 99% of connections while maintaining accuracy)
- **GROW:** Create new synapses between neurons that are both frequently active and not yet connected (analogous to the DEEP R structural plasticity rule)

This allows the network topology to **evolve in real-time**, discovering optimal connectivity patterns through experience rather than human design.

> *Reference: Research Paper 1, Section "Structural Plasticity and Adaptive Growth" — covering ETSM and DEEP R. Research Paper 2, Section on developmental plasticity and Probst bundle formation in Agenesis of the Corpus Callosum.*

### 5. Homeostatic Regulation

Without homeostasis, STDP-driven networks are prone to **runaway excitation** (seizure-like activity) or **complete silence** (neural death). The brain maintains stability through:

- **Synaptic Scaling:** A postsynaptic neuron adjusts ALL its incoming synapse strengths multiplicatively based on its firing rate
- **Threshold Adjustment:** Neurons that fire too often get higher thresholds; quiet neurons get lower thresholds
- **GABAergic Emergency Inhibition:** Fast-spiking inhibitory interneurons provide rapid global inhibition when >30% of neurons fire simultaneously

The brain maintains approximately 80% excitatory / 20% inhibitory neuron ratio (glutamate vs GABA balance).

> *Reference: Research Paper 2, Section "The Fundamental Unit" — covering excitatory/inhibitory balance (glutamate vs GABA). The P-FIT model shows that more intelligent systems show LESS activation during tasks (neural efficiency principle).*

### 6. Metaplasticity / BCM Theory

The **Bienenstock-Cooper-Munro (BCM) theory** describes how the crossover point between LTP and LTD shifts based on the neuron's recent activity history. This is "learning about learning":

- **Stable environment** (low coefficient of variation in firing rates) → **decrease** learning rates, **increase** prune threshold → consolidate existing knowledge
- **Unstable environment** (high CV) → **increase** learning rates, **decrease** prune threshold → explore new connections

Additionally, **synapse crystallization** reduces plasticity for old, stable synapses — modeling the biological transition from short-term to long-term memory and preventing catastrophic forgetting.

> *Reference: Research Paper 2, Section "Neuroplasticity" — covering BDNF-mediated plasticity modulation. Research Paper 1, Section "Self-Evolving Embodied AI" — covering memory self-updating and model self-evolution.*

---

## System Architecture

```
                    +------------------+
                    |   Input Layer    |
                    |  32 Poisson      |
                    |  Spike Generators|
                    +--------+---------+
                             |
                             v
                    +------------------+
                    | Sparse Weight    |
                    | Matrix (CSR)     |    <--- STDP updates weights
                    | ~3200 synapses   |    <--- Structural Plasticity grows/prunes
                    | 5% density       |    <--- Metaplasticity modulates learning rate
                    +--------+---------+
                             |
                             v
                    +------------------+
                    | LIF Neuron       |
                    | Population       |    <--- Homeostasis regulates firing rates
                    | 256 neurons      |    <--- Adaptive thresholds
                    | Vectorized NumPy |    <--- Refractory periods
                    +--------+---------+
                             |
                    +--------+---------+
                    | Liquid Dynamics  |
                    | (CfC Module)     |    <--- Input-dependent time constants
                    | Gain modulation  |    <--- Fast/slow pathway interpolation
                    +--------+---------+
                             |
                             v
                    +------------------+
                    |  Output Layer    |
                    |  16 readout      |
                    |  neurons         |
                    +------------------+

    Execution Flow (per timestep):
    1. Generate Poisson input spikes
    2. Propagate spikes: W_csr @ spike_vector (O(nnz) sparse mat-vec)
    3. Liquid dynamics: compute CfC gain modulation
    4. LIF update: dV/dt with liquid modulation
    5. Spike detection: V >= V_thresh
    6. STDP: update weights based on spike timing
    7. [Every 100 steps] Structural plasticity: prune + grow
    8. [Every  50 steps] Homeostasis: synaptic scaling + threshold adjust
    9. [Every 500 steps] Metaplasticity: adjust learning rates
```

---

## Project Structure

```
neuromorphic_brain/
|-- config.py              # All hyperparameters (biologically derived)
|-- brain_node.py          # Vectorized LIF neuron population
|-- synapse_manager.py     # Sparse STDP + structural plasticity engine
|-- liquid_dynamics.py     # CfC continuous-time modulation layer
|-- homeostasis.py         # Firing rate regulation + emergency brake
|-- metaplasticity.py      # BCM learning rate adaptation + crystallization
|-- environment.py         # Main simulation loop orchestrator
|-- visualization.py       # 4-panel dark-theme evolution dashboard
|-- main.py                # CLI entry point
|-- requirements.txt       # Dependencies (numpy, scipy, matplotlib)
|-- output/                # Generated visualization dashboards
|   |-- evolution_dashboard.png
|   +-- metaplasticity_panel.png
+-- README.md              # This file
```

---

## Module Documentation

### `config.py`

Central configuration file containing **all hyperparameters** organized by biological subsystem. Values are derived from neuroscience literature and the two reference research papers.

**Key Parameter Groups:**

| Group | Parameters | Source |
|-------|-----------|--------|
| **Neuron (LIF)** | V_REST=-70mV, V_THRESHOLD=-55mV, TAU_MEMBRANE=10ms | Hodgkin-Huxley simplified |
| **Connectivity** | N_NEURONS=256, INITIAL_CONNECTIVITY=5%, W_MAX=1.0 | Brain connectivity ~1-10% |
| **STDP** | A_PLUS=0.01, A_MINUS=-0.012, TAU_PLUS/MINUS=20ms | P-STDP (Paper 1) |
| **Liquid (CfC)** | TAU_MIN=1ms, TAU_MAX=100ms | C. elegans dynamics |
| **Homeostasis** | TARGET_RATE=5Hz, EMERGENCY_THRESHOLD=30% | Cortical neuron firing rates |
| **Metaplasticity** | STABILITY_WINDOW=500, CRYSTALLIZATION_AGE=5000 | BCM theory |

### `brain_node.py`

Implements a population of **Leaky Integrate-and-Fire neurons** as parallel NumPy arrays. All 256 neurons update in a single vectorized operation per timestep — no Python loops over individual neurons.

**Key Features:**
- Membrane potential with exponential leak
- Adaptive threshold (prevents pathological bursting)
- Absolute refractory period (2ms Na+ channel inactivation)
- Pre/post synaptic traces for STDP
- Eligibility traces for reward-modulated STDP
- Exponential moving average firing rate estimation

### `synapse_manager.py`

The heart of the system. Manages all synaptic connections using a **dual sparse matrix strategy**:

- `lil_matrix` (List of Lists) — for structural changes: O(1) insert/delete per row
- `csr_matrix` (Compressed Sparse Row) — for spike propagation: O(nnz) mat-vec multiply

**Key Operations:**
- `propagate_spikes()` — CSR mat-vec multiply with SYNAPTIC_GAIN scaling
- `apply_stdp()` — LTP/LTD based on spike timing traces
- `apply_reward_stdp()` — Eligibility trace modulated by delayed reward
- `structural_update()` — Prune weak synapses, grow correlated ones
- `normalize_weights()` — Synaptic scaling for stable drive

### `liquid_dynamics.py`

Implements the **Closed-form Continuous-time (CfC)** neural dynamics as a modulation layer. Each neuron has a hidden state that evolves continuously based on input, producing a gain factor that scales synaptic current.

**CfC Update Rule:**
1. Compute input-dependent time constant: `tau(x) = tau_min + range * sigmoid(W*x)`
2. Compute fast pathway candidate: `f1 = tanh(W1*x + 0.5*h)`
3. Compute slow pathway candidate: `f2 = tanh(W2*x + h)`
4. Gate interpolation: `h_new = gate * f1 + (1-gate) * f2`
5. Time constant smoothing: `h += (dt/tau) * (h_new - h)`
6. Output gain: `modulation = 1 + tanh(h)` (range [0, 2])

### `homeostasis.py`

Prevents neural explosions (runaway excitation / seizures) and death (complete silence) by regulating firing rates toward a biological target of ~5 Hz.

**Three mechanisms:**
1. **Gradual synaptic scaling** — Scale incoming weights based on individual neuron firing rate vs target
2. **Threshold adjustment** — Faster response than weight scaling
3. **Emergency GABAergic brake** — Hyperpolarize ALL neurons if >30% fire simultaneously

### `metaplasticity.py`

Dynamically adjusts learning parameters based on network stability, implementing the BCM sliding threshold theory.

**Algorithm:**
1. Track exponential moving average of mean firing rate
2. Compute stability index: `CV = sigma / mu` (coefficient of variation)
3. CV < 0.3 (stable) -> reduce A+/A-, increase prune threshold
4. CV > 0.7 (unstable) -> increase A+/A-, decrease prune threshold
5. Apply synapse crystallization for old synapses (reduced plasticity)

### `environment.py`

The main simulation loop that orchestrates all components. Streams Poisson-distributed input spikes with sinusoidal rate modulation to test adaptability.

**Execution Order (per timestep):**
1. Generate/receive input spikes
2. Propagate spikes through sparse weight matrix
3. Update liquid dynamics (CfC modulation)
4. Update membrane potentials (LIF + liquid gain)
5. Detect spikes
6. Apply STDP weight updates
7. Apply reward-modulated STDP (if enabled)
8. Structural plasticity (every 100 steps)
9. Homeostasis (every 50 steps)
10. Metaplasticity (every 500 steps)
11. Log metrics

### `visualization.py`

Generates a **4-panel dark-theme dashboard** and a supplementary metaplasticity panel using matplotlib.

**Dashboard Panels:**
1. **Neural Activity Over Time** — Mean firing rate + active neuron count
2. **Structural Plasticity** — Total synapse count with growth/pruning annotations
3. **Firing Rate Distribution** — Histogram with target and mean markers
4. **Weight Distribution** — Histogram showing STDP-driven bimodal pattern

### `main.py`

CLI entry point with argument parsing for timesteps, mode, visualization, and output directory.

---

## Installation

**Requirements:** Python 3.10+

```bash
# Clone or navigate to the project
cd neuromorphic_brain

# Install dependencies (CPU only - no GPU libraries)
pip install -r requirements.txt
```

**Dependencies:**
- `numpy >= 1.24.0` — Vectorized neuron computation
- `scipy >= 1.10.0` — Sparse matrix operations (CSR/LIL)
- `matplotlib >= 3.7.0` — Visualization dashboard

---

## Usage

```bash
# Default: 10,000 timesteps, unsupervised mode
python main.py

# Custom duration with visualization output
python main.py --timesteps 50000 --visualize

# Reward-modulated learning mode
python main.py --mode reward_modulated --timesteps 10000

# Long run, quiet mode, with dashboards
python main.py --timesteps 100000 --quiet --visualize

# Custom output directory
python main.py --timesteps 20000 --visualize --output-dir results
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--timesteps` | 10000 | Number of simulation timesteps |
| `--mode` | unsupervised | Learning mode: `unsupervised` or `reward_modulated` |
| `--visualize` | False | Generate dashboard PNGs after simulation |
| `--output-dir` | output/ | Directory for visualization files |
| `--quiet` | False | Suppress progress output |

---

## Simulation Results

### Evolution Dashboard (10,000 timesteps)

The 4-panel dashboard shows the network's self-organization over 10 seconds of simulated time:

![Evolution Dashboard](output/evolution_dashboard.png)

**Key Observations:**

1. **Top-Left (Neural Activity):** Firing rate spiked to 25+ Hz around t=6000, then homeostasis regulated it back to ~3.73 Hz (near the 5 Hz biological target). This demonstrates the system's **self-stabilizing** behavior.

2. **Top-Right (Structural Plasticity):** Synapse count grew steadily from 3,173 to 3,343 — a net gain of +170 connections. The network physically **rewired itself** by growing 180 new synapses and pruning 10 weak ones.

3. **Bottom-Left (Firing Rate Distribution):** Most neurons cluster around 0-5 Hz with a long tail from highly active input neurons. The distribution shows the homeostatic controller is working to keep neurons near the target rate.

4. **Bottom-Right (Weight Distribution):** The characteristic **bimodal distribution** produced by STDP — weights are pushed toward either 0 (weakened by LTD) or W_MAX=1.0 (strengthened by LTP). This is the hallmark of competitive Hebbian learning.

### Metaplasticity Panel

![Metaplasticity Panel](output/metaplasticity_panel.png)

**Key Observations:**

1. **Top (STDP Parameters):** A+ and |A-| remained relatively stable during the quiet period, dipped slightly during the stable phase (t=4000-5000), then recovered when the system detected environmental instability.

2. **Bottom (Stability CV):** The coefficient of variation jumped from near-zero to 1.3+ around t=6000, crossing the "unstable threshold" (0.7) and triggering **exploring mode**. The system correctly detected a change in its own dynamics and increased learning rates to adapt.

---

## Performance

| Metric | Value |
|--------|-------|
| Neurons | 256 |
| Synapses | ~3,300 (5% density) |
| Timestep | 1.0 ms |
| Simulated time | 10.0 seconds |
| Wall-clock time | ~81 seconds |
| Throughput | **123 timesteps/sec** |
| CPU only | Yes |
| GPU required | No |
| Memory footprint | ~50 MB |

**Scaling Notes:**
- Spike propagation is O(nnz) via CSR sparse mat-vec — scales with connection count, not neuron count squared
- Structural changes (grow/prune) amortized over 100-timestep intervals
- LIL-to-CSR conversion cost is O(nnz), occurring only after structural updates

---

## Research Papers & References

This implementation is based on the following research analyses:

### Paper 1: Architectures of Synthetic Cognition

*"Architectures of Synthetic Cognition: A Comprehensive Analysis of Brain-Inspired Neural Models, Real-Time Structural Evolution, and Autonomous Self-Management Systems"*

Key concepts drawn from this paper:
- LIF neuron models and spiking computation mechanics
- Liquid Time-Constant (LTC) Networks and CfC breakthrough
- P-STDP (presynaptic spike-driven STDP) for real-time connectivity inference
- ETSM (Enhanced Topographical Sparse Mapping) for biological pruning
- DEEP R structural plasticity rule for synapse growth
- HiVA framework and Semantic-Topological Evolution concepts
- Self-Evolving Embodied AI paradigm (memory self-updating, model self-evolution)

### Paper 2: The Neurobiological Architecture of Human Cognition

*"The Neurobiological Architecture of Human Cognition: A Comprehensive Analysis of Neural Systems, Structural Connectivity, and Emerging Frontiers in Brain Research"*

Key concepts drawn from this paper:
- Hodgkin-Huxley action potential mechanics and ionic gradients
- Excitatory/inhibitory balance (glutamate vs GABA)
- Structural connectome and white matter architecture
- P-FIT (Parieto-Frontal Integration Theory) and neural efficiency principle
- LTP/LTD mechanisms and BDNF-mediated plasticity modulation
- Predictive Coding framework and active inference
- Adult neurogenesis and microglia-regulated plasticity

### Additional Referenced Works

- Hodgkin, A.L. & Huxley, A.F. (1952). *A quantitative description of membrane current and its application to conduction and excitation in nerve.* Journal of Physiology.
- Bienenstock, E.L., Cooper, L.N., & Munro, P.W. (1982). *Theory for the development of neuron selectivity.* Journal of Neuroscience.
- Hasani, R. et al. (2022). *Closed-form continuous-time neural networks.* Nature Machine Intelligence.
- Bellec, G. et al. (2020). *A solution to the learning dilemma for recurrent networks of spiking neurons.* Nature Communications. (DEEP R rule)
- Jung, R. et al. (2007). *The Parieto-Frontal Integration Theory (P-FIT) of Intelligence.* Behavioral and Brain Sciences.

---

## How It Differs From Conventional AI

| Aspect | Traditional DNN | This System |
|--------|----------------|-------------|
| **Computation** | Dense matrix multiply (GPU) | Sparse event-driven (CPU) |
| **Learning** | Backpropagation (global gradient) | STDP (local spike timing) |
| **Architecture** | Fixed graph (designed by human) | Self-evolving (grows/prunes at runtime) |
| **Time** | Discrete steps (frames) | Continuous-time (CfC dynamics) |
| **Connectivity** | Fully connected layers | 5% sparse, biologically realistic |
| **Stability** | Batch norm / gradient clipping | Homeostasis + metaplasticity |
| **Memory** | Static weights (frozen after training) | Live state (never stops learning) |
| **Energy model** | Always computing (all neurons active) | Event-driven (only active neurons compute) |

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
