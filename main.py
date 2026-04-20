"""
main.py — Entry Point for Brain-Inspired Neuromorphic System
==============================================================

Initializes a "seed" network of 256 LIF neurons with 5% sparse
connectivity and runs the continuous unsupervised learning loop.

The system:
  ✅ Uses Spiking Neural Network (SNN) with LIF neurons
  ✅ Implements Structural Plasticity (grow + prune synapses at runtime)
  ✅ Integrates Liquid Neural Network (CfC) continuous-time dynamics
  ✅ Learns via STDP (no backpropagation)
  ✅ Supports Reward-Modulated STDP for task learning
  ✅ Homeostatic scaling prevents neural explosions
  ✅ Metaplasticity adapts learning rates to environmental stability
  ✅ Runs efficiently on CPU via sparse matrix operations
  ❌ No Transformers, Attention, or Backpropagation

Usage:
  python main.py                              # Default: 10,000 timesteps
  python main.py --timesteps 50000            # Custom duration
  python main.py --timesteps 10000 --visualize  # With dashboard output
  python main.py --mode reward_modulated      # Reward-modulated mode
"""

import argparse
import sys
import os

# Fix Windows console encoding for Unicode output
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Ensure the project directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from environment import EnvironmentLoop
from visualization import plot_evolution_dashboard, plot_metaplasticity_panel


def main():
    parser = argparse.ArgumentParser(
        description="Brain-Inspired Self-Evolving Neuromorphic AI System"
    )
    parser.add_argument(
        "--timesteps", type=int, default=cfg.DEFAULT_TIMESTEPS,
        help=f"Number of simulation timesteps (default: {cfg.DEFAULT_TIMESTEPS})"
    )
    parser.add_argument(
        "--mode", type=str, default="unsupervised",
        choices=["unsupervised", "reward_modulated"],
        help="Learning mode (default: unsupervised)"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Generate visualization dashboard after simulation"
    )
    parser.add_argument(
        "--output-dir", type=str, default="output",
        help="Directory for visualization output (default: output/)"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output"
    )
    args = parser.parse_args()

    # -- Banner -----------------------------------------------------------
    print()
    print("+==============================================================+")
    print("|   Brain-Inspired Self-Evolving Neuromorphic AI System        |")
    print("|                                                              |")
    print("|   Architecture: SNN (LIF) + LNN (CfC) + Structural Plast.  |")
    print("|   Learning: STDP + Reward-Modulated STDP (NO Backprop)      |")
    print("|   Stability: Homeostasis + Metaplasticity                   |")
    print("|   Hardware: CPU-optimized sparse matrices                   |")
    print("+==============================================================+")
    print()

    # -- Configuration summary --------------------------------------------
    print(f"[CONFIG] Neurons: {cfg.N_NEURONS} "
          f"(Input: {cfg.N_INPUT_NEURONS}, Output: {cfg.N_OUTPUT_NEURONS})")
    print(f"[CONFIG] Initial connectivity: {cfg.INITIAL_CONNECTIVITY*100:.1f}%")
    print(f"[CONFIG] Timestep: {cfg.DT} ms")
    print(f"[CONFIG] Mode: {args.mode}")
    print(f"[CONFIG] Duration: {args.timesteps} timesteps "
          f"({args.timesteps * cfg.DT / 1000:.1f} seconds simulated)")
    print()

    # -- Initialize and run -----------------------------------------------
    env = EnvironmentLoop(mode=args.mode)
    metrics = env.run(n_timesteps=args.timesteps, verbose=not args.quiet)

    # -- Visualization ----------------------------------------------------
    if args.visualize:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), args.output_dir
        )
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n[VIZ] Generating dashboards in {output_dir}/")

        # Main dashboard
        plot_evolution_dashboard(
            metrics, env.brain, env.synapses,
            save_path=os.path.join(output_dir, "evolution_dashboard.png")
        )

        # Metaplasticity panel
        plot_metaplasticity_panel(
            metrics,
            save_path=os.path.join(output_dir, "metaplasticity_panel.png")
        )

        print("[VIZ] Done - All dashboards generated")

    print("\n[DONE] System remains in live state - ready for continued learning.")
    return env


if __name__ == "__main__":
    main()
