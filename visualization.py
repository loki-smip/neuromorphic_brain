"""
visualization.py — Real-time Network Evolution Dashboard
==========================================================

4-panel visualization dashboard showing the brain's evolution:
  1. Network graph — neurons (color=firing rate), edges (active synapses)
  2. Synapse evolution — total count over time (growth/pruning dynamics)
  3. Firing rate distribution — histogram showing homeostatic regulation
  4. Weight distribution — histogram showing STDP-driven learning

Uses matplotlib for CPU-friendly rendering (no GPU dependencies).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os


def plot_evolution_dashboard(metrics_log: dict, brain, synapse_manager,
                             save_path: str = None, show: bool = False):
    """
    Generate a 4-panel dashboard showing network evolution.

    Parameters
    ----------
    metrics_log : dict
        Metrics collected by EnvironmentLoop.
    brain : BrainNetwork
        Current brain state.
    synapse_manager : SynapseManager
        Current synapse state.
    save_path : str, optional
        If provided, save the figure to this path.
    show : bool
        If True, display the plot interactively.
    """
    fig = plt.figure(figsize=(16, 10), facecolor='#1a1a2e')
    fig.suptitle('Brain-Inspired Neuromorphic System — Evolution Dashboard',
                 fontsize=16, fontweight='bold', color='#e0e0e0', y=0.98)

    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3,
                  left=0.08, right=0.95, top=0.92, bottom=0.08)

    # Color scheme
    bg_color = '#16213e'
    text_color = '#e0e0e0'
    accent1 = '#00d2ff'
    accent2 = '#ff6b6b'
    accent3 = '#feca57'
    accent4 = '#48dbfb'

    timesteps = metrics_log.get("timestep", [])

    # ── Panel 1: Network Activity (Top-Left) ────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(bg_color)

    if len(timesteps) > 0:
        rates = metrics_log["mean_firing_rate"]
        spikes = metrics_log["num_spikes"]

        ax1.plot(timesteps, rates, color=accent1, linewidth=1.5,
                 alpha=0.9, label='Mean Firing Rate (Hz)')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(timesteps, spikes, color=accent2, linewidth=1.0,
                      alpha=0.6, label='Active Neurons')
        ax1_twin.set_ylabel('Active Neurons', color=accent2, fontsize=9)
        ax1_twin.tick_params(axis='y', labelcolor=accent2, labelsize=8)
        ax1_twin.spines['right'].set_color(accent2)

    ax1.set_xlabel('Timestep', color=text_color, fontsize=9)
    ax1.set_ylabel('Firing Rate (Hz)', color=accent1, fontsize=9)
    ax1.set_title('Neural Activity Over Time', color=text_color,
                  fontsize=11, fontweight='bold')
    ax1.tick_params(colors=text_color, labelsize=8)
    ax1.spines['bottom'].set_color('#404060')
    ax1.spines['left'].set_color(accent1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.legend(loc='upper left', fontsize=8, framealpha=0.3,
               labelcolor=text_color)

    # ── Panel 2: Synapse Evolution (Top-Right) ──────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(bg_color)

    if len(timesteps) > 0:
        synapses = metrics_log["num_synapses"]
        ax2.plot(timesteps, synapses, color=accent3, linewidth=2.0, alpha=0.9)
        ax2.fill_between(timesteps, synapses, alpha=0.15, color=accent3)

        # Annotate growth/pruning
        weight_stats = synapse_manager.get_weight_stats()
        ax2.annotate(f"Pruned: {weight_stats['pruned_total']}\nGrown: {weight_stats['grown_total']}",
                     xy=(0.98, 0.95), xycoords='axes fraction',
                     ha='right', va='top', fontsize=9, color=text_color,
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f3460', alpha=0.7))

    ax2.set_xlabel('Timestep', color=text_color, fontsize=9)
    ax2.set_ylabel('Total Synapses', color=accent3, fontsize=9)
    ax2.set_title('Structural Plasticity — Synapse Count', color=text_color,
                  fontsize=11, fontweight='bold')
    ax2.tick_params(colors=text_color, labelsize=8)
    ax2.spines['bottom'].set_color('#404060')
    ax2.spines['left'].set_color(accent3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # ── Panel 3: Firing Rate Distribution (Bottom-Left) ─────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(bg_color)

    firing_rates = brain.get_firing_rates()
    if np.any(firing_rates > 0):
        ax3.hist(firing_rates, bins=30, color=accent4, alpha=0.7, edgecolor='white',
                 linewidth=0.5)
        ax3.axvline(x=5.0, color=accent2, linestyle='--', linewidth=1.5,
                    label=f'Target: 5 Hz', alpha=0.8)
        ax3.axvline(x=np.mean(firing_rates), color=accent3, linestyle='-.',
                    linewidth=1.5, label=f'Mean: {np.mean(firing_rates):.2f} Hz', alpha=0.8)
        ax3.legend(fontsize=8, framealpha=0.3, labelcolor=text_color)
    else:
        ax3.text(0.5, 0.5, 'No firing activity yet', ha='center', va='center',
                 color=text_color, fontsize=12, transform=ax3.transAxes)

    ax3.set_xlabel('Firing Rate (Hz)', color=text_color, fontsize=9)
    ax3.set_ylabel('Neuron Count', color=text_color, fontsize=9)
    ax3.set_title('Firing Rate Distribution (Homeostasis)', color=text_color,
                  fontsize=11, fontweight='bold')
    ax3.tick_params(colors=text_color, labelsize=8)
    ax3.spines['bottom'].set_color('#404060')
    ax3.spines['left'].set_color('#404060')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # ── Panel 4: Weight Distribution (Bottom-Right) ─────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(bg_color)

    weight_data = synapse_manager.weights_csr.data
    if len(weight_data) > 0:
        ax4.hist(weight_data, bins=50, color='#a29bfe', alpha=0.7,
                 edgecolor='white', linewidth=0.3)
        ax4.axvline(x=np.mean(weight_data), color=accent3, linestyle='-.',
                    linewidth=1.5, label=f'Mean: {np.mean(weight_data):.4f}', alpha=0.8)
        ax4.legend(fontsize=8, framealpha=0.3, labelcolor=text_color)
    else:
        ax4.text(0.5, 0.5, 'No synapses', ha='center', va='center',
                 color=text_color, fontsize=12, transform=ax4.transAxes)

    ax4.set_xlabel('Synaptic Weight', color=text_color, fontsize=9)
    ax4.set_ylabel('Synapse Count', color=text_color, fontsize=9)
    ax4.set_title('Weight Distribution (STDP Learning)', color=text_color,
                  fontsize=11, fontweight='bold')
    ax4.tick_params(colors=text_color, labelsize=8)
    ax4.spines['bottom'].set_color('#404060')
    ax4.spines['left'].set_color('#404060')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # ── Save / Show ─────────────────────────────────────────────────
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(),
                    edgecolor='none', bbox_inches='tight')
        print(f"[VIZ] Dashboard saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_metaplasticity_panel(metrics_log: dict, save_path: str = None):
    """
    Supplementary plot showing metaplasticity dynamics.

    Two subplots:
      - STDP parameters (A+, A-) over time
      - Stability CV over time
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), facecolor='#1a1a2e')

    bg_color = '#16213e'
    text_color = '#e0e0e0'
    timesteps = metrics_log.get("timestep", [])

    if len(timesteps) == 0:
        plt.close(fig)
        return

    # ── Panel A: STDP Parameters ────────────────────────────────────
    ax1.set_facecolor(bg_color)
    ax1.plot(timesteps, metrics_log["a_plus"], color='#00d2ff', linewidth=1.5,
             label='A+ (LTP)')
    a_minus_abs = [abs(a) for a in metrics_log["a_minus"]]
    ax1.plot(timesteps, a_minus_abs, color='#ff6b6b', linewidth=1.5,
             label='|A-| (LTD)')
    ax1.set_ylabel('STDP Amplitude', color=text_color, fontsize=10)
    ax1.set_title('Metaplasticity — Learning Rate Adaptation', color=text_color,
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, framealpha=0.3, labelcolor=text_color)
    ax1.tick_params(colors=text_color, labelsize=8)
    ax1.spines['bottom'].set_color('#404060')
    ax1.spines['left'].set_color('#404060')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ── Panel B: Stability Index ────────────────────────────────────
    ax2.set_facecolor(bg_color)
    ax2.plot(timesteps, metrics_log["stability_cv"], color='#feca57', linewidth=1.5)
    ax2.axhline(y=0.3, color='#00d2ff', linestyle='--', alpha=0.5, label='Stable threshold')
    ax2.axhline(y=0.7, color='#ff6b6b', linestyle='--', alpha=0.5, label='Unstable threshold')
    ax2.fill_between(timesteps, 0, 0.3, alpha=0.05, color='#00d2ff')
    ax2.fill_between(timesteps, 0.7, max(metrics_log["stability_cv"]) * 1.1 if metrics_log["stability_cv"] else 1.0,
                     alpha=0.05, color='#ff6b6b')
    ax2.set_xlabel('Timestep', color=text_color, fontsize=10)
    ax2.set_ylabel('Stability CV', color=text_color, fontsize=10)
    ax2.set_title('Environmental Stability Index', color=text_color,
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, framealpha=0.3, labelcolor=text_color)
    ax2.tick_params(colors=text_color, labelsize=8)
    ax2.spines['bottom'].set_color('#404060')
    ax2.spines['left'].set_color('#404060')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        fig.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(),
                    edgecolor='none', bbox_inches='tight')
        print(f"[VIZ] Metaplasticity panel saved to: {save_path}")

    plt.close(fig)
