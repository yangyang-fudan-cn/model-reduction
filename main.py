#!/usr/bin/env python3
"""MOR Framework — main CLI entry point.

Usage:
    python main.py <netlist.sp> [options]

Examples:
    python main.py examples/clock_tree.sp --simulate
    python main.py examples/rlc_filter.sp --reduce prima --order 10
    python main.py examples/rc_ladder.sp --reduce bt --order 5 --simulate --ac
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mor_framework.core.netlist_parser import parse_netlist
from mor_framework.core.mna import build_mna
from mor_framework.mor.prima import PRIMA
from mor_framework.mor.balanced_truncation import BalancedTruncation
from mor_framework.mor.pod import POD
from mor_framework.simulation.transient import simulate_transient
from mor_framework.simulation.ac import ac_analysis

MOR_REGISTRY = {
    "prima": PRIMA,
    "bt": BalancedTruncation,
    "pod": POD,
}


def _pulse_input(v1=0.0, v2=1.0, td=0.0, tr=1e-10, tf=1e-10, pw=5e-9, per=1e-8):
    """Create a SPICE-style pulse input function."""
    def func(t):
        t_mod = t % per
        if t_mod < td:
            return np.array([v1])
        if t_mod < td + tr:
            frac = (t_mod - td) / tr
            return np.array([v1 + frac * (v2 - v1)])
        if t_mod < td + tr + pw:
            return np.array([v2])
        if t_mod < td + tr + pw + tf:
            frac = (t_mod - td - tr - pw) / tf
            return np.array([v2 + frac * (v1 - v2)])
        return np.array([v1])
    return func


def main():
    parser = argparse.ArgumentParser(
        description="RC/RCL Network Model Order Reduction Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("netlist", type=str, help="Path to SPICE netlist")
    parser.add_argument("--reduce", "-r", type=str, default=None,
                        choices=list(MOR_REGISTRY),
                        help="MOR algorithm to use")
    parser.add_argument("--order", "-o", type=int, default=None,
                        help="Target reduced order")
    parser.add_argument("--expansion-point", "-s0", type=float, default=0.0,
                        help="Expansion point for PRIMA (default: 0)")
    parser.add_argument("--simulate", action="store_true",
                        help="Run transient simulation")
    parser.add_argument("--ac", action="store_true",
                        help="Run AC frequency sweep")
    parser.add_argument("--tstep", type=float, default=None,
                        help="Transient time step (overrides netlist)")
    parser.add_argument("--tstop", type=float, default=None,
                        help="Transient stop time (overrides netlist)")
    parser.add_argument("--fstart", type=float, default=1e3,
                        help="AC start frequency (Hz)")
    parser.add_argument("--fstop", type=float, default=1e11,
                        help="AC stop frequency (Hz)")
    parser.add_argument("--np", type=int, default=100,
                        help="Number of AC points")
    parser.add_argument("--compare", action="store_true",
                        help="Compare full vs reduced model simulation")
    parser.add_argument("--plot", action="store_true",
                        help="Generate plots (requires matplotlib)")

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # 1. Parse netlist
    # ---------------------------------------------------------------
    netlist_path = Path(args.netlist)
    if not netlist_path.exists():
        print(f"Error: netlist not found: {netlist_path}")
        sys.exit(1)

    print(f"Parsing netlist: {netlist_path}")
    circuit = parse_netlist(str(netlist_path))
    print(f"  Title: {circuit.title or '(none)'}")
    print(f"  Elements: {len(circuit.elements)}")
    for t, name in [(0, "R"), (1, "C"), (2, "L"), (3, "V"), (4, "I")]:
        cnt = sum(1 for e in circuit.elements if e.elem_type.value == name)
        print(f"    {name}: {cnt}")

    # ---------------------------------------------------------------
    # 2. Build MNA matrices
    # ---------------------------------------------------------------
    print("\nBuilding MNA system...")
    G, C, B, L, output_labels = build_mna(circuit)
    n_state = G.shape[0]
    n_inputs = B.shape[1]
    n_outputs = L.shape[1]
    print(f"  State dimension: {n_state}")
    print(f"  Inputs: {n_inputs}, Outputs: {n_outputs}")

    # ---------------------------------------------------------------
    # 3. Model Order Reduction
    # ---------------------------------------------------------------
    reduced_model = None
    if args.reduce:
        algo_cls = MOR_REGISTRY[args.reduce]
        algo_name = args.reduce.upper()
        order = args.order

        if order is None:
            order = max(1, n_state // 10)
            print(f"\n  Auto-selected reduced order: {order}")

        if order >= n_state:
            print(f"\n  Warning: reduced order ({order}) >= full order ({n_state}). "
                  f"Setting to {n_state - 1}.")
            order = n_state - 1

        print(f"\nReducing model ({algo_name}) to order {order}...")

        algo_kwargs = {}
        if args.reduce == "prima":
            algo_kwargs["expansion_point"] = args.expansion_point
            algo = algo_cls(expansion_point=args.expansion_point)
        else:
            algo = algo_cls()

        t0 = time.time()
        reduced_model = algo.reduce(G, C, B, L, order, **algo_kwargs)
        t_elapsed = time.time() - t0

        print(f"  Reduced order: {reduced_model.reduced_order}")
        print(f"  Reduction time: {t_elapsed:.4f} s")
        if "hsv" in reduced_model.info:
            hsv = reduced_model.info["hsv"]
            print(f"  Hankel SV range: [{hsv[-1]:.6e}, {hsv[0]:.6e}]")
            kept_energy = np.sum(hsv[:order] ** 2) / np.sum(hsv ** 2) * 100
            print(f"  Energy retained: {kept_energy:.2f}%")
        if "energy_ratio" in reduced_model.info:
            print(f"  POD energy ratio: {reduced_model.info['energy_ratio']:.6f}")
        if "singular_values" in reduced_model.info:
            sv = reduced_model.info["singular_values"]
            print(f"  Singular values range: [{sv[-1]:.6e}, {sv[0]:.6e}]")

        # Print matrix sparsity comparison
        nnz_full = G.nnz + C.nnz
        nnz_red = reduced_model.Gr.nnz + reduced_model.Cr.nnz
        print(f"  Non-zeros: {nnz_full} → {nnz_red} "
              f"({nnz_red / max(nnz_full, 1) * 100:.1f}%)")

    # ---------------------------------------------------------------
    # 4. Simulation
    # ---------------------------------------------------------------
    if args.simulate:
        _run_transient(circuit, G, C, B, L, output_labels, reduced_model, args)

    if args.ac:
        _run_ac_analysis(G, C, B, L, output_labels, reduced_model, args)

    if not (args.simulate or args.ac or args.reduce):
        print("\nDone. Use --simulate, --ac, and/or --reduce to perform analyses.")


def _run_transient(circuit, G, C, B, L, output_labels, reduced_model, args):
    """Run transient simulation, optionally on both full and reduced models."""
    # Determine time parameters
    dt = args.tstep
    t_end = args.tstop
    if dt is None:
        for a in circuit.analyses:
            if a.analysis_type == "tran" and a.tran_params:
                dt = a.tran_params[0]
                t_end = a.tran_params[1]
                break
    if dt is None:
        dt = 1e-11
    if t_end is None:
        t_end = 2e-8

    n_steps = max(10, int(t_end / dt))

    # Build input function from netlist sources
    vsrcs = circuit.voltage_sources()
    input_func = None
    if vsrcs and vsrcs[0].waveform and vsrcs[0].waveform.pulse_params:
        pp = vsrcs[0].waveform.pulse_params
        input_func = _pulse_input(v1=pp[0], v2=pp[1], td=pp[2],
                                  tr=pp[3], tf=pp[4], pw=pp[5], per=pp[6])

    print(f"\nTransient simulation ({n_steps} steps, dt={dt:.2e}, t_end={t_end:.2e})")

    if args.compare and reduced_model is not None:
        # Full model
        print("  Simulating full model...")
        t0 = time.time()
        x_full, tout = simulate_transient(G, C, B, (0, t_end), n_steps,
                                           input_func=input_func)
        t_full = time.time() - t0

        # Reduced model
        print("  Simulating reduced model...")
        t0 = time.time()
        x_red, _ = simulate_transient(reduced_model.Gr, reduced_model.Cr,
                                       reduced_model.Br, (0, t_end), n_steps,
                                       input_func=input_func)
        t_red = time.time() - t0

        # Project reduced state back to full space for comparison
        x_red_full = reduced_model.Q @ x_red

        # Compute error at output nodes
        n_out = L.shape[1]
        if n_out > 0:
            y_full = L.T @ x_full if hasattr(L.T, "__matmul__") else L.T @ x_full
            y_red = L.T @ x_red_full
            err = np.linalg.norm(y_full - y_red, axis=1) / (
                np.linalg.norm(y_full, axis=1) + 1e-30
            )
            print(f"  Full model time: {t_full:.4f} s")
            print(f"  Reduced model time: {t_red:.4f} s")
            print(f"  Speedup: {t_full / max(t_red, 1e-30):.1f}x")
            for i, label in enumerate(output_labels[:len(err)]):
                print(f"  {label} relative error: {err[i]:.4e}")

        if args.plot:
            _plot_results(tout, y_full, y_red, output_labels,
                          "Transient: Full vs Reduced", "Time (s)", "Voltage (V)")

    else:
        if reduced_model is not None:
            print("  Simulating reduced model...")
            G_sim, C_sim, B_sim = (reduced_model.Gr, reduced_model.Cr, reduced_model.Br)
        else:
            print("  Simulating full model...")
            G_sim, C_sim, B_sim = G, C, B

        t0 = time.time()
        x_sol, tout = simulate_transient(G_sim, C_sim, B_sim, (0, t_end),
                                          n_steps, input_func=input_func)
        t_sim = time.time() - t0
        print(f"  Simulation time: {t_sim:.4f} s")

        # Reconstruct full state if reduced
        if reduced_model is not None:
            x_sol = reduced_model.Q @ x_sol

        print(f"  State at t={t_end:.2e}: min={x_sol[:, -1].min():.4e}, "
              f"max={x_sol[:, -1].max():.4e}")


def _run_ac_analysis(G, C, B, L, output_labels, reduced_model, args):
    """Run AC frequency sweep."""
    if reduced_model is not None:
        print("\nAC analysis (reduced model)...")
        G_ac, C_ac, B_ac, L_ac = (
            reduced_model.Gr, reduced_model.Cr,
            reduced_model.Br, reduced_model.Lr,
        )
    else:
        print("\nAC analysis (full model)...")
        G_ac, C_ac, B_ac, L_ac = G, C, B, L

    t0 = time.time()
    freqs, Y = ac_analysis(G_ac, C_ac, B_ac, L_ac,
                            f_start=args.fstart, f_stop=args.fstop,
                            n_points=args.np)
    t_ac = time.time() - t0
    print(f"  Frequency points: {len(freqs)}")
    print(f"  AC analysis time: {t_ac:.4f} s")

    for i, label in enumerate(output_labels[:Y.shape[0]]):
        mag = np.abs(Y[i, :])
        phase = np.angle(Y[i, :], deg=True)
        print(f"  {label}: |H| range [{mag.min():.4e}, {mag.max():.4e}], "
              f"phase [{phase.min():.1f}°, {phase.max():.1f}°]")

    if args.plot:
        _plot_results(freqs, np.abs(Y), None, output_labels,
                      "AC Analysis", "Frequency (Hz)", "Magnitude",
                      logx=True, logy=True)


def _plot_results(x, y_full, y_red, labels, title, xlabel, ylabel,
                  logx=False, logy=False):
    """Generate matplotlib plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Install matplotlib for plots: pip install matplotlib")
        return

    n_plots = min(4, len(labels))
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 3 * n_plots), squeeze=False)
    fig.suptitle(title)

    for i in range(n_plots):
        ax = axes[i][0]
        label = labels[i] if i < len(labels) else f"Output {i}"
        ax.plot(x, y_full[i] if y_full.ndim > 1 else y_full, label=f"Full: {label}")
        if y_red is not None:
            ax.plot(x, y_red[i] if y_red.ndim > 1 else y_red,
                    "--", label=f"Reduced: {label}")
        if logx:
            ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = f"mor_{title.lower().replace(' ', '_')}.png"
    plt.savefig(out_path, dpi=150)
    print(f"  Plot saved: {out_path}")


if __name__ == "__main__":
    main()
