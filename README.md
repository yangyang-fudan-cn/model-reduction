# RC/RLC Circuit Model Order Reduction Framework

A Python framework for reducing the order of RC and RLC circuit models described via SPICE netlists. Converts netlists into Modified Nodal Analysis (MNA) descriptor systems and applies model order reduction (MOR) algorithms to produce smaller, faster-to-simulate models while preserving key system properties.

## Features

- **SPICE netlist parser** — reads standard SPICE netlists with R, C, L, V, I elements, PULSE sources, and `.TRAN`/`.AC`/`.PRINT` control directives.
- **Modified Nodal Analysis (MNA)** — builds sparse descriptor system matrices `(G + sC)x = Bu`, `y = Lᵀx`.
- **Three MOR algorithms:**
  - **PRIMA** — Passive Reduced-order Interconnect Macromodeling Algorithm (block Krylov subspace via Arnoldi, passivity-preserving).
  - **Balanced Truncation (BT)** — truncates low-energy states via Hankel singular values; optimal error bound.
  - **Proper Orthogonal Decomposition (POD)** — data-driven SVD-based reduction from transient simulation snapshots.
- **Transient simulation** — Backward Euler / Trapezoidal time-domain integration.
- **AC analysis** — frequency-domain sweep with decade/octave/linear spacing.
- **Full vs. reduced comparison** — project reduced state back to full space and compute relative error; reports speedup.
- **Plotting** — optional matplotlib-based output of transient and AC comparisons.

## Requirements

- Python 3.9+
- `numpy`, `scipy`
- `matplotlib` (optional, for plotting)

## Installation

```bash
git clone <repo-url>
cd model_reduction
pip install numpy scipy matplotlib
```

## Usage

The framework is driven via the `main.py` CLI:

```bash
python main.py <netlist.sp> [options]
```

### Options

| Flag | Description |
|------|-------------|
| `--reduce`, `-r` | MOR algorithm: `prima`, `bt`, or `pod` |
| `--order`, `-o` | Target reduced order (default: `n_state // 10`) |
| `--expansion-point`, `-s0` | Expansion point for PRIMA (default: `0`) |
| `--simulate` | Run transient simulation |
| `--ac` | Run AC frequency sweep |
| `--tstep` | Transient time step (overrides netlist) |
| `--tstop` | Transient stop time (overrides netlist) |
| `--fstart` | AC start frequency in Hz (default: `1e3`) |
| `--fstop` | AC stop frequency in Hz (default: `1e11`) |
| `--np` | Number of AC frequency points (default: `100`) |
| `--compare` | Compare full vs. reduced model simulation |
| `--plot` | Generate matplotlib comparison plots |

### Examples

Reduce an RLC filter using PRIMA with order 10:

```bash
python main.py mor_framework/examples/rlc_filter.sp --reduce prima --order 10 --compare --simulate --plot
```

Reduce an RC ladder using Balanced Truncation and run AC analysis:

```bash
python main.py mor_framework/examples/rc_ladder.sp --reduce bt --order 5 --ac --plot
```

Reduce using POD from transient snapshots:

```bash
python main.py mor_framework/examples/clock_tree.sp --reduce pod --order 20 --simulate --plot
```

## Project Structure

```
├── main.py                           # CLI entry point
├── mor_framework/
│   ├── core/
│   │   ├── circuit.py                # Circuit data model (Element, SourceWaveform, Circuit)
│   │   ├── netlist_parser.py         # SPICE netlist parser
│   │   └── mna.py                    # Modified Nodal Analysis matrix construction
│   ├── mor/
│   │   ├── base.py                   # Abstract MOR base class & ReducedModel dataclass
│   │   ├── prima.py                  # PRIMA: block Krylov subspace via Arnoldi
│   │   ├── balanced_truncation.py    # Balanced Truncation via Lyapunov equations
│   │   └── pod.py                    # POD: SVD-based reduction from snapshots
│   ├── simulation/
│   │   ├── transient.py             # Time-domain simulation (BE / TRAP)
│   │   └── ac.py                     # Frequency-domain AC analysis
│   └── examples/
│       ├── rlc_filter.sp             # Example RLC filter netlist
│       ├── rc_ladder.sp              # Example RC ladder netlist
│       └── clock_tree.sp             # Example clock tree netlist
├── rc_rcl_model_reduction_algorithms.md  # Survey of RC/RCL MOR algorithms
└── .gitignore
```

## References

- A. Odabasioglu, M. Celik, L. T. Pileggi, "PRIMA: passive reduced-order interconnect macromodeling algorithm," *IEEE TCAD*, 1998.
- B. Moore, "Principal component analysis in linear systems: Controllability, observability, and model reduction," *IEEE TAC*, 1981.
- S. Lall, "Structure-preserving model reduction," *Mechanical Systems and Signal Processing*, 2003.

## License

This project is for educational and research use.
