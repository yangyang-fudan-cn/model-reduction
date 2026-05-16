# RC/RCL Network Model Reduction Algorithms — GitHub Implementations

A comprehensive survey of resistor-capacitor (RC) and resistor-capacitor-inductor (RCL) network model order reduction (MOR) algorithms and their open-source implementations on GitHub.

---

## Key RC/RCL Model Reduction Algorithms

### 1. Krylov Subspace Methods

| Algorithm | Description |
|---|---|
| **PRIMA** | Passive Reduced-order Interconnect Macromodeling Algorithm — block Krylov subspace projection that preserves passivity |
| **SPRIM** | Structure-Preserving version of PRIMA that exploits block structure of MNA matrices |
| **Block SAPOR** | S-parameter-based Passive Order Reduction via block Krylov subspace |

### 2. Truncation-Based Methods

| Algorithm | Description |
|---|---|
| **Balanced Truncation (BT)** | Truncates states with small Hankel singular values; preserves stability |
| **Modal Truncation (MT)** | Truncates modes with negligible contribution |
| **Balanced/Modal Singular Perturbation (BSP/MSP)** | Residualizes fast modes instead of truncating |

### 3. SVD-Based Methods

| Algorithm | Description |
|---|---|
| **POD (Proper Orthogonal Decomposition)** | Data-driven SVD-based reduction from simulation snapshots |
| **POD-DEIM** | POD with Discrete Empirical Interpolation Method for nonlinear terms |

### 4. Rational Interpolation

| Algorithm | Description |
|---|---|
| **Vector Fitting (VF)** | Rational approximation in frequency domain for S-parameter fitting |
| **Loewner Framework** | Data-driven rational interpolation |
| **AAA Algorithm** | Adaptive Antoulas-Anderson for rational approximation |

### 5. RC-Specific Reduction

| Algorithm | Description |
|---|---|
| **TICER** | Time-Constant Equilibration Reduction — eliminates nodes with small time constants in RC trees |
| **Elmore Delay-based reduction** | PI-model / T-model reduction of RC interconnects |
| **SparseRC** | Partitioned BBD-form RC network reduction |

---

## GitHub Implementations

| Repository | Language | Algorithms | Description |
|---|---|---|---|
| **[mm318/rlc-circuit-mor](https://github.com/mm318/rlc-circuit-mor)** | Python | **PRIMA** | MNA + PRIMA block Krylov subspace MOR for RLC circuits. Reads SPICE netlists, performs reduction, compares transient/frequency response. |
| **[pymor/pymor](https://github.com/pymor/pymor)** | Python | **POD, RBM, DMD, Loewner, AAA, DEIM** | General-purpose MOR library. Supports system-theoretic methods for LTI systems (usable for RC/RCL models). NumFOCUS affiliated. |
| **[SciML/ModelOrderReduction.jl](https://github.com/SciML/ModelOrderReduction.jl)** | Julia | **POD-DEIM** | Integrates with ModelingToolkit.jl; automatically reduces ODE/PDESystems via projection. Has RC circuit benchmarks. |
| **[RasulChoupanzadeh/SROPEE](https://github.com/RasulChoupanzadeh/SROPEE)** | Python | **Vector Fitting + Block SAPOR** | Full pipeline: S-parameter → Vector Fitting → passivity enforcement → Block SAPOR MOR → reduced SPICE netlist. Optical interconnect focus. |
| **[forgi86/lru-reduction](https://github.com/forgi86/lru-reduction)** | Python | **BT, BSP, MT, MSP** | Balanced/Modal truncation and singular perturbation for state-space models (Linear Recurrent Units). Includes Hankel nuclear norm regularization. |
| **[erdc/pynirom](https://github.com/erdc/pynirom)** | Python | Non-intrusive ROM | Data-driven reduced order modeling tools. |
| **[mpimd-csc/morgen](https://github.com/mpimd-csc/morgen)** | Python/C++ | Moment-matching, POD | MOR for gas and energy networks (circuit-equivalent models). |
| **[ghenze/rcnetworks](https://github.com/ghenze/rcnetworks)** | Modelica | MOR optimization | Thermal RC network models in Modelica with optimization-based model order reduction. Teaching-oriented. |
| **[iampraj/Model-Order-Reduction](https://github.com/iampraj/Model-Order-Reduction)** | — | Overview | Educational repo explaining POD, RBM, balancing methods, Krylov subspace methods. |

---

## Summary by Use Case

### If you're working with SPICE netlists (RLC circuits)

Use **[mm318/rlc-circuit-mor](https://github.com/mm318/rlc-circuit-mor)** (PRIMA) — directly reads SPICE, does MNA, reduces via block Krylov subspace. Use **[SROPEE](https://github.com/RasulChoupanzadeh/SROPEE)** if starting from S-parameter touchstone files.

### If you need general-purpose MOR applicable to circuit matrices

Use **[pyMOR](https://github.com/pymor/pymor)** (Python) for POD, DMD, Loewner, AAA, reduced basis methods. Use **[ModelOrderReduction.jl](https://github.com/SciML/ModelOrderReduction.jl)** (Julia) for POD-DEIM with automated ODE transformation.

### If you want balanced truncation / singular perturbation

Use **[forgi86/lru-reduction](https://github.com/forgi86/lru-reduction)** for BT, BSP, MT, MSP on state-space models.

### For RC interconnect-specific reduction (TICER, Elmore, SparseRC)

These are more commonly found in commercial/industrial VLSI tools (e.g., **OpenROAD/OpenSTA**, **Cadence**, **Synopsys**) rather than standalone open-source repos. The **[mm318/rlc-circuit-mor](https://github.com/mm318/rlc-circuit-mor)** repo and **pyMOR** are the closest open-source options.

---

## References

- A. Odabasioglu, M. Celik, L. T. Pileggi, "PRIMA: passive reduced-order interconnect macromodeling algorithm," *IEEE TCAD*, 1998.
- R. W. Freund, "SPRIM: structure-preserving reduced-order interconnect macromodeling," *IEEE/ACM ICCAD*, 2004.
- B. N. Sheehan, "TICER: Realizable reduction of extracted RC circuits," *IEEE/ACM ICCAD*, 1999.
- R. Milk, S. Rave, F. Schindler, "pyMOR — Generic Algorithms and Interfaces for Model Order Reduction," *SIAM J. Sci. Comput.*, 2016.
- M. Forgione, M. Mejari, D. Piga, "Model order reduction of deep structured state-space models: A system-theoretic approach," *CDC*, 2024.

---

*Survey compiled on 2026-05-16.*
