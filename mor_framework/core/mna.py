"""Modified Nodal Analysis (MNA) matrix formulation for RC/RCL circuits.

Builds the descriptor system:

    (G + sC) x = B u
          y  = L^T x

where  x = [V_nodes; I_Vsrc; I_L]ᵀ.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse as sp

from .circuit import Circuit, Element, ElementType


def _node_idx(nodes: List[str], node_map: Dict[str, int], n: str) -> int:
    """Return the MNA index for a node (ground → -1, others → map)."""
    if n == "0":
        return -1
    return node_map[n]


def build_mna(circuit: Circuit) -> Tuple[sp.csc_matrix, sp.csc_matrix,
                                          sp.csc_matrix, sp.csc_matrix,
                                          List[str]]:
    """Build the MNA matrices G, C, B, L from the circuit.

    Returns
    -------
    G : sparse csc_matrix  (n_state × n_state)
    C : sparse csc_matrix  (n_state × n_state)
    B : sparse csc_matrix  (n_state × n_inputs)
    L : sparse csc_matrix  (n_state × n_outputs)
    output_labels : list of str describing each output
    """
    nmap = circuit.node_map()
    n_nodes = circuit.num_nodes()
    n_vsrc = circuit.num_voltage_sources()
    n_ind  = circuit.num_inductors()
    n_state = n_nodes + n_vsrc + n_ind

    # Count inputs (independent voltage + current sources)
    vsrcs = circuit.voltage_sources()
    isrcs = circuit.current_sources()
    n_inputs = len(vsrcs) + len(isrcs)
    n_outputs = max(1, n_inputs)

    G = sp.lil_matrix((n_state, n_state), dtype=np.float64)
    C = sp.lil_matrix((n_state, n_state), dtype=np.float64)
    B = sp.lil_matrix((n_state, n_inputs), dtype=np.float64)
    L = sp.lil_matrix((n_state, n_outputs), dtype=np.float64)

    # Helper: index into x vector
    def _idx(name: str) -> int:
        if name == "0":
            return -1
        return nmap[name]

    # ----- Resistor stamps -----
    for elem in circuit.elem_by_type(ElementType.RESISTOR):
        a = _idx(elem.n_plus)
        b = _idx(elem.n_minus)
        g = 1.0 / elem.value
        for (i, j) in [(a, a), (a, b), (b, a), (b, b)]:
            if i >= 0 and j >= 0:
                G[i, j] += g if i == j else -g
            elif i >= 0 and j < 0:  # ground connection
                pass  # voltage at ground is 0
            elif i < 0 and j >= 0:
                pass

    # ----- Capacitor stamps -----
    for elem in circuit.elem_by_type(ElementType.CAPACITOR):
        a = _idx(elem.n_plus)
        b = _idx(elem.n_minus)
        c_val = elem.value
        for (i, j) in [(a, a), (a, b), (b, a), (b, b)]:
            if i >= 0 and j >= 0:
                C[i, j] += c_val if i == j else -c_val

    # ----- Inductor stamps -----
    for k, elem in enumerate(circuit.elem_by_type(ElementType.INDUCTOR)):
        eq_idx = n_nodes + n_vsrc + k       # equation row
        var_idx = eq_idx                    # variable column
        a = _idx(elem.n_plus)
        b = _idx(elem.n_minus)
        l_val = elem.value

        # Structural: V(a) - V(b) - L * dI_L/dt = 0
        if a >= 0:
            G[eq_idx, a] = 1.0
            G[a, eq_idx] = 1.0
        if b >= 0:
            G[eq_idx, b] = -1.0
            G[b, eq_idx] = -1.0
        if eq_idx >= a:
            C[eq_idx, var_idx] = -l_val

    # ----- Voltage-source stamps -----
    for k, elem in enumerate(vsrcs):
        eq_idx = n_nodes + k
        var_idx = eq_idx
        a = _idx(elem.n_plus)
        b = _idx(elem.n_minus)

        if a >= 0:
            G[eq_idx, a] = 1.0
            G[a, eq_idx] = 1.0
        if b >= 0:
            G[eq_idx, b] = -1.0
            G[b, eq_idx] = -1.0

        # B: each V-source gets its own input column
        B[eq_idx, k] = 1.0

    # ----- Current-source stamps -----
    for k, elem in enumerate(isrcs):
        a = _idx(elem.n_plus)
        b = _idx(elem.n_minus)
        col = len(vsrcs) + k
        if a >= 0:
            B[a, col] = 1.0
        if b >= 0:
            B[b, col] = -1.0

    # ----- Output matrix L (observe node voltages by default) -----
    output_labels: List[str] = []
    # Collect from analysis print vars if available
    observed_vars: List[str] = []
    for analysis in circuit.analyses:
        for v in analysis.print_vars:
            observed_vars.append(v)

    if not observed_vars:
        # Default: observe all node voltages
        for node, idx in nmap.items():
            observed_vars.append(f"V({node})")

    for col, ov in enumerate(observed_vars[:n_outputs]):
        ov_upper = ov.upper()
        if ov_upper.startswith("V(") and ov_upper.endswith(")"):
            node = ov[2:-1]
            idx = _idx(node)
            if idx >= 0:
                L[idx, col] = 1.0
                output_labels.append(ov)
            else:
                # Observing ground
                output_labels.append(ov)
        else:
            output_labels.append(ov)

    return (G.tocsc(), C.tocsc(), B.tocsc(), L.tocsc(), output_labels)


def extract_state_space(G, C, B, L):
    """Convert (G + sC) descriptor form to state-space (E, A, B, C, D).

    Returns
    -------
    E, A, B, C, D : sparse matrices
        E * dx/dt = A * x + B * u
        y         = C * x + D * u
    """
    E = C.copy()
    A = -G.copy()
    C_out = L.T.copy() if L.nnz > 0 else sp.csc_matrix((0, G.shape[0]))

    from scipy.sparse import csc_matrix
    D = csc_matrix((C_out.shape[0], B.shape[1]), dtype=np.float64)

    return E, A, B, C_out, D
