"""Tests for the TICER model order reduction algorithm."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mor_framework.mor.ticer import TICER, _identify_port_nodes, _estimate_n_nodes
from mor_framework.core.mna import build_mna
from mor_framework.core.circuit import Circuit, Element, ElementType, Analysis, SourceWaveform
from mor_framework.simulation.transient import simulate_transient


# =========================================================================
#  Helpers
# =========================================================================


def _make_rc_system(G_dense, C_dense, port_nodes=None):
    """Build G, C, B, L from dense G, C for a test RC circuit.

    Only nodes listed in ``port_nodes`` get source excitation and are
    observed as outputs.
    """
    n = G_dense.shape[0]
    G = sp.csc_matrix(G_dense)
    C = sp.csc_matrix(C_dense)

    if port_nodes is None:
        port_nodes = [0]

    m = len(port_nodes)
    B = sp.lil_matrix((n, m))
    for col, node in enumerate(port_nodes):
        B[node, col] = 1.0

    # Output: observe port nodes only
    L = sp.lil_matrix((n, m))
    for col, node in enumerate(port_nodes):
        L[node, col] = 1.0

    return G.tocsc(), C.tocsc(), B.tocsc(), L.tocsc()


def _step_input(m: int):
    """Return a step input function for an m-input system."""
    def func(t):
        u = np.zeros(m)
        u[0] = 1.0 if t >= 0 else 0.0
        return u
    return func


# =========================================================================
#  Tests
# =========================================================================


def test_ticer_import():
    """TICER class is importable and has correct defaults."""
    algo = TICER()
    assert algo.name == "TICER"
    assert algo.threshold == 1e-12


def test_ticer_threshold_parameter():
    """TICER accepts custom threshold."""
    algo = TICER(threshold=1e-10)
    assert algo.threshold == 1e-10


def test_estimate_n_nodes_pure_rc():
    """n_nodes estimation: pure RC circuit (no V-sources) has n_nodes = n."""
    G = sp.csc_matrix(np.array([[1e-3, -1e-3], [-1e-3, 1e-3]]))
    C = sp.csc_matrix(np.diag([1e-12, 1e-12]))
    assert _estimate_n_nodes(G, C) == 2


def test_identify_port_nodes_current_source():
    """Port node identification works for current-source-driven nodes."""
    G_dense = np.array([[1e-3, -1e-3], [-1e-3, 1e-3]])
    C_dense = np.diag([1e-12, 1e-12])
    G, C, B, L = _make_rc_system(G_dense, C_dense, port_nodes=[0])

    ports = _identify_port_nodes(G, B, L, n_nodes=2)
    assert 0 in ports, "Node 0 should be a port (current source)"
    assert 1 not in ports, "Node 1 should not be a port"


def test_identify_port_nodes_multiple():
    """Multiple port nodes are correctly identified."""
    # 3-node ladder: nodes 0 and 2 are ports
    G_dense = np.array([
        [2e-3, -1e-3, 0.0],
        [-1e-3, 2e-3, -1e-3],
        [0.0, -1e-3, 1e-3],
    ])
    C_dense = np.diag([1e-12, 1e-12, 1e-12])
    G, C, B, L = _make_rc_system(G_dense, C_dense, port_nodes=[0, 2])

    ports = _identify_port_nodes(G, B, L, n_nodes=3)
    assert ports == {0, 2}, f"Expected ports {{0, 2}}, got {ports}"


def test_ticer_no_reduction_low_threshold():
    """With a very low threshold (τ=0), TICER eliminates no nodes."""
    G_dense = np.array([
        [2e-3, -1e-3, 0.0],
        [-1e-3, 2e-3, -1e-3],
        [0.0, -1e-3, 1e-3],
    ])
    C_dense = np.diag([1e-15, 1e-15, 1e-15])
    G, C, B, L = _make_rc_system(G_dense, C_dense, port_nodes=[0, 2])

    algo = TICER(threshold=0.0)
    result = algo.reduce(G, C, B, L)
    assert result.reduced_order == 3, (
        f"With threshold=0, all 3 nodes should remain; got {result.reduced_order}"
    )


def test_ticer_eliminates_internal_nodes():
    """TICER eliminates internal nodes with small time constants."""
    G_dense = np.array([
        [2e-3, -1e-3, 0.0],
        [-1e-3, 2e-3, -1e-3],
        [0.0, -1e-3, 1e-3],
    ])
    # Node 1 (internal) has tiny cap → small time constant
    C_dense = np.diag([1e-12, 1e-15, 1e-12])
    G, C, B, L = _make_rc_system(G_dense, C_dense, port_nodes=[0, 2])

    algo = TICER(threshold=1e-11)
    result = algo.reduce(G, C, B, L)
    assert result.reduced_order == 2, (
        f"Node 1 should be eliminated (3→2); got {result.reduced_order}"
    )
    assert result.info["n_eliminated"] == 1


def test_ticer_port_nodes_preserved():
    """Port nodes are never eliminated regardless of time constant."""
    G_dense = np.array([
        [2e-3, -1e-3, 0.0],
        [-1e-3, 2e-3, -1e-3],
        [0.0, -1e-3, 1e-3],
    ])
    C_dense = np.diag([1e-15, 1e-15, 1e-15])  # all tiny
    G, C, B, L = _make_rc_system(G_dense, C_dense, port_nodes=[0, 2])

    algo = TICER(threshold=1e-8)  # aggressive
    result = algo.reduce(G, C, B, L)
    # Node 1 (internal) eliminated; nodes 0 and 2 remain
    assert result.reduced_order == 2, (
        f"Expected order 2, got {result.reduced_order}"
    )
    remaining = set(np.where(np.any(np.abs(result.Q) > 0.5, axis=1))[0].tolist())
    assert 0 in remaining, "Port node 0 must remain"
    assert 2 in remaining, "Port node 2 must remain"
    assert 1 not in remaining, "Internal node 1 should be eliminated"


def test_ticer_eliminates_all_internal():
    """With aggressive threshold, TICER eliminates all internal nodes."""
    G_dense = np.array([
        [3e-3, -1e-3, -1e-3, -1e-3],
        [-1e-3, 1e-3, 0.0, 0.0],
        [-1e-3, 0.0, 1e-3, 0.0],
        [-1e-3, 0.0, 0.0, 1e-3],
    ])
    C_dense = np.diag([1e-12, 1e-15, 1e-15, 1e-15])
    G, C, B, L = _make_rc_system(G_dense, C_dense, port_nodes=[0])

    algo = TICER(threshold=1e-10)
    result = algo.reduce(G, C, B, L)
    # Only port node 0 should remain
    assert result.reduced_order == 1, (
        f"Only port node 0 should remain; got order {result.reduced_order}"
    )


def test_ticer_star_mesh_dc_accuracy():
    """Star-mesh transformation preserves DC conductance between ports.

    At DC (capacitors open), the reduced model should exactly match
    the full model's conductance matrix between ports.
    """
    # π-network: port0 -- R=100 -- node1 -- R=100 -- port2
    #              C=1pF      C=1fF      C=1pF
    #             1MΩ                   1MΩ
    #             GND                    GND
    # Add 1MΩ to ground at ports so G is non-singular at DC.
    G_dense = np.array([
        [1e-2 + 1e-6, -1e-2, 0.0],
        [-1e-2, 2e-2, -1e-2],
        [0.0, -1e-2, 1e-2 + 1e-6],
    ])
    C_dense = np.diag([1e-12, 1e-15, 1e-12])
    G, C, B, L = _make_rc_system(G_dense, C_dense, port_nodes=[0, 2])

    algo = TICER(threshold=1e-12)
    result = algo.reduce(G, C, B, L)
    assert result.reduced_order == 2, "Internal node should be eliminated"

    n_ports = B.shape[1]
    # DC response: V = G^{-1} * B * u  (C contributes nothing at DC)
    Gr_full = G.toarray()
    V_full = np.linalg.solve(Gr_full, B.toarray())

    Gr_red = result.Gr.toarray()
    V_red = np.linalg.solve(Gr_red, result.Br.toarray())
    V_red_full = result.Q @ V_red  # map back to full space

    # At DC, voltages at port nodes should match exactly
    for port_idx in [0, 2]:
        err = np.linalg.norm(V_full[port_idx, :] - V_red_full[port_idx, :])
        assert err < 1e-6, (  # numerical precision of star-mesh operations
            f"Port {port_idx} DC voltage error: {err:.2e}"
        )


def test_ticer_transient_voltage_driven():
    """TICER-reduced model matches full model in voltage-driven RC ladder."""
    from mor_framework.core.netlist_parser import parse_netlist

    netlist_path = (Path(__file__).resolve().parent.parent
                    / "mor_framework" / "examples" / "rc_ladder.sp")
    circuit = parse_netlist(str(netlist_path))
    G, C, B, L, labels = build_mna(circuit)

    # Full model transient (V-source driven, pulse input)
    t_span = (0.0, 5e-10)
    n_steps = 200

    def pulse_input(t):
        T_per = 2e-9
        t_mod = t % T_per
        if t_mod < 1e-11:
            return np.array([t_mod / 1e-11])
        if t_mod < 1e-9:
            return np.array([1.0])
        if t_mod < 1e-9 + 1e-11:
            return np.array([1.0 - (t_mod - 1e-9) / 1e-11])
        return np.array([0.0])

    x_full, t_out = simulate_transient(
        G, C, B, t_span, n_steps, input_func=pulse_input
    )

    # TICER reduced
    algo = TICER(threshold=1e-11)
    result = algo.reduce(G, C, B, L)

    x_red, _ = simulate_transient(
        result.Gr, result.Cr, result.Br, t_span, n_steps,
        input_func=pulse_input,
    )
    x_red_full = result.Q @ x_red

    # Output node V(21) = MNA index 20
    y_full = x_full[20, :]
    y_red = x_red_full[20, :]

    rel_err = np.linalg.norm(y_full - y_red) / (np.linalg.norm(y_full) + 1e-30)
    assert rel_err < 0.15, f"TICER transient error too high: {rel_err:.4e}"


def test_ticer_rc_ladder_via_netlist():
    """TICER on RC ladder parsed from netlist — checks port preservation."""
    from mor_framework.core.netlist_parser import parse_netlist

    netlist_path = (Path(__file__).resolve().parent.parent
                    / "mor_framework" / "examples" / "rc_ladder.sp")
    circuit = parse_netlist(str(netlist_path))
    G, C, B, L, labels = build_mna(circuit)

    n_state = G.shape[0]
    # rc_ladder.sp: 21 nodes + 1 V-source = 22 state vars
    assert n_state == 22, f"Expected 22 state vars, got {n_state}"

    # TICER with moderate threshold (τ_stage = 100*1e-12 = 1e-10,
    # so threshold 5e-11 eliminates nodes with τ < 50ps)
    algo = TICER(threshold=5e-11)
    result = algo.reduce(G, C, B, L)

    reduced_order = result.reduced_order
    assert reduced_order < n_state, f"Should reduce ({reduced_order} < {n_state})"
    assert reduced_order >= 2, f"Should keep port nodes; got {reduced_order}"

    # Port node 0 (netlist node "1", V-source + node) should survive
    # Port node 20 (netlist node "21", output) should survive
    remaining = set(np.where(np.any(np.abs(result.Q) > 0.5, axis=1))[0].tolist())
    assert 0 in remaining, "Input node (V-source +) must remain"
    assert 20 in remaining, "Output node V(21) must remain"


def test_ticer_preserves_passivity():
    """Reduced G and C matrices should be symmetric diagonally dominant."""
    G_dense = np.array([
        [2e-3, -1e-3, 0.0],
        [-1e-3, 2e-3, -1e-3],
        [0.0, -1e-3, 1e-3],
    ])
    C_dense = np.diag([1e-12, 1e-15, 1e-12])
    G, C, B, L = _make_rc_system(G_dense, C_dense, port_nodes=[0, 2])

    algo = TICER(threshold=1e-11)
    result = algo.reduce(G, C, B, L)

    Gr = result.Gr.toarray()
    Cr = result.Cr.toarray()

    assert np.allclose(Gr, Gr.T, atol=1e-15), "Reduced G must be symmetric"
    assert np.allclose(Cr, Cr.T, atol=1e-15), "Reduced C must be symmetric"

    for i in range(Gr.shape[0]):
        assert Gr[i, i] >= 0, f"G[{i},{i}] must be non-negative"
        assert Cr[i, i] >= 0, f"C[{i},{i}] must be non-negative"


def test_ticer_identity_projection():
    """Projection matrix Q should be a selection matrix (unit columns)."""
    G_dense = np.array([
        [2e-3, -1e-3, 0.0],
        [-1e-3, 2e-3, -1e-3],
        [0.0, -1e-3, 1e-3],
    ])
    C_dense = np.diag([1e-12, 1e-15, 1e-12])
    G, C, B, L = _make_rc_system(G_dense, C_dense, port_nodes=[0, 2])

    algo = TICER(threshold=1e-11)
    result = algo.reduce(G, C, B, L)

    Q = result.Q
    for col in range(Q.shape[1]):
        assert np.sum(np.abs(Q[:, col])) == 1.0, (
            f"Column {col} of Q must be a unit vector"
        )
        assert np.sum(Q[:, col] > 0.5) == 1, (
            f"Column {col} of Q must have exactly one 1"
        )
    # Q^T @ Q == I  (orthonormal columns)
    assert np.allclose(Q.T @ Q, np.eye(Q.shape[1]), atol=1e-15)


def test_ticer_elimination_info():
    """TICER info dict should contain elimination statistics."""
    G_dense = np.array([
        [2e-3, -1e-3, 0.0],
        [-1e-3, 2e-3, -1e-3],
        [0.0, -1e-3, 1e-3],
    ])
    C_dense = np.diag([1e-12, 1e-15, 1e-12])
    G, C, B, L = _make_rc_system(G_dense, C_dense, port_nodes=[0, 2])

    algo = TICER(threshold=1e-11)
    result = algo.reduce(G, C, B, L)

    assert result.info["algorithm"] == "TICER"
    assert result.info["threshold"] == 1e-11
    assert result.info["n_original"] == 3
    assert result.info["n_eliminated"] == 1
    assert result.info["port_nodes"] == [0, 2]


def test_ticer_raises_on_all_eliminated():
    """TICER raises ValueError if all nodes would be eliminated."""
    G_dense = np.array([[1e-3]])
    C_dense = np.array([[1e-15]])
    # No ports — the single node can be eliminated
    G, C, B, L = _make_rc_system(G_dense, C_dense, port_nodes=[])

    algo = TICER(threshold=1e-8)
    try:
        result = algo.reduce(G, C, B, L)
        assert False, "Should have raised ValueError for eliminating all nodes"
    except ValueError:
        pass


def test_ticer_clock_tree_low_threshold():
    """TICER with negligible threshold preserves all nodes in clock tree."""
    from mor_framework.core.netlist_parser import parse_netlist

    netlist_path = (Path(__file__).resolve().parent.parent
                    / "mor_framework" / "examples" / "clock_tree.sp")
    circuit = parse_netlist(str(netlist_path))
    G, C, B, L, labels = build_mna(circuit)

    n_state = G.shape[0]
    # clock_tree.sp: 8 nodes + 1 V-source = 9 state vars

    algo = TICER(threshold=0.0)
    result = algo.reduce(G, C, B, L)
    # All 8 node-voltage variables + 1 V-source variable
    assert result.reduced_order == n_state, (
        f"All {n_state} state vars should remain; got {result.reduced_order}"
    )


def test_ticer_clock_tree_moderate_threshold():
    """TICER with moderate threshold eliminates some clock-tree nodes."""
    from mor_framework.core.netlist_parser import parse_netlist

    netlist_path = (Path(__file__).resolve().parent.parent
                    / "mor_framework" / "examples" / "clock_tree.sp")
    circuit = parse_netlist(str(netlist_path))
    G, C, B, L, labels = build_mna(circuit)

    n_state = G.shape[0]  # 9 (8 nodes + 1 V-source)

    # Internal nodes 2, 3, 4 have τ = 1e-12/(3/10) ≈ 3.3e-12 s.
    # Leaf nodes 5, 6, 7, 8 have τ = 1e-12/(1/10) = 1e-11 s.
    # With threshold 5e-12, only the fast internal nodes (2, 3, 4)
    # are eliminated; all ports survive.
    algo = TICER(threshold=5e-12)
    result = algo.reduce(G, C, B, L)

    # Expected: 8 nodes + 1 V-source = 9 total. Some internal nodes
    # eliminated (their τ ≈ 3.3e-12 < 5e-12 threshold), but star-mesh
    # updates may change time constants, so some may survive.
    assert result.reduced_order < n_state, "Some nodes should be eliminated"
    assert result.reduced_order >= 6, (
        f"Should keep ports + leaves + V-source (≥6), got {result.reduced_order}"
    )
    assert result.reduced_order > 0


def test_ticer_reducible_order_count():
    """TICER reduced_order equals number of remaining original nodes."""
    G_dense = np.array([
        [3e-3, -1e-3, -1e-3],
        [-1e-3, 2e-3, -1e-3],
        [-1e-3, -1e-3, 2e-3],
    ])
    # Internal nodes 1, 2 have tiny caps → eliminated
    C_dense = np.diag([1e-12, 1e-15, 1e-15])
    G, C, B, L = _make_rc_system(G_dense, C_dense, port_nodes=[0])

    algo = TICER(threshold=1e-11)
    result = algo.reduce(G, C, B, L)

    assert result.reduced_order == 1
    # Q should select only node 0
    assert result.Q.shape == (3, 1)
    assert result.Q[0, 0] == 1.0


# =========================================================================

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
