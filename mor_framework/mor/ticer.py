"""TICER: Time-Constant Equilibration Reduction for RC circuits.

Eliminates internal nodes with small time constants using star-mesh
transformation and capacitance redistribution.

Reference:
    B. N. Sheehan, "TICER: Realizable reduction of extracted RC circuits,"
    IEEE/ACM ICCAD, 1999.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse as sp

from .base import MORBase, ReducedModel


def _estimate_n_nodes(G, C):
    """Estimate the number of node-voltage variables in the MNA system.

    In MNA the first ``n_nodes`` variables are node voltages, followed by
    V-source equation variables, then inductor current variables.

    A V-source equation row has: C[row,:] ≈ 0, G[row,row] ≈ 0, and
    off-diagonal entries that are exactly ±1 (the V-source incidence).
    Inductor rows have non-zero C on the diagonal (-L value).

    Returns
    -------
    n_nodes : int  — number of node-voltage variables.
    """
    n = G.shape[0]
    G_dense = G.toarray() if hasattr(G, "toarray") else np.asarray(G)
    C_dense = C.toarray() if hasattr(C, "toarray") else np.asarray(C)

    for row in range(n):
        # V-source equation rows have zero C and zero diagonal G
        if np.any(np.abs(C_dense[row, :]) > 1e-30):
            continue
        if abs(G_dense[row, row]) > 1e-30:
            continue
        # Off-diagonals are exactly ±1 (V-source incidence pattern)
        off_diag = np.delete(G_dense[row, :], row)
        mask = np.abs(off_diag) > 1e-30
        if not np.any(mask):
            continue
        if np.all(np.abs(np.abs(off_diag[mask]) - 1.0) < 1e-10):
            return row

    return n  # All rows appear to be node voltages


def _identify_port_nodes(G, B, L, n_nodes: int):
    """Identify which node indices (0..n_nodes-1) are ports.

    Ports are nodes that connect to voltage/current sources or are
    observed as outputs — they must be preserved during reduction.

    Only indices 0 .. n_nodes-1 are considered (actual node voltages).
    V-source equation variables and inductor current variables are
    never port *nodes*.
    """
    port_nodes: set = set()

    # Nodes connected to current sources (B[row, col] != 0 for row < n_nodes)
    B_dense = B.toarray() if hasattr(B, "toarray") else np.asarray(B)
    for row in range(min(n_nodes, B.shape[0])):
        if np.any(np.abs(B_dense[row, :]) > 1e-30):
            port_nodes.add(row)

    # Nodes connected to voltage sources.
    # V-source equations occupy rows n_nodes .. n_nodes+n_vsrc-1.
    # G[row, col] = ±1.0 where col < n_nodes is the V-source's node.
    G_dense = G
    if hasattr(G, "toarray"):
        G_dense = G.toarray()
    for row in range(n_nodes, G.shape[0]):
        for col in range(n_nodes):
            if abs(G_dense[row, col]) > 1e-30:
                port_nodes.add(col)

    # Nodes that appear in the output matrix L
    L_dense = L.toarray() if hasattr(L, "toarray") else np.asarray(L)
    for row in range(min(n_nodes, L.shape[0])):
        if np.any(np.abs(L_dense[row, :]) > 1e-30):
            port_nodes.add(row)

    return port_nodes


class TICER(MORBase):
    """TICER model order reduction for RC circuits.

    Eliminates internal nodes whose time constant τ = C_kk / G_kk
    falls below ``threshold``.  Port nodes (connected to sources or
    observed as outputs) are always preserved.

    The algorithm applies star-mesh conductance transformations and
    capacitance redistribution to produce a reduced RC network that
    remains realizable as an RC circuit.
    """

    def __init__(self, threshold: float = 1e-12):
        super().__init__(name="TICER")
        self.threshold = threshold

    def reduce(self, G, C, B, L, order=None, **kwargs):
        """Reduce the RC descriptor system via TICER.

        Parameters
        ----------
        G, C : sparse (n × n) — descriptor system matrices
        B : sparse (n × m) — input matrix
        L : sparse (n × p) — output matrix
        order : ignored (TICER eliminates based on time-constant threshold)
        threshold : float — time-constant threshold in seconds (default: 1e-12)

        Returns
        -------
        ReducedModel containing the reduced RC system.
        """
        threshold = kwargs.get("threshold", self.threshold)

        n = G.shape[0]
        n_nodes = _estimate_n_nodes(G, C)

        # Identify port nodes (only among the node-voltage variables)
        port_nodes = _identify_port_nodes(G, B, L, n_nodes)

        # Work with dense arrays for in-place modification
        G_dense = G.toarray() if hasattr(G, "toarray") else np.asarray(G).copy()
        C_dense = C.toarray() if hasattr(C, "toarray") else np.asarray(C).copy()

        # Active set — only node-voltage indices (0 .. n_nodes-1) can be
        # candidates for TICER elimination.  V-source / inductor variables
        # (n_nodes .. n-1) are always preserved.
        active = set(range(n_nodes))

        # Iteratively eliminate nodes with the smallest time constant
        while True:
            candidates: list = []
            for k in active:
                if k in port_nodes:
                    continue
                G_kk = G_dense[k, k]
                if G_kk <= 0.0:
                    continue
                C_kk = C_dense[k, k]
                tau_k = C_kk / G_kk
                candidates.append((tau_k, k))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[0])
            tau_k, k = candidates[0]

            if tau_k > threshold:
                break

            # ---- Eliminate node k ----
            neighbors = sorted(
                i for i in active if i != k and abs(G_dense[i, k]) > 1e-30
            )

            if not neighbors:
                active.discard(k)
                continue

            G_kk = G_dense[k, k]
            C_kk = C_dense[k, k]

            # Star-mesh transformation + capacitance redistribution
            for ii, i in enumerate(neighbors):
                g_ik = -G_dense[i, k]  # conductance between i and k

                # 1. Remove conductance to eliminated node k from G[i,i]
                G_dense[i, i] -= g_ik
                G_dense[i, k] = 0.0
                G_dense[k, i] = 0.0

                # 2. Distribute capacitance from node k to neighbor i
                if G_kk > 0.0:
                    C_dense[i, i] += C_kk * g_ik / G_kk

                # 3. Add star-mesh conductances between i and other neighbors
                for jj in range(ii + 1, len(neighbors)):
                    j = neighbors[jj]
                    g_jk = -G_dense[j, k]
                    if G_kk <= 0.0:
                        continue
                    g_new = g_ik * g_jk / G_kk

                    if g_new <= 0.0:
                        continue

                    G_dense[i, i] += g_new
                    G_dense[j, j] += g_new
                    G_dense[i, j] -= g_new
                    G_dense[j, i] -= g_new

            active.discard(k)
            G_dense[k, :] = 0.0
            G_dense[:, k] = 0.0
            C_dense[k, :] = 0.0
            C_dense[:, k] = 0.0

        # --- Build reduced system ---
        # Preserve all non-node variables (V-source / inductor equations)
        branch_indices = list(range(n_nodes, n))
        reduced_indices = sorted(active) + branch_indices
        r = len(reduced_indices)

        if r == 0:
            raise ValueError("TICER eliminated all nodes — check threshold / circuit.")

        Gr = G_dense[np.ix_(reduced_indices, reduced_indices)]
        Cr = C_dense[np.ix_(reduced_indices, reduced_indices)]

        # Projection matrix (selection): x ≈ Q @ xr
        Q = np.zeros((n, r))
        for new_idx, old_idx in enumerate(reduced_indices):
            Q[old_idx, new_idx] = 1.0

        Br = Q.T @ (B.toarray() if hasattr(B, "toarray") else np.asarray(B))
        Lr = Q.T @ (L.toarray() if hasattr(L, "toarray") else np.asarray(L))

        return ReducedModel(
            Gr=sp.csc_matrix(Gr),
            Cr=sp.csc_matrix(Cr),
            Br=sp.csc_matrix(Br),
            Lr=sp.csc_matrix(Lr),
            Q=Q,
            reduced_order=r,
            info={
                "algorithm": "TICER",
                "threshold": threshold,
                "n_original": n_nodes,
                "n_eliminated": n_nodes - len(active),
                "port_nodes": sorted(port_nodes),
            },
        )
