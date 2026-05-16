"""Proper Orthogonal Decomposition (POD) model order reduction.

POD computes a reduced basis from snapshot data via singular-value
decomposition (SVD), then projects the system matrices onto that basis.

This implementation collects snapshots from a transient simulation of the
full-order model.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import sparse as sp
from scipy.linalg import svd

from .base import MORBase, ReducedModel
from ..simulation.transient import simulate_transient


class POD(MORBase):
    """POD-based model order reduction."""

    def __init__(self, snapshot_dt: Optional[float] = None,
                 t_end: Optional[float] = None):
        super().__init__(name="POD")
        self.snapshot_dt = snapshot_dt
        self.t_end = t_end

    def reduce(self, G, C, B, L, order: int, **kwargs) -> ReducedModel:
        """Reduce via POD.

        Parameters
        ----------
        G, C, B, L : system matrices in (G + sC) form
        order : target reduced order
        **kwargs : may include:
            snapshot_dt : time step for snapshot collection
            t_end : end time for snapshot simulation
            input_func : function f(t) → input vector
        """
        if order >= G.shape[0]:
            raise ValueError(f"Reduced order {order} must be < {G.shape[0]}")

        snapshot_dt = kwargs.get("snapshot_dt", self.snapshot_dt or 1e-9)
        t_end = kwargs.get("t_end", self.t_end or 1e-6)

        # Run a transient simulation to collect snapshots
        n_state = G.shape[0]
        n_tsteps = max(2, int(t_end / snapshot_dt))
        t_span = (0.0, t_end)

        # Use input function from kwargs or default pulse
        input_func = kwargs.get("input_func", None)

        sol, tout = simulate_transient(
            G, C, B, t_span, n_tsteps,
            input_func=input_func,
        )
        # sol: (n_state × n_tsteps+1)

        # Build snapshot matrix
        snaps = sol.copy()

        # SVD
        U, S, Vt = svd(snaps, full_matrices=False)

        energy = np.cumsum(S ** 2) / np.sum(S ** 2)
        n_keep = min(order, U.shape[1])
        Q = U[:, :n_keep]

        # Project
        Gr = Q.T @ (G @ Q)
        Cr = Q.T @ (C @ Q)
        Br = Q.T @ B.toarray()
        Lr = Q.T @ L.toarray()

        return ReducedModel(
            Gr=sp.csc_matrix(Gr),
            Cr=sp.csc_matrix(Cr),
            Br=sp.csc_matrix(Br),
            Lr=sp.csc_matrix(Lr),
            Q=Q,
            reduced_order=n_keep,
            info={
                "algorithm": "POD",
                "singular_values": S[:n_keep],
                "energy_ratio": float(energy[n_keep - 1]) if n_keep > 0 else 0.0,
            },
        )
