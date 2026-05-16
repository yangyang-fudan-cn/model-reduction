"""PRIMA: Passive Reduced-order Interconnect Macromodeling Algorithm.

Block Arnoldi Krylov-subspace method that preserves passivity.

Reference:
    Odabasioglu, Celik, Pileggi. "PRIMA: passive reduced-order interconnect
    macromodeling algorithm." IEEE TCAD, 1998.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy import sparse as sp
from scipy.linalg import qr
from scipy.sparse.linalg import factorized, splu

from .base import MORBase, ReducedModel


def _block_arnoldi(G, C, B, s0: float, k: int, reorth: bool = True):
    """Block Arnoldi process for PRIMA.

    Generates orthonormal basis Q for the block Krylov subspace
        K_r((G+s0*C)^{-1}*C, (G+s0*C)^{-1}*B)

    Parameters
    ----------
    G, C : sparse matrices (n × n)
    B : sparse matrix (n × m_block)
    s0 : expansion frequency point
    k : number of block iterations
    reorth : full reorthogonalization

    Returns
    -------
    Q : ndarray (n × (k * m_block)) — orthonormal basis
    """
    n = G.shape[0]
    m = B.shape[1]

    # Factor (G + s0*C) once
    M = G + s0 * C
    solve = factorized(M.tocsc())  # returns a direct solver

    # R = (G + s0*C)^{-1} * B
    R = np.zeros((n, m))
    B_dense = B.toarray() if hasattr(B, "toarray") else np.asarray(B)
    for j in range(m):
        R[:, j] = solve(B_dense[:, j])

    # QR factor R to get initial block
    Q_blocks = []
    Q_full_list = []

    Q, _ = qr(R, mode="economic")
    Q_blocks.append(Q.copy())
    Q_full_list.append(Q)

    r = m

    for it in range(1, k):
        # V = (G + s0*C)^{-1} * C * Q_{j-1}
        CQ = C @ Q_blocks[-1]
        if hasattr(CQ, "toarray"):
            CQ = CQ.toarray()
        V = np.zeros((n, CQ.shape[1]))
        for j in range(CQ.shape[1]):
            V[:, j] = solve(CQ[:, j].ravel())

        if reorth:
            # Full reorthogonalization against all previous blocks
            for Qj in Q_full_list:
                V = V - Qj @ (Qj.T @ V)

        Qi, Ri = qr(V, mode="economic")
        # Remove near-zero columns
        tol = max(n, m) * np.finfo(float).eps * np.linalg.norm(Ri, ord=2)
        valid = np.abs(np.diag(Ri)) > tol
        if not np.any(valid):
            break
        Qi = Qi[:, valid]

        Q_blocks.append(Qi)
        Q_full_list.append(Qi)
        r += Qi.shape[1]

    Q = np.hstack(Q_full_list) if len(Q_full_list) > 1 else Q_full_list[0]

    # Truncate to requested order
    max_cols = k * m
    if Q.shape[1] > max_cols:
        Q = Q[:, :max_cols]

    return Q


class PRIMA(MORBase):
    """PRIMA model order reduction using block Arnoldi."""

    def __init__(self, expansion_point: float = 0.0):
        super().__init__(name="PRIMA")
        self.expansion_point = expansion_point

    def reduce(self, G, C, B, L, order: int, **kwargs) -> ReducedModel:
        if order >= G.shape[0]:
            raise ValueError(f"Reduced order {order} must be < {G.shape[0]}")

        s0 = kwargs.get("expansion_point", self.expansion_point)
        m = B.shape[1]
        k = max(1, order // m)
        k = min(k, order)

        Q = _block_arnoldi(G, C, B, s0, k)

        # Truncate if we generated more columns than needed
        if Q.shape[1] > order:
            Q = Q[:, :order]

        # Project
        Gr = Q.T @ (G @ Q)
        Cr = Q.T @ (C @ Q)
        Br = Q.T @ B.toarray()
        Lr = Q.T @ L.toarray()

        reduced_order = Q.shape[1]

        return ReducedModel(
            Gr=sp.csc_matrix(Gr),
            Cr=sp.csc_matrix(Cr),
            Br=sp.csc_matrix(Br),
            Lr=sp.csc_matrix(Lr),
            Q=Q,
            reduced_order=reduced_order,
            info={"expansion_point": s0, "algorithm": "PRIMA"},
        )
