"""Balanced Truncation (BT) model order reduction.

Truncates states with small Hankel singular values while preserving
stability.  Operates on the state-space descriptor form:

    E * dx/dt = A * x + B * u
    y         = C * x

Reference:
    Moore, "Principal component analysis in linear systems:
    controllability, observability, and model reduction,"
    IEEE TAC, 1981.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import sparse as sp
from scipy.linalg import svd, solve_continuous_lyapunov as lyap
from scipy.sparse.linalg import spsolve

from .base import MORBase, ReducedModel


class BalancedTruncation(MORBase):
    """Balanced truncation MOR for stable LTI descriptor systems."""

    def __init__(self):
        super().__init__(name="BalancedTruncation")

    def reduce(self, G, C, B, L, order: int, **kwargs) -> ReducedModel:
        """Reduce via balanced truncation.

        For large systems this requires dense Lyapunov solves,
        so it is best suited for systems where n ≲ 2000.
        """
        n = G.shape[0]
        if order >= n:
            raise ValueError(f"Reduced order {order} must be < {n}")

        # Convert to state-space: E*dx/dt = A*x + B*u, y = C*x
        E = C.tocsc()
        A = -G.tocsc()
        B_dense = B.toarray() if hasattr(B, "toarray") else np.asarray(B)
        C_dense = L.T.toarray() if hasattr(L.T, "toarray") else np.asarray(L.T)

        # E must be invertible for standard BT
        try:
            E_inv = sp.linalg.inv(E)
        except Exception:
            raise ValueError(
                "Balanced truncation requires invertible E (C) matrix. "
                "Use PRIMA or POD instead."
            )

        # Standard system: dx/dt = E^{-1}*A*x + E^{-1}*B*u
        E_inv_dense = E_inv.toarray() if hasattr(E_inv, "toarray") else np.asarray(E_inv)
        A_tilde = E_inv_dense @ A.toarray()
        B_tilde = E_inv_dense @ B_dense
        C_tilde = C_dense

        # Solve Lyapunov equations for controllability and observability Gramians
        # A*P + P*A' + B*B' = 0
        # A'*Q + Q*A + C'*C = 0
        try:
            P = lyap(A_tilde, B_tilde @ B_tilde.T)
            Q = lyap(A_tilde.T, C_tilde.T @ C_tilde)
        except Exception as e:
            raise ValueError(f"Lyapunov solve failed: {e}. "
                             "System may be unstable or poorly conditioned.")

        # Compute Hankel singular values via Cholesky factors
        Rp = np.linalg.cholesky(P)
        Rq = np.linalg.cholesky(Q)
        U, S, Vt = svd(Rq.T @ Rp, full_matrices=False)

        # Transformation matrices
        T1 = np.linalg.solve(Rp.T, Vt.T)  # actually T^{-1}
        # Actually let's use the standard square-root method properly
        S_diag = np.diag(S)
        # Ti = Rp.T @ Vt.T @ np.diag(1.0 / np.sqrt(S))
        Ti = (Rp.T @ Vt.T) @ np.diag(1.0 / np.sqrt(S + 1e-30))
        # To = np.diag(1.0 / np.sqrt(S + 1e-30)) @ U.T @ Rq.T
        To = np.diag(1.0 / np.sqrt(S + 1e-30)) @ U.T @ Rq.T

        hsv = S.copy()

        # Project onto first 'order' states
        T_left = To[:order, :]
        T_right = Ti[:, :order]

        Ar = T_left @ A_tilde @ T_right
        Br = T_left @ B_tilde
        Cr = C_tilde @ T_right
        Dr = np.zeros((C_tilde.shape[0], B_tilde.shape[1]))

        # Map back to descriptor form
        # Since we transformed x = T_right * xr, the reduced system is:
        #   dxr/dt = Ar * xr + Br * u
        #   y      = Cr * xr
        # In descriptor form: Ir * dxr/dt = Ar * xr + Br * u
        Ir = np.eye(order)
        # Reduced G, C in (G + sC) form:
        # C * dx/dt = A * x + B*u  →  Cr = Ir, Gr = -Ar
        Gr = -Ar
        Cr = Ir

        Q_mat = T_right  # projection matrix

        return ReducedModel(
            Gr=sp.csc_matrix(Gr),
            Cr=sp.csc_matrix(Cr),
            Br=sp.csc_matrix(Br),
            Lr=sp.csc_matrix(Cr.T),  # output matrix
            Q=Q_mat,
            reduced_order=order,
            info={
                "algorithm": "BalancedTruncation",
                "hsv": hsv,
            },
        )
