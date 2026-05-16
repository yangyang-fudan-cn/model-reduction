"""Transient (time-domain) simulation for descriptor systems.

Solves the descriptor system

    C * dx/dt = -G * x + B * u(t)

using the Backward Euler (BE) or Trapezoidal (TRAP) integration method.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve, factorized


def simulate_transient(
    G, C, B,
    t_span: Tuple[float, float],
    n_steps: int,
    input_func: Optional[Callable[[float], np.ndarray]] = None,
    method: str = "be",
    x0: Optional[np.ndarray] = None,
    output_func: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run transient simulation of the descriptor system.

    The system is:
        C * dx/dt = -G * x + B * u(t)

    with Backward-Euler (BE, default) integration:
        (C/dt + G) * x_{k+1} = (C/dt) * x_k + B * u(t_{k+1})

    Parameters
    ----------
    G, C, B : sparse matrices — system in (G + sC) form
    t_span : (t_start, t_end)
    n_steps : number of time steps
    input_func : callable f(t) → ndarray of shape (n_inputs,) or None
    method : "be" (backward Euler) or "trap" (trapezoidal)
    x0 : initial state vector or None (zero)
    output_func : if True also return system output

    Returns
    -------
    x_sol : ndarray (n_state × n_steps+1) — state trajectory
    t_out : ndarray (n_steps+1,) — time points
    """
    t_start, t_end = t_span
    dt = (t_end - t_start) / n_steps
    n_state = G.shape[0]
    n_inputs = B.shape[1]

    if x0 is None:
        x0 = np.zeros(n_state)

    if input_func is None:
        # Default: step on first input
        def input_func(t):
            u = np.zeros(n_inputs)
            u[0] = 1.0 if t >= 0 else 0.0
            return u

    # Pre-factor the system matrix for BE
    C_sp = sp.csc_matrix(C) if not isinstance(C, sp.spmatrix) else C.tocsc()
    G_sp = sp.csc_matrix(G) if not isinstance(G, sp.spmatrix) else G.tocsc()
    B_arr = B.toarray() if hasattr(B, "toarray") else np.asarray(B)

    A_be = C_sp / dt + G_sp

    # Factor once
    try:
        solve_be = factorized(A_be)
    except Exception:
        # Fall back to splu
        A_be_lu = sp.linalg.splu(A_be.tocsc())
        solve_be = lambda rhs: A_be_lu.solve(rhs)

    t_out = np.linspace(t_start, t_end, n_steps + 1)
    x_sol = np.zeros((n_state, n_steps + 1))
    x_sol[:, 0] = x0

    if method == "trap":
        # C*(x_{k+1} - x_k)/dt = -0.5*G*(x_{k+1} + x_k) + 0.5*B*(u_{k+1} + u_k)
        # (C/dt + G/2) * x_{k+1} = (C/dt - G/2) * x_k + B * (u_{k+1} + u_k)/2
        A_trap = C_sp / dt + G_sp / 2.0
        A_trap_lu = sp.linalg.splu(A_trap.tocsc())
        B_over2 = B_arr / 2.0

    for k in range(n_steps):
        t_cur = t_out[k + 1]
        u_cur = input_func(t_cur)
        rhs = (C_sp @ x_sol[:, k]) / dt + B_arr @ u_cur

        if method == "trap":
            u_k = input_func(t_out[k])
            u_avg = (u_cur + u_k) / 2.0
            rhs = (C_sp @ x_sol[:, k]) / dt - (G_sp @ x_sol[:, k]) / 2.0 + B_arr @ u_avg
            x_sol[:, k + 1] = A_trap_lu.solve(rhs)
        else:
            x_sol[:, k + 1] = solve_be(rhs)

    return x_sol, t_out
