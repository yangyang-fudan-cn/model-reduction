"""AC (frequency-domain) analysis for descriptor systems.

Solves the frequency-domain system

    (G + j*ω*C) * X = B * U

for a set of frequency points and computes the output Y = Lᵀ * X.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import sparse as sp
from scipy.sparse.linalg import spsolve, factorized


def ac_analysis(
    G, C, B, L,
    f_start: float = 1.0,
    f_stop: float = 1e9,
    n_points: int = 100,
    sweep_type: str = "dec",
    input_idx: int = 0,
    input_func: Optional[callable] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run AC (frequency sweep) analysis.

    Parameters
    ----------
    G, C, B, L : sparse matrices — system in (G + sC) form
    f_start : start frequency (Hz)
    f_stop : stop frequency (Hz)
    n_points : number of frequency points
    sweep_type : "dec" (decade), "lin" (linear), or "oct" (octave)
    input_idx : which input to excite (default 0)
    input_func : optional function f(freq) → input scaling

    Returns
    -------
    freqs : ndarray (n_points,) — frequency points
    Y : ndarray (n_outputs, n_points) — frequency response (complex)
    """
    G_sp = G.tocsc() if hasattr(G, "tocsc") else sp.csc_matrix(G)
    C_sp = C.tocsc() if hasattr(C, "tocsc") else sp.csc_matrix(C)
    B_arr = B.toarray() if hasattr(B, "toarray") else np.asarray(B)
    L_arr = L.toarray() if hasattr(L, "toarray") else np.asarray(L)
    n_outputs = L_arr.shape[1]

    # Generate frequency points
    if sweep_type.lower() == "dec":
        freqs = np.logspace(np.log10(f_start), np.log10(f_stop), n_points)
    elif sweep_type.lower() == "oct":
        n_oct = np.log2(f_stop / f_start)
        freqs = f_start * (2 ** np.linspace(0, n_oct, n_points))
    else:
        freqs = np.linspace(f_start, f_stop, n_points)

    Y = np.zeros((n_outputs, len(freqs)), dtype=np.complex128)

    u = np.zeros(B_arr.shape[1])
    u[input_idx] = 1.0

    for i, f in enumerate(freqs):
        omega = 2.0 * np.pi * f
        A = G_sp + 1j * omega * C_sp
        try:
            x = spsolve(A, B_arr @ u)
        except Exception:
            # Use dense solve as fallback
            A_dense = A.toarray() if hasattr(A, "toarray") else np.asarray(A)
            x = np.linalg.solve(A_dense, B_arr @ u)
        Y[:, i] = L_arr.T @ x

    return freqs, Y
