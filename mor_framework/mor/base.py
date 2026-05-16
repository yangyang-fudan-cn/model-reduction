"""Abstract base class for all MOR algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import sparse as sp


@dataclass
class ReducedModel:
    """Container for a reduced-order model.

    The reduced descriptor system is:
        (Gr + s*Cr) * xr = Br * u
        y                = Lrᵀ * xr
    """
    Gr: sp.csc_matrix
    Cr: sp.csc_matrix
    Br: sp.csc_matrix
    Lr: sp.csc_matrix
    Q: np.ndarray          # projection matrix (full_state × reduced_state)
    reduced_order: int
    info: Dict = field(default_factory=dict)


class MORBase(ABC):
    """Base class for model order reduction algorithms."""

    def __init__(self, name: str = "generic"):
        self.name = name

    @abstractmethod
    def reduce(self, G, C, B, L, order: int, **kwargs) -> ReducedModel:
        """Reduce the descriptor system (G, C, B, L) to the given order.

        Parameters
        ----------
        G : sparse matrix (n × n)
        C : sparse matrix (n × n)
        B : sparse matrix (n × m)
        L : sparse matrix (n × p)
        order : int — target reduced order (must be < n)
        **kwargs : algorithm-specific options

        Returns
        -------
        ReducedModel containing projected matrices.
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.name}>"
