"""Circuit data structure for RC/RCL networks."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


class ElementType(Enum):
    RESISTOR = "R"
    CAPACITOR = "C"
    INDUCTOR = "L"
    VSOURCE = "V"
    ISOURCE = "I"


@dataclass
class SourceWaveform:
    """Waveform specification for independent sources."""
    dc_value: float = 0.0
    ac_amplitude: float = 0.0
    ac_phase: float = 0.0
    # Pulse parameters: V1 V2 TD TR TF PW PER
    pulse_params: Optional[Tuple[float, float, float, float, float, float, float]] = None


@dataclass
class Element:
    name: str
    elem_type: ElementType
    n_plus: str
    n_minus: str
    value: float
    waveform: Optional[SourceWaveform] = None


@dataclass
class Analysis:
    """Desired analysis type from .TRAN / .AC / .PRINT lines."""
    analysis_type: str  # "tran", "ac", "op"
    # Transient params: tstep tstop tstart ...
    tran_params: Optional[Tuple[float, float, Optional[float]]] = None
    # AC params: type (DEC/LIN/OCT), np, fstart, fstop
    ac_params: Optional[Tuple[str, int, float, float]] = None
    print_vars: List[str] = field(default_factory=list)


@dataclass
class Circuit:
    """Complete circuit description parsed from a SPICE netlist."""
    title: str = ""
    elements: List[Element] = field(default_factory=list)
    analyses: List[Analysis] = field(default_factory=list)

    def _all_nodes(self) -> List[str]:
        """Return sorted list of all unique node names (excluding ground)."""
        nodes: set = set()
        for elem in self.elements:
            if elem.n_plus != "0":
                nodes.add(elem.n_plus)
            if elem.n_minus != "0":
                nodes.add(elem.n_minus)
        return sorted(nodes, key=lambda n: (not n.startswith("_"), n))

    def node_map(self) -> Dict[str, int]:
        """Map node names to indices (excluding ground)."""
        return {n: i for i, n in enumerate(self._all_nodes())}

    def num_nodes(self) -> int:
        return len(self._all_nodes())

    def elem_by_type(self, elem_type: ElementType) -> List[Element]:
        return [e for e in self.elements if e.elem_type == elem_type]

    def voltage_sources(self) -> List[Element]:
        return self.elem_by_type(ElementType.VSOURCE)

    def current_sources(self) -> List[Element]:
        return self.elem_by_type(ElementType.ISOURCE)

    def num_voltage_sources(self) -> int:
        return len(self.voltage_sources())

    def num_inductors(self) -> int:
        return len(self.elem_by_type(ElementType.INDUCTOR))

    def state_dim(self) -> int:
        """Total number of MNA variables = node voltages + V-source currents + inductor currents."""
        return self.num_nodes() + self.num_voltage_sources() + self.num_inductors()
