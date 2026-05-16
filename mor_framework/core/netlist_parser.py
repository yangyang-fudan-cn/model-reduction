"""SPICE netlist parser for RC/RCL circuits."""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .circuit import Analysis, Circuit, Element, ElementType, SourceWaveform


class ParseError(Exception):
    """Raised when a netlist line cannot be parsed."""
    pass


def _parse_value(s: str) -> float:
    """Parse a SPICE-format number with optional scale suffix.

    Supports: T, G, MEG, k, m, u, n, p, f (case-insensitive after letter).
    Strips trailing parentheses for pulse parameter values.
    """
    s = s.strip().lower().replace("meg", "x")  # MEG = mega, use placeholder
    # Strip trailing closing paren
    s = s.rstrip(")")
    scale = {
        "t": 1e12, "g": 1e9, "x": 1e6,  # MEG → x
        "k": 1e3,
        "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15,
    }
    match = re.match(r"^([+-]?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*([tgxkmunpf]?)$", s)
    if not match:
        raise ParseError(f"Cannot parse value: {s}")
    num = float(match.group(1))
    suffix = match.group(2)
    return num * scale.get(suffix, 1.0)


def _tokenize(line: str) -> List[str]:
    """Split a SPICE line into tokens, handling inline comments and PULSE(...)."""
    # strip inline comment after first space-separated ';'
    idx = line.find(";")
    if idx > 0:
        line = line[:idx]
    line = line.strip()

    # Merge PULSE(...) into a single token
    line = re.sub(r"PULSE\s*\(([^)]*)\)", lambda m: "PULSE(" + m.group(1).strip().replace(" ", "\x00") + ")", line, flags=re.IGNORECASE)

    parts = []
    for tok in line.split():
        tok = tok.strip()
        if tok and not tok.startswith(";"):
            # Restore spaces inside PULSE
            tok = tok.replace("\x00", " ")
            parts.append(tok)
    return parts


def _is_element(line: str) -> bool:
    """Check if a line is an element definition (starts with a letter but not a dot)."""
    return bool(re.match(r"^[a-zA-Z]", line)) and not line.startswith(".")


def _parse_element(tokens: List[str]) -> Element:
    """Parse a single element line.  Format:
        Rxxx n+ n- value
        Cxxx n+ n- value [IC=...]
        Lxxx n+ n- value [IC=...]
        Vxxx n+ n- [[DC] value] [AC ampl [phase]] [PULSE(...)]
        Ixxx n+ n- [[DC] value] [AC ampl [phase]] [PULSE(...)]
    """
    name = tokens[0].upper()
    elem_type_char = name[0]
    try:
        elem_type = ElementType(elem_type_char)
    except ValueError:
        raise ParseError(f"Unknown element type '{elem_type_char}' in '{name}'")

    n_plus = tokens[1]
    n_minus = tokens[2]
    rest = tokens[3:]
    value = 0.0
    waveform: Optional[SourceWaveform] = None

    if elem_type in (ElementType.RESISTOR, ElementType.CAPACITOR, ElementType.INDUCTOR):
        if not rest:
            raise ParseError(f"{name}: missing value")
        value = _parse_value(rest[0])

    elif elem_type == ElementType.VSOURCE:
        waveform = SourceWaveform()
        i = 0
        while i < len(rest):
            tok = rest[i]
            tok_upper = tok.upper()
            if tok_upper == "DC":
                i += 1
                if i < len(rest):
                    waveform.dc_value = _parse_value(rest[i])
                    i += 1
            elif tok_upper == "AC":
                i += 1
                if i < len(rest) and re.match(r"^[+-]?\d", rest[i]):
                    waveform.ac_amplitude = _parse_value(rest[i])
                    i += 1
                    if i < len(rest) and re.match(r"^[+-]?\d", rest[i]):
                        waveform.ac_phase = _parse_value(rest[i])
                        i += 1
            elif tok_upper.startswith("PULSE"):
                m = re.search(r"PULSE\s*\(\s*([^)]+)\s*\)", tok, re.IGNORECASE)
                if m:
                    params = [_parse_value(p) for p in re.split(r"\s+", m.group(1).strip())]
                    if len(params) >= 7:
                        waveform.pulse_params = tuple(params[:7])
                i += 1
            elif re.match(r"^[+-]?\d", tok):
                waveform.dc_value = _parse_value(tok)
                i += 1
            else:
                i += 1
        value = waveform.dc_value

    elif elem_type == ElementType.ISOURCE:
        waveform = SourceWaveform()
        i = 0
        while i < len(rest):
            tok = rest[i]
            tok_upper = tok.upper()
            if tok_upper == "DC":
                i += 1
                if i < len(rest):
                    waveform.dc_value = _parse_value(rest[i])
                    i += 1
            elif tok_upper == "AC":
                i += 1
                if i < len(rest) and re.match(r"^[+-]?\d", rest[i]):
                    waveform.ac_amplitude = _parse_value(rest[i])
                    i += 1
                    if i < len(rest) and re.match(r"^[+-]?\d", rest[i]):
                        waveform.ac_phase = _parse_value(rest[i])
                        i += 1
            elif tok_upper.startswith("PULSE"):
                m = re.search(r"PULSE\s*\(\s*([^)]+)\s*\)", tok, re.IGNORECASE)
                if m:
                    params = [_parse_value(p) for p in re.split(r"\s+", m.group(1).strip())]
                    if len(params) >= 7:
                        waveform.pulse_params = tuple(params[:7])
                i += 1
            elif re.match(r"^[+-]?\d", tok):
                waveform.dc_value = _parse_value(tok)
                i += 1
            else:
                i += 1
        value = waveform.dc_value

    return Element(name, elem_type, n_plus, n_minus, value, waveform)


def _parse_control_line(tokens: List[str]) -> Optional[Analysis]:
    """Parse .TRAN, .AC, .PRINT, .END lines."""
    if not tokens:
        return None
    cmd = tokens[0].upper()
    if cmd == ".TRAN":
        tstep = _parse_value(tokens[1]) if len(tokens) > 1 else 0.0
        tstop = _parse_value(tokens[2]) if len(tokens) > 2 else 0.0
        tstart = _parse_value(tokens[3]) if len(tokens) > 3 else None
        return Analysis("tran", tran_params=(tstep, tstop, tstart))
    elif cmd == ".AC":
        sweep_type = tokens[1].upper() if len(tokens) > 1 else "DEC"
        np = int(tokens[2]) if len(tokens) > 2 else 10
        fstart = _parse_value(tokens[3]) if len(tokens) > 3 else 1.0
        fstop = _parse_value(tokens[4]) if len(tokens) > 4 else 1e9
        return Analysis("ac", ac_params=(sweep_type, np, fstart, fstop))
    elif cmd == ".PRINT":
        vars_ = [t for t in tokens[1:] if not t.startswith(".")]
        return Analysis("print", print_vars=vars_)
    return None


def parse_netlist(path: str) -> Circuit:
    """Parse a SPICE netlist file and return a Circuit object."""
    with open(path, "r") as f:
        raw_lines = f.readlines()

    # Merge continuation lines (starting with '+')
    merged: List[str] = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("*"):
            continue
        if stripped.startswith("+"):
            if merged:
                merged[-1] += " " + stripped[1:].strip()
        else:
            merged.append(stripped)

    circuit = Circuit()
    analyses_map: dict = {}

    for line in merged:
        if line.upper().startswith(".END"):
            break

        tokens = _tokenize(line)
        if not tokens:
            continue

        first = tokens[0].upper()

        if first == ".TITLE":
            circuit.title = " ".join(tokens[1:])
        elif _is_element(first):
            elem = _parse_element(tokens)
            circuit.elements.append(elem)
        elif first.startswith("."):
            analysis = _parse_control_line(tokens)
            if analysis:
                circuit.analyses.append(analysis)

    _reconcile_analyses(circuit)
    return circuit


def _reconcile_analyses(circuit: Circuit) -> None:
    """Combine .PRINT variables into .TRAN / .AC analyses."""
    print_vars: List[str] = []
    kept: List[Analysis] = []
    for a in circuit.analyses:
        if a.analysis_type == "print":
            print_vars.extend(a.print_vars)
        else:
            kept.append(a)
    if not kept:
        kept.append(Analysis("op"))
    for a in kept:
        a.print_vars = print_vars
    circuit.analyses = kept
