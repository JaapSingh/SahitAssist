from __future__ import annotations
import csv
from typing import Dict, Iterable

def load_anmol_lipi_mapping(path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row["anmol_lipi"].strip()
            dst = row["unicode"].strip()
            if src and dst:
                mapping[src] = dst
    return mapping

def normalize_line(line: str, mapping: Dict[str, str]) -> str:
    i = 0
    out = []
    while i < len(line):
        matched = False
        for w in range(4, 0, -1):
            seg = line[i:i+w]
            if seg in mapping:
                out.append(mapping[seg])
                i += w
                matched = True
                break
        if not matched:
            out.append(line[i])
            i += 1
    return "".join(out)

def normalize_text(lines: Iterable[str], mapping: Dict[str, str]) -> Iterable[str]:
    for ln in lines:
        yield normalize_line(ln.rstrip("\n"), mapping)