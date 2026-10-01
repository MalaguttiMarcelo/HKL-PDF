from __future__ import annotations

from .defaults import PARAM_META

def param_tooltip(name: str) -> str:
    meta = PARAM_META.get((name or "").lower())
    if not meta:
        return f"{name}"
    unit = (meta.get("unit", "") or "").strip()
    desc = (meta.get("desc", "") or "").strip()
    if unit:
        return f"{name} [{unit}]\n{desc}"
    return f"{name}\n{desc}"
