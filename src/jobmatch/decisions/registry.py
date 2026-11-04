"""Decision backend registry with LAZY imports (heavy deps such as torch load only on demand).

model = get_backend("qwen-logits", permute=True)
available_backends()  -> {"qwen-logits": True, "laya": False, ...}
"""

from __future__ import annotations

import importlib
from typing import Any

from jobmatch.decisions.base import DecisionModel

# canonical name -> (module, class, default kwargs)
_BACKENDS: dict[str, tuple[str, str, dict[str, Any]]] = {
    "qwen-logits": ("jobmatch.decisions.qwen_logits", "QwenLogitsDecisionModel", {}),
    "qwen-logits-perm": (
        "jobmatch.decisions.qwen_logits",
        "QwenLogitsDecisionModel",
        {"permute": True},
    ),
    "laya": ("jobmatch.decisions.laya_backend", "LayaDecisionModel", {}),
    "jevos": ("jobmatch.decisions.jevos_backend", "JevosDecisionModel", {}),
}

_ALIASES: dict[str, str] = {
    "qwen": "qwen-logits",
    "qwen_logits": "qwen-logits",
    "logits": "qwen-logits",
    "qwen-perm": "qwen-logits-perm",
    "qwen_logits_perm": "qwen-logits-perm",
    "jev": "jevos",
    "jev-os": "jevos",
}


def canonical_name(name: str) -> str:
    key = name.strip().lower()
    key = _ALIASES.get(key, key)
    if key not in _BACKENDS:
        raise KeyError(f"unknown decision backend {name!r}; known: {sorted(_BACKENDS)}")
    return key


def get_backend(name: str, **kw: Any) -> DecisionModel:
    """Instantiate a backend by name (or alias). Raises ImportError if its deps are missing."""
    module, cls_name, defaults = _BACKENDS[canonical_name(name)]
    cls = getattr(importlib.import_module(module), cls_name)
    return cls(**{**defaults, **kw})


def available_backends() -> dict[str, bool]:
    """Canonical backend name -> whether its module imports (does not load any model)."""
    out = {}
    for name, (module, _cls, _kw) in _BACKENDS.items():
        try:
            importlib.import_module(module)
            out[name] = True
        except Exception:  # noqa: BLE001 - any import failure means "not available"
            out[name] = False
    return out
