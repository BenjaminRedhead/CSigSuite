"""Foundation model registry.

Usage::

    from foundation_models import load_model

    fm = load_model("chronos", device="cuda")
    samples = fm.forecast(contexts, horizon=96, n_samples=20, device="cuda")
    # samples: (N, D, 20, 96)
"""

from foundation_models.base import FoundationModel

# Lazy imports — each model is only imported when requested, so missing
# packages don't break the whole registry.
_REGISTRY = {
    "sundial":  ("foundation_models.sundial",       "SundialModel"),
    "chronos":  ("foundation_models.chronos",       "ChronosModel"),
    "moirai":   ("foundation_models.moirai",        "MoiraiModel"),
    "timesfm":  ("foundation_models.timesfm_model", "TimesFMModel"),
}

# Which models go in which experiment
PROBABILISTIC_MODELS = ["sundial", "chronos", "moirai", "timesfm"]
POINT_ONLY_MODELS = []
ALL_MODELS = PROBABILISTIC_MODELS + POINT_ONLY_MODELS


def available_models():
    return list(_REGISTRY.keys())


def load_model(name: str, device: str = "cpu", **kwargs) -> FoundationModel:
    """Load a foundation model by name.

    Parameters
    ----------
    name   : one of ``available_models()``.
    device : e.g. "cuda" or "cpu".
    **kwargs : forwarded to the model constructor.

    Returns
    -------
    FoundationModel instance, ready to call ``.forecast()``.
    """
    name = name.lower().replace("-", "").replace("_", "")
    # Normalise common aliases
    aliases = {"timesfm": "timesfm", "times_fm": "timesfm"}
    name = aliases.get(name, name)

    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {available_models()}")

    module_path, class_name = _REGISTRY[name]
    import importlib
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return cls(device=device, **kwargs)
