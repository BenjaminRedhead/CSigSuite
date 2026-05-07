"""TimesFM foundation model adapter (Google). Quantile-based → pseudo-probabilistic."""

import time
import numpy as np
import torch
from foundation_models.base import FoundationModel


# 21 evenly spaced quantiles
QUANTILE_LEVELS = [round(0.025 + i * 0.05, 3) for i in range(20)]
QUANTILE_LEVELS = [0.025] + [round(0.05 * i, 2) for i in range(1, 20)] + [0.975]
# Simplified: 0.025, 0.05, 0.10, ..., 0.90, 0.95, 0.975
QUANTILE_LEVELS = [round(q, 3) for q in
                   [0.025] + [i / 20 for i in range(1, 20)] + [0.975]]


class TimesFMModel(FoundationModel):
    """TimesFM: quantile-based forecaster.

    Extracts 21 quantile levels and stacks them as pseudo-samples
    of shape (N, D, 21, H) for the probabilistic experiment.
    """

    def __init__(self, device="cpu",
                 model_id="google/timesfm-2.0-500m-pytorch",
                 context_length=512):
        import timesfm

        self._model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu" if "cuda" in device else "cpu",
                per_core_batch_size=32,
                horizon_len=512,  # max; actual horizon set per forecast
                num_layers=50,
                context_len=context_length,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                huggingface_repo_id=model_id,
            ),
        )
        self._context_length = context_length
        self._device = device

    @property
    def model_name(self):
        return "TimesFM"

    @torch.no_grad()
    def forecast(self, contexts, horizon, n_samples, device, batch_size=32,
                 verbose=True):
        """Flatten (window, variate) pairs into one big TimesFM call.

        TimesFM's internal `per_core_batch_size` handles the actual GPU
        batching, so handing it all N*D series at once is faster than
        looping over variates in Python.
        """
        N, L, D = contexts.shape
        n_q = len(QUANTILE_LEVELS)
        total = N * D

        # Build one big series list: [(n=0,d=0), (n=0,d=1), ..., (n=0,d=D-1),
        # (n=1,d=0), ...]. Row-major flatten matching UnivariateFMBase.
        # contexts: (N, L, D) → (N, D, L) → flatten first two dims.
        flat = np.ascontiguousarray(
            contexts.transpose(0, 2, 1).reshape(total, L)
        ).astype(np.float32)
        series_list = [flat[i] for i in range(total)]
        freqs = [0] * total

        t0 = time.time()
        if verbose:
            print(f"    TimesFM: {total} series (N={N}, D={D})...",
                  flush=True)

        # Try quantile_levels first; fall back to point forecast.
        try:
            _, quantile_forecasts = self._model.forecast(
                series_list, freq=freqs, quantile_levels=QUANTILE_LEVELS)
            # quantile_forecasts: list of (H, n_q) per item
            qf_stack = np.stack(
                [q[:horizon, :] for q in quantile_forecasts],
                axis=0)  # (total, H, n_q)
            out_flat = torch.tensor(
                qf_stack.transpose(0, 2, 1), dtype=torch.float32
            )  # (total, n_q, H)
        except TypeError:
            point_forecasts, _ = self._model.forecast(
                series_list, freq=freqs)
            pf_stack = np.stack(
                [p[:horizon] for p in point_forecasts], axis=0)  # (total, H)
            pf_t = torch.tensor(pf_stack, dtype=torch.float32)  # (total, H)
            # Repeat the point forecast across n_q pseudo-samples
            out_flat = pf_t.unsqueeze(1).expand(-1, n_q, -1).contiguous()

        if verbose:
            print(f"    Done ({time.time() - t0:.1f}s)", flush=True)

        # (total, n_q, H) → (N, D, n_q, H)
        return out_flat.reshape(N, D, n_q, horizon)
