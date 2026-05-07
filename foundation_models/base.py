"""Abstract base class for time-series foundation models."""

from abc import ABC, abstractmethod
import time

import numpy as np
import torch


class FoundationModel(ABC):
    """Unified interface for time-series foundation models.

    Every adapter must implement ``forecast`` which returns per-variate
    samples in the shape ``(N, D, K, H)`` — matching the existing
    ``run_sundial`` output so that all downstream code (Ch.Indep, ICA,
    evaluation) works unchanged.
    """

    @abstractmethod
    def forecast(self, contexts: np.ndarray, horizon: int,
                 n_samples: int, device: str, batch_size: int = 16,
                 verbose: bool = True) -> torch.Tensor:
        """Generate probabilistic forecasts.

        Parameters
        ----------
        contexts : (N, L, D) numpy array — lookback windows.
        horizon  : int — prediction length.
        n_samples: int — number of samples per variate (1 for point).
        device   : str — e.g. "cuda" or "cpu".
        batch_size : int — micro-batch size for GPU inference.
        verbose  : bool — print progress (False during CoRA training).

        Returns
        -------
        torch.Tensor of shape (N, D, n_samples, H).
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Short display name for tables/logs."""

    @property
    def supports_probabilistic(self) -> bool:
        """Override to False for point-only models (Timer, TTM)."""
        return True

    @property
    def supports_variable_context(self) -> bool:
        """Override to False for fixed-context models (TTM).
        AdaPTS strips prediction_horizon from the context by design,
        so models that require an exact context length are incompatible."""
        return True

    # ── Convenience helpers ──────────────────────────────────────────

    def forecast_point(self, contexts: np.ndarray, horizon: int,
                       device: str, batch_size: int = 16) -> torch.Tensor:
        """Mean point forecast: (N, H, D)."""
        samples = self.forecast(contexts, horizon, n_samples=20,
                                device=device, batch_size=batch_size)
        return samples.mean(dim=2).permute(0, 2, 1)  # (N, D, K, H) → (N, H, D)


class UnivariateFMBase(FoundationModel):
    """Mixin for models that process each variate independently.

    Subclasses implement ``_forecast_univariate`` which takes a batch of
    univariate contexts and returns samples.  The outer loop over variates
    is handled here.
    """

    @abstractmethod
    def _forecast_univariate(self, seq: torch.Tensor, horizon: int,
                             n_samples: int) -> torch.Tensor:
        """
        seq:     (batch, L) float32 tensor on the correct device.
        Returns: (batch, n_samples, H) tensor.
        """

    def forecast(self, contexts, horizon, n_samples, device, batch_size=64,
                 verbose=True):
        """Batch variates along with windows.

        Each (window, variate) pair is an independent univariate forecast,
        so we flatten (N, L, D) → (N*D, L) and let the GPU process large
        batches. This avoids the 862-step Python loop that dominated
        wall-time on high-dim datasets (traffic, electricity).
        """
        N, L, D = contexts.shape
        total = N * D

        # (N, L, D) → (N, D, L) → (N*D, L), row-major: item (n*D + d)
        flat = np.ascontiguousarray(
            contexts.transpose(0, 2, 1).reshape(total, L))

        out_flat = torch.zeros(total, n_samples, horizon)
        t0 = time.time()
        # Report ~5 progress lines across the whole flattened work.
        report_every = max(1, total // 5)
        last_reported = 0

        for s in range(0, total, batch_size):
            e = min(s + batch_size, total)
            seq = torch.tensor(flat[s:e], dtype=torch.float32).to(device)
            out_flat[s:e] = self._forecast_univariate(
                seq, horizon, n_samples).cpu()
            if verbose and (e - last_reported >= report_every or e == total):
                print(f"    {e}/{total} items ({time.time() - t0:.0f}s)",
                      flush=True)
                last_reported = e

        if verbose:
            print(f"    Done ({time.time() - t0:.1f}s)", flush=True)
        # (N*D, K, H) → (N, D, K, H)
        return out_flat.reshape(N, D, n_samples, horizon)
