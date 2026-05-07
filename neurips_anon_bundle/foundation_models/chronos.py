"""Chronos foundation model adapter (Amazon).

Tries ChronosPipeline (v1, supports num_samples) first, falls back to
Chronos2Pipeline with predict_quantiles if v1 is unavailable.
"""

import logging
import warnings
import numpy as np
import torch
from foundation_models.base import UnivariateFMBase


# Silence Chronos's long-horizon warning. The installed chronos package
# uses logger.warning() (not warnings.warn), so warnings.filterwarnings
# alone doesn't catch it. Cover both paths.
warnings.filterwarnings("ignore", message=".*recommend keeping prediction length.*")
warnings.filterwarnings("ignore", message=".*quality of longer predictions may degrade.*")


class _ChronosMsgFilter(logging.Filter):
    _NEEDLES = (
        "recommend keeping prediction length",
        "quality of longer predictions may degrade",
    )

    def filter(self, record):
        return not any(n in record.getMessage() for n in self._NEEDLES)


_chronos_filter = _ChronosMsgFilter()
# Attach to root + its handlers + last-resort handler. The chronos logger
# is created as `logging.getLogger(__file__)`, so records propagate up to
# root where our filter catches them on the output handler.
logging.getLogger().addFilter(_chronos_filter)
for _h in logging.getLogger().handlers:
    _h.addFilter(_chronos_filter)
if logging.lastResort is not None:
    logging.lastResort.addFilter(_chronos_filter)


class ChronosModel(UnivariateFMBase):
    def __init__(self, device="cpu", model_id=None):
        self._pipeline = None
        self._version = None

        # Try v1 API first (ChronosPipeline — supports num_samples natively)
        try:
            from chronos import ChronosPipeline
            mid = model_id or "amazon/chronos-t5-small"
            self._pipeline = ChronosPipeline.from_pretrained(
                mid, device_map=device, torch_dtype=torch.bfloat16)
            self._version = "v1"
        except (ImportError, Exception):
            pass

        # Fall back to v2 (Chronos2Pipeline — uses predict_quantiles)
        if self._pipeline is None:
            from chronos import Chronos2Pipeline
            mid = model_id or "amazon/chronos-2"
            try:
                self._pipeline = Chronos2Pipeline.from_pretrained(
                    mid, device_map=device, torch_dtype=torch.bfloat16)
            except TypeError:
                self._pipeline = Chronos2Pipeline.from_pretrained(
                    mid, device_map=device)
            self._version = "v2"

        self._device = device
        print(f"    Chronos loaded ({self._version}): {mid}")

    @property
    def model_name(self):
        return "Chronos"

    @torch.no_grad()
    def _forecast_univariate(self, seq, horizon, n_samples):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*prediction length.*")

            if self._version == "v1":
                # ChronosPipeline handles device internally — pass CPU tensors
                samples = self._pipeline.predict(
                    seq.cpu(), prediction_length=horizon, num_samples=n_samples,
                    limit_prediction_length=False)
                return samples.float()
            else:
                # Chronos2Pipeline — use predict_quantiles with n evenly
                # spaced quantile levels as pseudo-samples
                n_q = min(n_samples, 21)
                q_levels = [round(i / (n_q + 1), 4) for i in range(1, n_q + 1)]
                # predict_quantiles internally uses DataLoader(pin_memory=True)
                # on cuda, which requires CPU tensors. Move to CPU — the pipeline
                # transfers to the model's device internally.
                seq_cpu = seq.cpu()
                ctx = [seq_cpu[i] for i in range(seq_cpu.shape[0])]
                qf = self._pipeline.predict_quantiles(
                    ctx, prediction_length=horizon,
                    quantile_levels=q_levels)
                # In recent chronos-forecasting versions, predict_quantiles
                # returns a tuple (quantiles, mean). Older versions return
                # just the quantiles tensor / list.
                if isinstance(qf, tuple):
                    qf = qf[0]
                # qf shape varies — typically list of (channels, quantiles, horizon)
                # or (batch, channels, quantiles, horizon)
                if isinstance(qf, list):
                    qf = torch.stack(qf) if isinstance(qf[0], torch.Tensor) else torch.tensor(np.array(qf))
                qf = qf.float()
                # Flatten any singleton/channel dim first — some versions
                # return (B, 1, H, n_q).
                if qf.ndim == 4:
                    qf = qf[:, 0, :, :]
                # Ensure layout is (B, n_q, H) so the repeat below expands
                # the quantile axis. Chronos-2 often returns (B, H, n_q).
                if qf.ndim == 3 and qf.shape[-1] != horizon \
                        and qf.shape[-2] == horizon:
                    qf = qf.transpose(-1, -2)
                if qf.shape[1] < n_samples:
                    # Pad by repeating if we have fewer quantiles than requested
                    reps = (n_samples + qf.shape[1] - 1) // qf.shape[1]
                    qf = qf.repeat(1, reps, 1)[:, :n_samples, :]
                return qf[:, :n_samples, :horizon]