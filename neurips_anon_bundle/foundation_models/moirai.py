"""Moirai foundation model adapter (Salesforce). Natively multivariate."""

import time
import numpy as np
import torch
from foundation_models.base import FoundationModel


class MoiraiModel(FoundationModel):
    """Moirai processes all variates jointly (native multivariate).

    Uses the GluonTS predictor interface via uni2ts.
    """

    def __init__(self, device="cpu",
                 model_id="Salesforce/moirai-1.0-R-small",
                 patch_size=64):
        from uni2ts.model.moirai import MoiraiModule, MoiraiForecast
        self._MoiraiForecast = MoiraiForecast
        self._module = MoiraiModule.from_pretrained(
            model_id, map_location=device)
        self._device = device
        self._patch_size = patch_size
        self._model_id = model_id

    @property
    def model_name(self):
        return "Moirai"

    @torch.no_grad()
    def forecast(self, contexts, horizon, n_samples, device, batch_size=16,
                 verbose=True):
        from gluonts.dataset.common import ListDataset

        N, L, D = contexts.shape
        # Ensure device has an index (e.g. "cuda:0" not bare "cuda")
        if device == "cuda":
            device = "cuda:0"
        if "cuda" in device:
            torch.cuda.set_device(device)

        forecast_model = self._MoiraiForecast(
            module=self._module,
            prediction_length=horizon,
            context_length=L,
            patch_size=self._patch_size,
            num_samples=n_samples,
            target_dim=D,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )
        predictor = forecast_model.create_predictor(batch_size=batch_size)
        predictor.prediction_net.to(device)

        data_list = []
        for i in range(N):
            data_list.append({
                "target": contexts[i].T.astype(np.float32),
                "start": "2024-01-01 00:00:00",
            })
        dataset = ListDataset(data_list, freq="h", one_dim_target=False)

        t0 = time.time()
        if verbose:
            print(f"    Moirai: forecasting {N} windows...", flush=True)
        samples_list = []
        for f in predictor.predict(dataset):
            samples_list.append(
                torch.tensor(f.samples, device="cpu"))  # (n_samples, H, D)

        batch_samples = torch.stack(samples_list, dim=0)  # (N, K, H, D)
        # Rearrange to (N, D, K, H)
        out = batch_samples.permute(0, 3, 1, 2)
        print(f"    Done ({time.time() - t0:.1f}s)", flush=True)
        return out
