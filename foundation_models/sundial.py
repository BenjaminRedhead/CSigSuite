"""Sundial foundation model adapter."""

import torch
from transformers import AutoModelForCausalLM
from foundation_models.base import UnivariateFMBase


class SundialModel(UnivariateFMBase):
    def __init__(self, device="cpu", model_id="thuml/sundial-base-128m"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True).to(device).eval()
        self._device = device

    @property
    def model_name(self):
        return "Sundial"

    @torch.no_grad()
    def _forecast_univariate(self, seq, horizon, n_samples):
        return self.model.generate(
            seq, max_new_tokens=horizon, num_samples=n_samples)
