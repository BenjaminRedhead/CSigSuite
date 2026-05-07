import torch
import numpy as np
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from huggingface_hub import hf_hub_download
import functools
torch.load = functools.partial(torch.load, weights_only=False)


# === 1. CHRONOS 2 (Amazon) ===
# Method: Quantile Sampling / Autoregressive Sampling
class ChronosAdapter(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Import the specific pipeline for the new architecture
        try:
            from chronos import Chronos2Pipeline
        except ImportError:
            raise ImportError("Please run: pip install 'chronos-forecasting>=2.0.0'")

        self.device = getattr(args, 'device', 'cuda:0')
        print(f"Loading Chronos-2 from {args.model_path} on {self.device}...")
        # The kwarg has changed across chronos / transformers versions
        # (older = `torch_dtype`, newer wheels reject it). Match the
        # `CopuAdapt/foundation_models/chronos.py` strategy: try with
        # bfloat16 first, fall back to library-default dtype on TypeError.
        try:
            self.pipeline = Chronos2Pipeline.from_pretrained(
                args.model_path,
                device_map=self.device,
                torch_dtype=torch.bfloat16,
            )
        except TypeError:
            self.pipeline = Chronos2Pipeline.from_pretrained(
                args.model_path,
                device_map=self.device,
            )
        self.pred_len = args.pred_len
        self.quantiles = args.quantiles

    def forward(self, x):
        # Input x: [Batch, Lookback, Channels]

        # Chronos 2 Native Multivariate Input:
        # It expects a list of tensors, where each tensor is (Channels, Length).
        # It treats the channels in that single tensor as a "Group" that attends to each other.
        x_cpu = x.cpu()
        context = [ts.permute(1, 0) for ts in x_cpu]

        # Predict
        quantiles_list = self.pipeline.predict_quantiles(
            context,
            prediction_length=self.pred_len,
            quantile_levels=self.quantiles
        )
        quantiles = quantiles_list[0]

        if isinstance(quantiles, torch.Tensor):
            forecast = quantiles.to(self.device)
        # If it returned a List of Tensors, stack them.
        elif isinstance(quantiles, list):
            # Handle nested lists if they occur [[T1], [T2]]
            if len(quantiles) > 0 and isinstance(quantiles[0], list):
                 quantiles = [torch.stack(inner) for inner in quantiles]
            forecast = torch.stack(quantiles, dim=0).to(self.device)
        else:
            raise ValueError(f"Unexpected output type from Chronos: {type(quantiles)}")
        return forecast.permute(0, 3, 2, 1) # [B, S, P, C]

# ==============================================================================
# 2. MOIRAI STANDARD (Salesforce)
# ==============================================================================
class MoiraiAdapter(nn.Module):
    def __init__(self, args):
        super().__init__()
        try:
            from uni2ts.model.moirai import MoiraiModule, MoiraiForecast
        except ImportError:
            raise ImportError("Could not import Moirai. Ensure 'uni2ts' is installed.")
        
        self.device = getattr(args, 'device', 'cuda:0')
        print(f"Loading Moirai Standard from {args.model_path} on {self.device}...")
        module = MoiraiModule.from_pretrained(args.model_path, map_location=self.device)
        self.num_samples = args.n_samples

        # Moirai Small/Base typically use patch_size=32
        # We explicitly set it to prevent 32!=4 errors.
        self.forecast_model = MoiraiForecast(
            module=module,
            prediction_length=args.pred_len,
            context_length=args.seq_len,
            patch_size=64,                 # <--- Increased for 862-channel memory fit
            num_samples=args.n_samples,
            target_dim=args.enc_in,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

        batch_size = getattr(args, 'batch_size', 16)
        self.predictor = self.forecast_model.create_predictor(batch_size=batch_size)
        self.predictor.prediction_net.to(self.device)

    def forward(self, x):
        B, L, C = x.shape
        x_numpy = x.cpu().numpy()
        from gluonts.dataset.common import ListDataset
        
        # Prepare Multivariate Data (Time Major)
        data_list = []
        for i in range(B):
            data_list.append({
                "target": x_numpy[i].T, 
                "start": "2024-01-01 00:00:00"
            })
        
        dataset = ListDataset(data_list, freq="h", one_dim_target=False)
        forecast_it = self.predictor.predict(dataset)
        
        samples_list = [torch.tensor(f.samples, device=self.device) for f in forecast_it]
        batch_samples = torch.stack(samples_list, dim=0)
        
        # Output: [Batch, Samples, PredLen, Channels]
        return batch_samples


# ==============================================================================
# 3. MOIRAI MOE (Salesforce)
# ==============================================================================
class MoiraiMoEAdapter(nn.Module):
    def __init__(self, args):
        super().__init__()
        try:
            from uni2ts.model.moirai_moe import MoiraiMoEModule, MoiraiMoEForecast
        except ImportError:
            raise ImportError("Could not import MoiraiMoE. Ensure 'uni2ts' is installed.")
        
        self.device = getattr(args, 'device', 'cuda:0')
        print(f"Loading Moirai MoE from {args.model_path} on {self.device}...")
        module = MoiraiMoEModule.from_pretrained(args.model_path, map_location=self.device)
        self.num_samples = args.n_samples

        # MoiraiMoE-small only supports patch_size=16 (single pretrained size)
        self.forecast_model = MoiraiMoEForecast(
            module=module,
            prediction_length=args.pred_len,
            context_length=args.seq_len,
            patch_size=16,                 # <--- Only supported size for MoE-small
            num_samples=args.n_samples,
            target_dim=args.enc_in,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        )

        batch_size = getattr(args, 'batch_size', 16)
        self.predictor = self.forecast_model.create_predictor(batch_size=batch_size)
        self.predictor.prediction_net.to(self.device)

    def forward(self, x):
        # Uses exact same logic as Standard Moirai
        B, L, C = x.shape
        x_numpy = x.cpu().numpy()
        from gluonts.dataset.common import ListDataset
        
        data_list = []
        for i in range(B):
            data_list.append({
                "target": x_numpy[i].T, 
                "start": "2024-01-01 00:00:00"
            })
        
        dataset = ListDataset(data_list, freq="h", one_dim_target=False)
        forecast_it = self.predictor.predict(dataset)
        
        samples_list = [torch.tensor(f.samples, device=self.device) for f in forecast_it]
        batch_samples = torch.stack(samples_list, dim=0)
        
        return batch_samples

