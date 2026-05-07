"""
Multi-dataset, multi-horizon foundation-model benchmark (single seed 2026).

Six foundation models × twelve datasets × four prediction horizons.
For every dataset except illness: horizons (32, 96, 192, 336).
For illness: horizons (24, 32, 64, 96).

The ETT-small family is treated as four separate datasets (ETTh1, ETTh2,
ETTm1, ETTm2). Censoring quantile defaults to 0.95 across all cells.
Use `update_fm_results.py` to rerun ETT under additional quantiles
without touching the main result file.

Per-(dataset, horizon) metric tables + a final Wins/Losses/Draws summary
for SigMMD, CSig(Kernel), CSig(Systemic) (on ECL/ETT*/Traffic/Cloud),
and CSig(Sharpe) (on Exchange only). W/L/D denominators are counted
over *all applicable (dataset, horizon) combinations*.

Usage:
    python -u src/experiments/benchmark_fm_multi.py
    python -u src/experiments/benchmark_fm_multi.py --only_dataset illness
"""

import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*weights_only=False.*")
warnings.filterwarnings("ignore", message=".*non-tuple sequence.*")
warnings.filterwarnings("ignore", message=".*prediction length.*")

import os
import sys
import gc
import json
import time
import argparse
from argparse import Namespace
from collections import OrderedDict

import torch
import numpy as np
import pandas as pd

# ── Path setup ─────────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
_bundle_root = os.path.abspath(os.path.join(_here, "../.."))
_sigrisk_root = _bundle_root        # bundle root doubles as the data root
_esk_root = os.path.join(_bundle_root, "ESK")
_sigkernel_root = os.path.join(_bundle_root, "sigkernel")
for p in [_bundle_root, _esk_root, _sigkernel_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.metrics.eval_metrics import (
    ESKSigMMD,
    CensoredESKSigMMD,
    make_tail_detector,
    AugmentPipeline,
    PathReVIN,
    SigmaEstimator,
)
from src.models.adapters import MoiraiMoEAdapter
from foundation_models import load_model
from foundation_models.chronos import ChronosModel


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

SEED = 2026
MODELS = ["Chronos", "Chronos-2", "Moirai", "MoiraiMoE", "Sundial", "TimesFM"]
N_SAMPLES = 100
MAX_TEST_WINDOWS = 100         # uniform cap across all datasets
CALIB_RATIO = 0.2
CENSORING_QUANTILE = 0.95
BATCH_SIZE = 8
REL_TOL = 1e-6                 # draw tolerance for W/L/D

# Metrics included in the final W/L/D summary (all "lower is better")
WLD_METRICS = ["SigMMD", "CSig(Kernel)", "CSig(Systemic)", "CSig(Sharpe)"]

# Horizon sweeps — illness gets shorter lengths because its series is only 967 rows.
HORIZONS_DEFAULT = [32, 96, 192, 336]
HORIZONS_ILLNESS = [24, 32, 64, 96]

# Per-dataset applicability / registry
DATASETS = OrderedDict([
    ("ERA5", {
        "path": os.path.join(_sigrisk_root, "data/ERA5/era5_land_merged.csv"),
        "max_variates": 0, "horizons": HORIZONS_DEFAULT, "lookback": 512,
        "cens": ["kernel"],
    }),
    ("EWELD", {
        "path": os.path.join(_sigrisk_root,
                             "data/EWELD/Weather Data/EWELD_Merged.csv"),
        "max_variates": 0, "horizons": HORIZONS_DEFAULT, "lookback": 512,
        "cens": ["kernel"],
    }),
    ("electricity", {
        "path": os.path.join(_sigrisk_root,
            "data/long_term_forecast/electricity/electricity.csv"),
        "max_variates": 50, "horizons": HORIZONS_DEFAULT, "lookback": 512,
        "cens": ["kernel", "systemic"],
        "skip_models": ["MoiraiMoE"],   # OOM at 50 channels × 512 lookback
    }),
    ("ETTh1", {
        "path": os.path.join(_sigrisk_root,
            "data/long_term_forecast/ETT-small/ETTh1.csv"),
        "max_variates": 0, "horizons": HORIZONS_DEFAULT, "lookback": 512,
        "cens": ["kernel", "systemic"],
    }),
    ("ETTh2", {
        "path": os.path.join(_sigrisk_root,
            "data/long_term_forecast/ETT-small/ETTh2.csv"),
        "max_variates": 0, "horizons": HORIZONS_DEFAULT, "lookback": 512,
        "cens": ["kernel", "systemic"],
    }),
    ("ETTm1", {
        "path": os.path.join(_sigrisk_root,
            "data/long_term_forecast/ETT-small/ETTm1.csv"),
        "max_variates": 0, "horizons": HORIZONS_DEFAULT, "lookback": 512,
        "cens": ["kernel", "systemic"],
    }),
    ("ETTm2", {
        "path": os.path.join(_sigrisk_root,
            "data/long_term_forecast/ETT-small/ETTm2.csv"),
        "max_variates": 0, "horizons": HORIZONS_DEFAULT, "lookback": 512,
        "cens": ["kernel", "systemic"],
    }),
    ("exchange_rate", {
        "path": os.path.join(_sigrisk_root,
            "data/long_term_forecast/exchange_rate/exchange_rate.csv"),
        "max_variates": 0, "horizons": HORIZONS_DEFAULT, "lookback": 512,
        "cens": ["kernel", "sharpe"],
    }),
    ("illness", {
        "path": os.path.join(_sigrisk_root,
            "data/long_term_forecast/illness/national_illness.csv"),
        "max_variates": 0, "horizons": HORIZONS_ILLNESS, "lookback": 96,
        "cens": ["kernel"],
    }),
    ("traffic", {
        "path": os.path.join(_sigrisk_root,
            "data/long_term_forecast/traffic/traffic.csv"),
        "max_variates": 50, "horizons": HORIZONS_DEFAULT, "lookback": 512,
        "cens": ["kernel", "systemic"],
        "skip_models": ["MoiraiMoE"],   # OOM
    }),
    ("weather", {
        "path": os.path.join(_sigrisk_root,
            "data/long_term_forecast/weather/weather.csv"),
        "max_variates": 0, "horizons": HORIZONS_DEFAULT, "lookback": 512,
        "cens": ["kernel"],
    }),
    ("cloud", {
        "path": os.path.join(_sigrisk_root, "data/cloud/cloud.csv"),
        "max_variates": 50, "horizons": HORIZONS_DEFAULT, "lookback": 512,
        "cens": ["kernel", "systemic"],
    }),
])

# Denominator for each W/L/D metric — count of applicable (dataset × horizon) pairs.
WLD_TOTALS = {m: 0 for m in WLD_METRICS}
for _name, _cfg in DATASETS.items():
    n_h = len(_cfg["horizons"])
    WLD_TOTALS["SigMMD"] += n_h
    if "kernel" in _cfg["cens"]:
        WLD_TOTALS["CSig(Kernel)"] += n_h
    if "systemic" in _cfg["cens"]:
        WLD_TOTALS["CSig(Systemic)"] += n_h
    if "sharpe" in _cfg["cens"]:
        WLD_TOTALS["CSig(Sharpe)"] += n_h

# Display column keys for the per-dataset tables.
PER_DATASET_COLS = ["CRPS", "ES", "QL", "VS",
                    "SigMMD", "CSig(Kernel)", "CSig(Systemic)", "CSig(Sharpe)"]


# ═══════════════════════════════════════════════════════════════════════════
# Data / window helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_dataset(path, max_v=0):
    df = pd.read_csv(path)
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    elif "Time" in df.columns:
        df = df.drop(columns=["Time"])
    # Drop non-numeric / all-NaN columns, forward/back-fill sparse NaNs.
    df = df.select_dtypes(include=[np.number])
    df = df.dropna(axis=1, how="all")
    df = df.ffill().bfill()
    if max_v > 0 and df.shape[1] > max_v:
        df = df.iloc[:, :max_v]
    return df.values.astype(np.float32)


def create_nonoverlapping_windows(data, lookback, horizon, start_idx):
    contexts, targets = [], []
    idx = start_idx
    while idx + horizon <= len(data):
        if idx - lookback < 0:
            break
        contexts.append(data[idx - lookback: idx])
        targets.append(data[idx: idx + horizon])
        idx += horizon
    return np.stack(contexts), np.stack(targets)


def create_overlapping_windows(data, lookback, horizon, step, limit=2000):
    targets = []
    for i in range(lookback, len(data) - horizon, step):
        targets.append(data[i: i + horizon])
        if len(targets) >= limit:
            break
    return np.stack(targets)


# ═══════════════════════════════════════════════════════════════════════════
# Standard metrics
# ═══════════════════════════════════════════════════════════════════════════

def compute_probabilistic_metrics(preds, target, quantiles=(0.1, 0.5, 0.9)):
    preds, target = preds.cpu().float(), target.cpu().float()
    target_exp = target.unsqueeze(1)
    diff_target = torch.norm(preds - target_exp, dim=-1).mean(dim=-1).mean(dim=1)
    preds_roll = torch.roll(preds, 1, dims=1)
    diff_intra = torch.norm(preds - preds_roll, dim=-1).mean(dim=-1).mean(dim=1)
    es = (diff_target - 0.5 * diff_intra).mean().item()
    diff = torch.abs(preds - target_exp).mean(dim=1)
    intra = torch.abs(preds - preds_roll).mean(dim=1)
    crps = (diff - 0.5 * intra).mean().item()
    losses = []
    for q in quantiles:
        pred_q = torch.quantile(preds, q, dim=1)
        errors = target - pred_q
        losses.append(torch.max((q - 1) * errors, q * errors))
    ql = torch.stack(losses).mean().item()
    return es, crps, ql


def compute_variogram_score(preds, target, p=0.5):
    preds, target = preds.cpu().float(), target.cpu().float()
    target_diff = torch.abs(target.unsqueeze(-1) - target.unsqueeze(-2)).pow(p)
    preds_diff = torch.abs(preds.unsqueeze(-1) - preds.unsqueeze(-2)).pow(p)
    expected_preds_diff = preds_diff.mean(dim=1)
    vs_matrix = (target_diff - expected_preds_diff).pow(2)
    return vs_matrix.sum(dim=(-1, -2)).mean().item()


# ═══════════════════════════════════════════════════════════════════════════
# Model loaders
# ═══════════════════════════════════════════════════════════════════════════

class MoiraiMoEWrapper:
    """Adapts SigRisk's MoiraiMoEAdapter to the CopuAdapt foundation-model API."""

    def __init__(self, device, seq_len, pred_len, n_samples, n_variates,
                 batch_size=1,
                 model_path="Salesforce/moirai-moe-1.0-R-small"):
        args = Namespace(
            model_path=model_path, device=device,
            pred_len=pred_len, seq_len=seq_len,
            n_samples=n_samples, enc_in=n_variates, batch_size=batch_size,
        )
        self.adapter = MoiraiMoEAdapter(args)
        self.adapter.eval()
        self._device = device

    @property
    def model_name(self):
        return "MoiraiMoE"

    @torch.no_grad()
    def forecast(self, contexts, horizon, n_samples, device, batch_size=4,
                 verbose=True):
        N, L, D = contexts.shape
        out_list = []
        t0 = time.time()
        outer_bs = max(1, batch_size)
        for s in range(0, N, outer_bs):
            e = min(s + outer_bs, N)
            x = torch.tensor(contexts[s:e], dtype=torch.float32).to(self._device)
            samples = self.adapter(x)                          # (B, S, P, C)
            out_list.append(samples.permute(0, 3, 1, 2).cpu()) # → (B, D, K, H)
            if verbose and (e % max(1, N // 4) == 0 or e == N):
                print(f"      Batch {e}/{N} ({time.time() - t0:.0f}s)",
                      flush=True)
        return torch.cat(out_list, dim=0)


def get_model(name, device, seq_len, pred_len, n_samples, n_variates):
    if name == "Chronos":
        return load_model("chronos", device=device)
    if name == "Chronos-2":
        return ChronosModel(device=device, model_id="amazon/chronos-2")
    if name == "Moirai":
        return load_model("moirai", device=device)
    if name == "MoiraiMoE":
        return MoiraiMoEWrapper(
            device=device, seq_len=seq_len, pred_len=pred_len,
            n_samples=n_samples, n_variates=n_variates, batch_size=1)
    if name == "Sundial":
        return load_model("sundial", device=device)
    if name == "TimesFM":
        return load_model("timesfm", device=device)
    raise ValueError(f"Unknown model: {name}")


def _drop_model_attrs(model):
    """In-place tear-down of a model's heavy GPU attributes.

    `del model` inside a function only clears the local binding — the caller
    still holds its reference. So we null-out internal state in-place before
    the caller sets its own reference to None.
    """
    if model is None:
        return
    for attr in ("adapter", "_pipeline", "_module", "_model", "model"):
        if hasattr(model, attr):
            try:
                setattr(model, attr, None)
            except Exception:
                pass


def purge_cuda():
    """Aggressively reclaim GPU memory.

    Runs gc.collect a few times (to break reference cycles that keep
    tensors alive), then empty_cache + ipc_collect + synchronize.
    """
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        torch.cuda.synchronize()


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════════
# Per-dataset pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_one_dataset(name, cfg, horizon, device, seed, log_fn):
    """Execute the benchmark on a single (dataset, horizon) combination.

    Returns:
        results : dict {model_name: {metric: value}} (None if model skipped/failed)
    """
    log_fn(f"\n{'#' * 110}")
    log_fn(f"# DATASET: {name}  |  HORIZON: {horizon}")
    log_fn(f"{'#' * 110}")
    log_fn(f"  Path: {cfg['path']}")
    if cfg.get("static_kernel", "laplacian") != "laplacian":
        log_fn(f"  Static kernel: {cfg['static_kernel']}  "
               f"(per-dataset override of the project Laplacian default)")

    skip_models = set(cfg.get("skip_models", []))
    if skip_models:
        log_fn(f"  Skip-list (OOM-known): {sorted(skip_models)}")

    # ── Load & window ───────────────────────────────────────────────
    if not os.path.isfile(cfg["path"]):
        log_fn(f"  MISSING FILE — skipping {name}.")
        return {m: None for m in MODELS}

    data = load_dataset(cfg["path"], max_v=cfg["max_variates"])
    T, D = data.shape
    log_fn(f"  Data shape: ({T}, {D})")

    lookback = cfg["lookback"]
    if T < lookback + 2 * horizon + 10:
        log_fn(f"  Not enough rows ({T}) for lookback={lookback} + horizon={horizon}.")
        return {m: None for m in MODELS}

    # Leave enough room for ≥5 calibration windows even on tiny datasets.
    calib_end = max(int(T * CALIB_RATIO), lookback + 5 * horizon)
    # Make sure at least one test window still fits after the calibration split.
    calib_end = min(calib_end, T - lookback - horizon)
    test_start = max(calib_end, lookback)
    contexts, targets = create_nonoverlapping_windows(
        data, lookback, horizon, test_start)
    N_total = len(contexts)
    if N_total == 0:
        log_fn(f"  Zero test windows — skipping {name}.")
        return {m: None for m in MODELS}

    if N_total > MAX_TEST_WINDOWS:
        keep = np.linspace(0, N_total - 1, MAX_TEST_WINDOWS, dtype=int)
        contexts = contexts[keep]
        targets = targets[keep]
        log_fn(f"  Test windows: {MAX_TEST_WINDOWS} "
               f"(sub-sampled from {N_total})")
    else:
        log_fn(f"  Test windows: {N_total}")
    N_windows = len(contexts)

    # ── Calibration for metrics ─────────────────────────────────────
    # Finer step gives more windows on short calibration splits.
    calib_step = max(1, horizon // 4)
    calib_targets = create_overlapping_windows(
        data[:calib_end], lookback, horizon, step=calib_step)
    if len(calib_targets) < 3:
        log_fn(f"  Too few calibration windows ({len(calib_targets)}).")
        return {m: None for m in MODELS}
    log_fn(f"  Calibration windows: {len(calib_targets)} (step={calib_step})")

    calib_targets_t = torch.tensor(calib_targets, dtype=torch.float32).to(device)
    revin = PathReVIN(D).to(device)
    revin.fit(calib_targets_t)
    calib_norm = revin(calib_targets_t)
    aug = AugmentPipeline(append_time=True).to(device)
    calib_aug = aug(calib_norm)
    calib_aug_raw = aug(calib_targets_t)
    sigma = SigmaEstimator.fit(calib_norm)
    log_fn(f"  Sigma: {sigma:.4f}")

    # ── Fit only the detectors this dataset needs ──────────────────
    cens_list = cfg["cens"]
    detectors = OrderedDict()       # display_name → detector
    det_input_is_raw = {}           # display_name → bool (raw or norm aug?)
    if "kernel" in cens_list:
        d = make_tail_detector(
            "kernel", sigma=sigma, quantile=CENSORING_QUANTILE)
        d.fit(calib_aug)
        detectors["Kernel"] = d
        det_input_is_raw["Kernel"] = False
    if "systemic" in cens_list:
        d = make_tail_detector("systemic", quantile=CENSORING_QUANTILE)
        d.fit(calib_aug_raw)
        detectors["Systemic"] = d
        det_input_is_raw["Systemic"] = True
    if "sharpe" in cens_list:
        d = make_tail_detector("sharpe", quantile=CENSORING_QUANTILE)
        d.fit(calib_aug_raw)
        detectors["Sharpe"] = d
        det_input_is_raw["Sharpe"] = True
    log_fn(f"  Censored-metric detectors: {list(detectors.keys())}")

    calib_aug_cpu = calib_aug.cpu()
    del calib_targets_t, calib_norm, calib_aug, calib_aug_raw
    purge_cuda()

    targets_t = torch.tensor(targets, dtype=torch.float32)  # CPU

    # ── Loop over models ────────────────────────────────────────────
    results = {}
    for model_name in MODELS:
        if model_name in skip_models:
            log_fn(f"\n  [{model_name}] SKIPPED on {name} (known-OOM).")
            results[model_name] = None
            continue

        log_fn(f"\n  [{model_name}] Loading model...")
        t0 = time.time()
        try:
            row = _run_one_model(
                model_name, device, seed, lookback, horizon,
                contexts, targets_t, N_windows, D, sigma,
                revin, aug, detectors, det_input_is_raw,
                calib_aug_cpu, log_fn,
                static_kernel=cfg.get("static_kernel", "laplacian"),
            )
            results[model_name] = row
            log_fn(f"    OK ({time.time() - t0:.1f}s)  "
                   f"CRPS={row['CRPS']:.4f}  SigMMD={row['SigMMD']:.4f}  "
                   + "  ".join(
                       f"CSig({d})={row[f'CSig({d})']:.4f}"
                       for d in detectors))
        except torch.cuda.OutOfMemoryError as exc:
            log_fn(f"    OOM: {str(exc)[:200]}")
            results[model_name] = None
        except Exception as exc:
            log_fn(f"    FAILED: {type(exc).__name__}: {str(exc)[:200]}")
            results[model_name] = None
        finally:
            # Drop the just-caught exception's traceback aggressively — it
            # pins frame locals (including GPU tensors) in _run_one_model
            # until cleared. sys.exc_info is already cleared once we exit
            # the except block, but _purge_cuda runs multiple gc passes.
            purge_cuda()

    # Free calibration-side state before returning so the next horizon
    # starts clean even if this one didn't complete normally.
    del calib_aug_cpu, targets_t, revin, aug, detectors, det_input_is_raw
    purge_cuda()
    return results


def _run_one_model(model_name, device, seed, lookback, horizon,
                   contexts, targets_t, N_windows, D, sigma,
                   revin, aug, detectors, det_input_is_raw,
                   calib_aug_cpu, log_fn, static_kernel="laplacian"):
    """Run one foundation model on the pre-computed windows.

    Returns the per-metric row dict. Raises on failure. All GPU-resident
    locals (model, forecasts, per-batch tensors, MMD accumulators) are
    bound inside this function's frame, so on `return` or `raise` they
    are released by reference-counting before the caller's `purge_cuda`.
    """
    model = None
    try:
        model = get_model(model_name, device, lookback, horizon,
                          N_SAMPLES, D)

        set_seed(seed)
        forecasts = model.forecast(
            contexts, horizon, N_SAMPLES,
            device=device, batch_size=BATCH_SIZE)
        fcst_np = forecasts.numpy()                    # (N, D, K, H) on CPU
        del forecasts

        # NaN scrub — last-context persistence fallback.
        n_nan = int(np.isnan(fcst_np).sum())
        if n_nan > 0:
            last_ctx = contexts[:, -1, :]
            fallback = last_ctx[:, :, None, None]
            fcst_np = np.where(
                np.isnan(fcst_np),
                np.broadcast_to(fallback, fcst_np.shape),
                fcst_np,
            ).astype(np.float32)
            log_fn(f"    NaN samples scrubbed: {n_nan}")

        preds_bslc = torch.from_numpy(
            np.ascontiguousarray(fcst_np.transpose(0, 2, 3, 1))
        )  # (N, K, H, D) on CPU
        del fcst_np

        # ── Standard metrics (CPU-batched) ─────────────────────────
        es_vals, crps_vals, ql_vals, vs_vals = [], [], [], []
        for i in range(0, N_windows, BATCH_SIZE):
            j = min(i + BATCH_SIZE, N_windows)
            p_batch = preds_bslc[i:j]
            t_batch = targets_t[i:j]
            es, crps, ql = compute_probabilistic_metrics(p_batch, t_batch)
            vs = compute_variogram_score(p_batch, t_batch)
            es_vals.append(es); crps_vals.append(crps)
            ql_vals.append(ql); vs_vals.append(vs)
            del p_batch, t_batch
        # We no longer need the model itself — free it before the
        # GPU-heavy MMD work to keep the headroom high.
        _drop_model_attrs(model)
        model = None
        purge_cuda()

        # ── SigMMD + applicable censored variants ──────────────────
        # Peak chunk memory in SigKernelDiagFn.forward is roughly
        # 3 × mb × S² × L² × 8 bytes (three fp64 tensors G / M / K live
        # simultaneously).  Scale mb down with horizon to stay under the
        # 80 GB A100 budget; the no_grad wrap below further suppresses
        # ctx.save_for_backward so M/K are freed at forward exit.
        if horizon <= 96:
            mb_sig, mb_cen = 4, 2
        elif horizon <= 192:
            mb_sig, mb_cen = 2, 1
        else:                          # horizon >= 336
            mb_sig, mb_cen = 1, 1
        log_fn(f"    mb_sig={mb_sig}  mb_cen={mb_cen}  (horizon={horizon})")

        with torch.no_grad():
            calib_aug_device = calib_aug_cpu.to(device)
            sig_mmd = ESKSigMMD(sigma=sigma, dyadic_order=1,
                                max_batch=mb_sig,
                                static_kernel=static_kernel).to(device)
            cens_mmd = OrderedDict()
            for d_name in detectors:
                c = CensoredESKSigMMD(sigma=sigma, dyadic_order=1,
                                      max_batch=mb_cen,
                                      static_kernel=static_kernel).to(device)
                c.set_pivot(calib_aug_device)
                cens_mmd[d_name] = c

            try:
                for i in range(0, N_windows, BATCH_SIZE):
                    j = min(i + BATCH_SIZE, N_windows)
                    p_batch = preds_bslc[i:j].to(device, non_blocking=True)
                    t_batch = targets_t[i:j].to(device, non_blocking=True)
                    B, S, L, C = p_batch.shape

                    t_norm = revin(t_batch)
                    t_aug = aug(t_norm)
                    p_flat = p_batch.reshape(B * S, L, C)
                    p_norm = revin(p_flat)
                    p_aug_flat = aug(p_norm)
                    L_aug, D_aug = p_aug_flat.shape[1], p_aug_flat.shape[2]
                    p_aug = p_aug_flat.view(B, S, L_aug, D_aug)
                    t_aug_raw = aug(t_batch)
                    p_aug_raw_flat = aug(p_flat)

                    sig_mmd.update(p_aug, t_aug)
                    for d_name, det in detectors.items():
                        if det_input_is_raw[d_name]:
                            ty, px = t_aug_raw, p_aug_raw_flat
                        else:
                            ty, px = t_aug, p_aug_flat
                        mask_y = det.get_mask(ty)
                        mask_x = det.get_mask(px).view(B, S)
                        cens_mmd[d_name].update(p_aug, t_aug, mask_x, mask_y)

                    # Explicit drop — loop-variable bindings otherwise retain
                    # the last iteration's GPU tensors across the loop boundary.
                    del (p_batch, t_batch, t_norm, t_aug, p_flat, p_norm,
                         p_aug_flat, p_aug, t_aug_raw, p_aug_raw_flat)

                sig_mmd_val = sig_mmd.compute().item()
                cen_vals = {d: c.compute().item() for d, c in cens_mmd.items()}
            finally:
                # Always release the MMD accumulators + pivot even if update/
                # compute raised mid-loop.
                sig_mmd = None
                for k in list(cens_mmd.keys()):
                    cens_mmd[k] = None
                cens_mmd = None
                calib_aug_device = None
                preds_bslc = None

        row = {
            "CRPS": float(np.mean(crps_vals)),
            "ES":   float(np.mean(es_vals)),
            "QL":   float(np.mean(ql_vals)),
            "VS":   float(np.mean(vs_vals)),
            "SigMMD": sig_mmd_val,
        }
        for d_name, v in cen_vals.items():
            row[f"CSig({d_name})"] = v
        return row
    finally:
        _drop_model_attrs(model)
        model = None


# ═══════════════════════════════════════════════════════════════════════════
# Per-dataset printing + W/L/D
# ═══════════════════════════════════════════════════════════════════════════

def print_dataset_table(name, horizon, results, log_fn):
    """Print per-model metric table for one (dataset, horizon), '—' for N/A."""
    log_fn(f"\n  Per-model metrics on {name} @ H={horizon}:")
    hdr = f"  {'Model':<12}"
    for m in PER_DATASET_COLS:
        hdr += f" | {m:>13}"
    log_fn(hdr)
    log_fn("  " + "-" * (len(hdr) - 2))

    # Best-value per metric (for "*" marker).
    best_per_metric = {}
    for m in PER_DATASET_COLS:
        vals = []
        for model in MODELS:
            r = results.get(model)
            if r is not None and m in r:
                vals.append(r[m])
        if vals:
            best_per_metric[m] = min(vals)

    for model in MODELS:
        r = results.get(model)
        row = f"  {model:<12}"
        for m in PER_DATASET_COLS:
            if r is None or m not in r:
                row += f" | {'—':>13}"
            else:
                v = r[m]
                best = best_per_metric.get(m)
                marker = "*" if best is not None and v == best else " "
                row += f" |{marker}{v:>12.4f}"
        log_fn(row)


def applicable_metrics_for_dataset(cfg):
    """Which W/L/D metrics apply to this dataset config."""
    applicable = {"SigMMD"}
    if "kernel" in cfg["cens"]:
        applicable.add("CSig(Kernel)")
    if "systemic" in cfg["cens"]:
        applicable.add("CSig(Systemic)")
    if "sharpe" in cfg["cens"]:
        applicable.add("CSig(Sharpe)")
    return applicable


def compute_wld_for_dataset(name, cfg, results):
    """Determine each (model, metric) W/L/D outcome on one dataset.

    - None           : metric not applicable (N/A for this dataset)
    - "W" | "L" | "D": model's outcome on that metric vs the cohort
    Models that didn't produce a value (skip/fail) are credited with "L"
    on every applicable metric (keeps denominators symmetric across models).
    """
    applicable = applicable_metrics_for_dataset(cfg)
    out = {}
    for metric in WLD_METRICS:
        if metric not in applicable:
            for model in MODELS:
                out[(model, metric)] = None
            continue

        vals = {m: results[m][metric]
                for m in MODELS
                if results.get(m) is not None and metric in results[m]}
        if not vals:
            for model in MODELS:
                out[(model, metric)] = None
            continue

        min_v = min(vals.values())
        tol = REL_TOL * max(abs(min_v), 1.0)
        best_set = {m for m, v in vals.items() if abs(v - min_v) <= tol}

        for model in MODELS:
            if model in vals:
                if model in best_set and len(best_set) == 1:
                    out[(model, metric)] = "W"
                elif model in best_set:
                    out[(model, metric)] = "D"
                else:
                    out[(model, metric)] = "L"
            else:
                # Skipped or failed — counts as loss.
                out[(model, metric)] = "L"
    return out


def print_final_wld_summary(all_wld, keys_run, log_fn):
    """keys_run: list of (ds_name, horizon) tuples."""
    log_fn(f"\n\n{'#' * 110}")
    log_fn(f"# FINAL WINS / LOSSES / DRAWS — across {len(keys_run)} (dataset, horizon) combination(s)")
    log_fn(f"{'#' * 110}")
    log_fn(f"  (W — L — D per applicable-combination count; lower metric = win)")
    log_fn(f"  Max-applicable denominators: "
           + ", ".join(f"{m}={WLD_TOTALS[m]}" for m in WLD_METRICS))

    hdr = f"\n  {'Model':<12}"
    for m in WLD_METRICS:
        hdr += f" | {m + f'  (of {WLD_TOTALS[m]})':>26}"
    log_fn(hdr)
    log_fn("  " + "-" * (len(hdr) - 2))

    for model in MODELS:
        row = f"  {model:<12}"
        for metric in WLD_METRICS:
            w = l = d = 0
            for key in keys_run:
                outcome = all_wld.get(key, {}).get((model, metric))
                if outcome == "W":
                    w += 1
                elif outcome == "L":
                    l += 1
                elif outcome == "D":
                    d += 1
            row += f" | {f'{w}-{l}-{d}':>26}"
        log_fn(row)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--only_dataset", type=str, default=None,
                   help="Run on a single named dataset instead of all nine.")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--static_kernel", type=str, default=None,
                   choices=["rbf", "matern", "laplacian"],
                   help="Override the per-dataset static-kernel default for "
                        "every dataset in this run. When omitted, the "
                        "project-wide default (Laplacian) is used. Use this "
                        "to A/B against matern or rbf: "
                        "`--static_kernel rbf --only_dataset cloud`.")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device

    writeable = os.environ.get("WRITEABLE_DIR",
                               os.path.join(_sigrisk_root, "results"))
    out_dir = args.output_dir or os.path.join(writeable, "benchmark_fm_multi")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "benchmark.log")
    json_path = os.path.join(out_dir, "raw_results.json")
    if os.path.exists(log_path):
        os.remove(log_path)

    def log_fn(*s):
        msg = " ".join(str(x) for x in s)
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log_fn(f"Multi-dataset FM benchmark")
    log_fn(f"  Seed:              {args.seed}")
    log_fn(f"  Device:            {device}")
    log_fn(f"  Models:            {MODELS}")
    log_fn(f"  N samples / window:{N_SAMPLES}")
    log_fn(f"  Max test windows:  {MAX_TEST_WINDOWS}")
    log_fn(f"  Output dir:        {out_dir}")

    # Apply CLI static-kernel override if provided. This patches the
    # DATASETS registry in-place so every dataset in this run uses the
    # given kernel; when omitted, every dataset falls back to the
    # project-wide default (Laplacian).
    if args.static_kernel is not None:
        log_fn(f"  Static-kernel override: {args.static_kernel} "
               f"(applies to all datasets in this run)")
        for _ds_cfg in DATASETS.values():
            _ds_cfg["static_kernel"] = args.static_kernel

    if args.only_dataset:
        if args.only_dataset not in DATASETS:
            log_fn(f"Unknown dataset: {args.only_dataset}. "
                   f"Available: {list(DATASETS.keys())}")
            sys.exit(2)
        ds_list = [args.only_dataset]
    else:
        ds_list = list(DATASETS.keys())
    log_fn(f"  Datasets to run:   {ds_list}")
    log_fn(f"  Horizons per ds:   " +
           ", ".join(f"{d}={DATASETS[d]['horizons']}" for d in ds_list))

    all_results = OrderedDict()       # (ds_name, horizon) → {model → metrics}
    all_wld = OrderedDict()           # (ds_name, horizon) → {(model, metric) → outcome}
    keys_run = []                     # list of (ds_name, horizon) tuples

    for ds_name in ds_list:
        cfg = DATASETS[ds_name]
        for horizon in cfg["horizons"]:
            key = (ds_name, horizon)
            t_ds = time.time()
            # Clean slate before each (dataset, horizon) — reclaim anything
            # still reserved by PyTorch from the previous combo.
            purge_cuda()
            if torch.cuda.is_available():
                free_gb = (torch.cuda.mem_get_info()[0] / (1024 ** 3)
                           if hasattr(torch.cuda, "mem_get_info") else -1)
                log_fn(f"  [pre-run GPU free: {free_gb:.1f} GiB]")
            try:
                results = run_one_dataset(
                    ds_name, cfg, horizon, device, args.seed, log_fn)
            except Exception as exc:
                log_fn(f"  DATASET-LEVEL FAILURE on {ds_name}@H={horizon}: "
                       f"{type(exc).__name__}: {str(exc)[:200]}")
                results = {m: None for m in MODELS}

            all_results[key] = results
            print_dataset_table(ds_name, horizon, results, log_fn)
            wld = compute_wld_for_dataset(ds_name, cfg, results)
            all_wld[key] = wld
            keys_run.append(key)
            log_fn(f"\n  {ds_name}@H={horizon} done in {time.time() - t_ds:.1f}s")

            # Incremental JSON dump after every (dataset, horizon) so a
            # long-running sweep always leaves partial results on disk.
            _dump_results_json(json_path, args.seed, keys_run,
                               all_results, all_wld)
            purge_cuda()

    # ── Final W/L/D summary ──────────────────────────────────────────
    print_final_wld_summary(all_wld, keys_run, log_fn)

    # ── Final JSON dump ──────────────────────────────────────────────
    _dump_results_json(json_path, args.seed, keys_run, all_results, all_wld)
    log_fn(f"\n  Raw results JSON: {json_path}")
    log_fn(f"  Benchmark log:    {log_path}")
    log_fn("\nDone.")


def _dump_results_json(json_path, seed, keys_run, all_results, all_wld):
    """Serialise (dataset, horizon)-keyed dicts using '<ds>@h<H>' string keys."""
    def kstr(key):
        ds, h = key
        return f"{ds}@h{h}"

    per_combo = OrderedDict()
    for key in keys_run:
        per_combo[kstr(key)] = all_results.get(key)

    wld_out = OrderedDict()
    for key in keys_run:
        wld = all_wld.get(key, {})
        wld_out[kstr(key)] = {
            f"{model}|{metric}": outcome
            for (model, metric), outcome in wld.items()
        }

    json_data = {
        "seed": seed,
        "models": MODELS,
        "combinations": [kstr(k) for k in keys_run],
        "horizons_per_dataset": {
            d: DATASETS[d]["horizons"] for d in DATASETS
        },
        "wld_metrics": WLD_METRICS,
        "wld_totals": WLD_TOTALS,
        "per_combo_results": per_combo,
        "wld_outcomes": wld_out,
    }
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2, default=str)


if __name__ == "__main__":
    main()
