"""Censoring-quantile ablation: how do CSig(Kernel), CSig(Systemic), and
CSig(Sharpe) converge to plain SigMMD as the censoring quantile is loosened?

Plot layout: one figure, three side-by-side panels (one per censoring
type). Each panel:
  - dashed horizontal line at the model's SigMMD value (constant in q)
  - solid line for that model's CSig(<this type>) at each quantile

X-axis is *inverted* — q=0.95 (strict, only top 5% included) on the left,
q=0.25 (loose, top 75% included) on the right — so the convergence story
reads naturally left-to-right: strict-tail behaviour on the left, full-
distribution behaviour on the right where CSig should approach SigMMD.
"""

import warnings
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import sys
import yaml
import importlib
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import Namespace
from tqdm import tqdm
from collections import defaultdict, OrderedDict

warnings.filterwarnings("ignore", message="Using a non-tuple sequence for multidimensional indexing is deprecated")
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", message=".*weights_only=False.*")

_here = os.path.dirname(os.path.abspath(__file__))
_bundle_root = os.path.abspath(os.path.join(_here, "../.."))
_esk_root = os.path.join(_bundle_root, "ESK")
_sigkernel_root = os.path.join(_bundle_root, "sigkernel")
for _p in [_bundle_root, _esk_root, _sigkernel_root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.metrics.eval_metrics import (
    ESKSigMMD as SigMMD, CensoredESKSigMMD as CensoredSigMMD,
    AugmentPipeline, SigmaEstimator, make_tail_detector,
)
from src.data_provider.data_loader import data_factory

# === GLOBAL SETTINGS ===
ABLATION_DIR = os.environ.get(
    "WRITEABLE_DIR",
    os.path.join(_bundle_root, "results", "ablation_cquants"))
os.makedirs(ABLATION_DIR, exist_ok=True)

# Cap on the number of test-loader batches consumed during inference.
# The ablation only needs enough samples for stable mean estimates of
# SigMMD / CSig at each quantile — the q-curve shape converges quickly.
# Set to None to use the full test set.
MAX_TEST_BATCHES = 100

PAPER_COLORS = ['#76b5c5', '#eab676', '#87ace1', '#b39eb5', '#9fd9b3', '#fabfb7']

# Three censoring detector types we sweep.
#   label   = display label used in result columns / plots
#   method  = make_tail_detector method
#   raw     = whether the detector calibrates / scores on RAW aug paths
#             (Systemic / Sharpe) or on normalised aug paths (Kernel)
#   dataset = which DATASETS key the ablation evaluates this censoring on.
#             Each detector type is paired with the dataset where its
#             semantics are most realistic: kernel/systemic on `cloud`
#             (capacity-constrained system; aggregate-load risks), sharpe
#             on `exchange_rate` (FX returns; risk-adjusted opportunity).
CENSORINGS = [
    ("Kernel",   "kernel",   False, "cloud"),
    ("Systemic", "systemic", True,  "cloud"),
    ("Sharpe",   "sharpe",   True,  "exchange_rate"),
]

# Per-dataset overrides applied on top of the base data_config in
# config_fm.yml. Only the keys present here change between datasets;
# everything else (seq_len, batch_size, etc.) is inherited.
DATASETS = {
    "cloud": {
        "root_path": os.path.join(_bundle_root, "data", "cloud") + os.sep,
        "data_path": "cloud.csv",
        "enc_in":  51,
        "c_out":   51,
        "pred_len": 96,
    },
    "exchange_rate": {
        "root_path": os.path.join(
            _bundle_root, "data", "long_term_forecast", "exchange_rate") + os.sep,
        "data_path": "exchange_rate.csv",
        "enc_in":  8,
        "c_out":   8,
        "pred_len": 96,
    },
}


# === PLOTTING STYLE SETUP (Non-LaTeX) ===
def set_non_latex_style():
    sns.set_context("paper")
    plt.rcParams.update({
        'text.usetex': False,
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'mathtext.fontset': 'stix',
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.titlesize': 12,
        'lines.linewidth': 1.0,
        'grid.linewidth': 0.4,
        'axes.linewidth': 0.8,
    })


class RobustPathReVIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer('mean', None)
        self.register_buffer('stdev', None)

    def fit(self, x):
        self.mean = x.mean(dim=(0, 1), keepdim=True).detach()
        var = x.var(dim=(0, 1), keepdim=True, unbiased=False)
        self.stdev = torch.sqrt(var + self.eps).detach()

    def forward(self, x):
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
            self.stdev = self.stdev.to(x.device)
        return (x - self.mean) / self.stdev


def compute_probabilistic_metrics(preds, target, quantiles=[0.1, 0.5, 0.9]):
    preds, target = preds.cpu(), target.cpu()
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
        loss = torch.max((q - 1) * errors, q * errors)
        losses.append(loss)
    ql = torch.stack(losses).mean().item()
    return es, crps, ql


def compute_variogram_score(preds, target, p=0.5):
    preds, target = preds.cpu(), target.cpu()
    target_diff = torch.abs(target.unsqueeze(-1) - target.unsqueeze(-2)).pow(p)
    preds_diff = torch.abs(preds.unsqueeze(-1) - preds.unsqueeze(-2)).pow(p)
    expected_preds_diff = preds_diff.mean(dim=1)
    vs_matrix = (target_diff - expected_preds_diff).pow(2)
    return vs_matrix.sum(dim=(-1, -2)).mean().item()


def _purge_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


@torch.no_grad()
def _accumulate_sigmmd(processed_batches, sigma, device):
    """SigMMD is independent of the quantile — compute once per model.

    Uses the same memory-safety breakdown as ``benchmark_fm_multi`` and
    ``focus_experiment``: ``no_grad`` context (no save_for_backward)
    + ``max_batch=1`` to bound per-chunk Gram tensor memory.
    """
    m = SigMMD(sigma=sigma, max_batch=1).to(device)
    for X_n, Y_n, _, _ in processed_batches:
        m.update(X_n, Y_n)
    val = m.compute().item()
    del m
    _purge_cuda()
    return val


@torch.no_grad()
def _accumulate_csigmmd(processed_batches, detector, det_input_is_raw,
                       train_aug_norm, sigma, device):
    """CSigMMD with one specific detector. Pivot is the normalised
    train-augmentation; detector masks come from raw or normalised inputs
    depending on ``det_input_is_raw``.
    """
    m = CensoredSigMMD(sigma=sigma, max_batch=1).to(device)
    m.set_pivot(train_aug_norm)
    for X_n, Y_n, X_r, Y_r in processed_batches:
        B_b, S_b = X_n.shape[:2]
        if det_input_is_raw:
            ty = Y_r
            tx = X_r.reshape(-1, *X_r.shape[2:])
        else:
            ty = Y_n
            tx = X_n.reshape(-1, *X_n.shape[2:])
        mask_y = detector.get_mask(ty)
        mask_x = detector.get_mask(tx).reshape(B_b, S_b)
        m.update(X_n, Y_n, mask_x, mask_y)
    val = m.compute().item()
    del m
    _purge_cuda()
    return val


def _prepare_dataset(model, base_data_cfg, ds_overrides, ds_key,
                     model_name, device):
    """Run inference on one dataset and build all calibration state needed
    by the censoring sweeps (revin, aug pipeline, sigma, training-aug
    tensors, processed test-batch cache, SigMMD baseline).

    Returns a dict of GPU-resident state. The model is *not* moved off
    GPU here — the caller manages model lifetime (since the same model
    may be reused across datasets).
    """
    import time
    cfg = {**base_data_cfg, **ds_overrides}
    pred_len = cfg['pred_len']

    print(f"\n>>> [{ds_key}] Loading data...", flush=True)
    dm = data_factory(cfg)
    dm.setup()
    test_loader = dm.test_dataloader()

    print(f"   [{ds_key}] Building calibration tensors...", flush=True)
    train_loader = dm.train_dataloader()
    train_samples = []
    for batch_x, batch_y in train_loader:
        if isinstance(batch_x, (list, tuple)):
            batch_x = batch_x[0]
        train_samples.append(batch_y[:, -pred_len:, :])
        if len(train_samples) * batch_y.shape[0] > 1000:
            break
    train_data = torch.cat(train_samples, dim=0)[:1000].to(device)
    D = train_data.shape[-1]

    path_revin = RobustPathReVIN(D, eps=1e-5).to(device)
    path_revin.fit(train_data)
    train_norm = path_revin(train_data)
    aug_pipe = AugmentPipeline(append_time=True).to(device)
    train_aug_norm = aug_pipe(train_norm)
    train_aug_raw = aug_pipe(train_data)
    sigma = SigmaEstimator.fit(train_norm)
    print(f"   [{ds_key}] D={D}  sigma={sigma:.4f}", flush=True)

    # Determine progress denominator.
    try:
        full_batches = len(test_loader)
    except TypeError:
        full_batches = None
    if MAX_TEST_BATCHES is not None:
        n_batches = (min(full_batches, MAX_TEST_BATCHES)
                     if full_batches is not None else MAX_TEST_BATCHES)
    else:
        n_batches = full_batches
    n_str = str(n_batches) if n_batches is not None else "?"
    if MAX_TEST_BATCHES is not None and full_batches not in (None, n_batches):
        print(f"   [{ds_key}] capping inference at {MAX_TEST_BATCHES} "
              f"of {full_batches} test-loader batches.", flush=True)

    base_metrics = defaultdict(list)
    processed_batches = []
    t_start = time.time()
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_loader):
            if MAX_TEST_BATCHES is not None and i >= MAX_TEST_BATCHES:
                break
            if isinstance(batch_x, (list, tuple)):
                batch_x = batch_x[0]
            batch_x = batch_x.to(device)
            batch_y = batch_y[:, -pred_len:, :].to(device)

            samples = model(batch_x)
            B_curr, S, L, C = samples.shape

            es, crps, ql = compute_probabilistic_metrics(samples, batch_y)
            vs = compute_variogram_score(samples, batch_y)
            base_metrics['ES'].append(es); base_metrics['CRPS'].append(crps)
            base_metrics['QL'].append(ql); base_metrics['VS'].append(vs)

            X_flat = torch.nan_to_num(samples.reshape(-1, L, C), nan=0.0)
            X_aug_norm = aug_pipe(path_revin(X_flat)).view(B_curr, S, -1, D + 1)
            Y_aug_norm = aug_pipe(path_revin(batch_y))
            X_aug_raw = aug_pipe(X_flat).view(B_curr, S, -1, D + 1)
            Y_aug_raw = aug_pipe(batch_y)
            processed_batches.append(
                (X_aug_norm, Y_aug_norm, X_aug_raw, Y_aug_raw))

            elapsed = time.time() - t_start
            print(f"    [{ds_key}/{model_name}] batch {i + 1}/{n_str}  "
                  f"elapsed {elapsed:5.1f}s", flush=True)

    sigmmd_value = _accumulate_sigmmd(processed_batches, sigma, device)
    print(f"   [{ds_key}] SigMMD baseline = {sigmmd_value:.6f}", flush=True)

    return {
        "ds_key":         ds_key,
        "D":              D,
        "sigma":          sigma,
        "train_aug_norm": train_aug_norm,
        "train_aug_raw":  train_aug_raw,
        "batches":        processed_batches,
        "sigmmd":         sigmmd_value,
        "base_metrics":   base_metrics,
    }


def run_benchmark(config, experiments,
                  cquants=(0.01, 0.1, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95)):
    set_non_latex_style()
    cquants = list(cquants)

    # `load_probabilistic_models` guarantees a single-element list.
    device = torch.device(config['data_config']['device'])
    base_data_cfg = config['data_config']

    # The set of datasets we need is the union of CENSORINGS' dataset keys.
    needed_ds = list(dict.fromkeys(c[3] for c in CENSORINGS))
    for k in needed_ds:
        if k not in DATASETS:
            raise ValueError(f"CENSORINGS references unknown dataset key "
                             f"`{k}` — add it to the DATASETS dict.")

    exp = experiments[0]
    name = exp['name']
    model = exp['model'].to(device)
    model.eval()
    print(f"\n>>> Evaluating {name} on datasets: {needed_ds}")

    # Run inference once per unique dataset using the same model.
    dataset_state = {}
    for ds_key in needed_ds:
        dataset_state[ds_key] = _prepare_dataset(
            model, base_data_cfg, DATASETS[ds_key], ds_key, name, device)
        _purge_cuda()

    # Free the model — only PDE-solver work remains.
    try:
        model.to('cpu')
    except Exception:
        pass
    del model
    exp['model'] = None
    _purge_cuda()

    # ── Combined output naming reflects multi-dataset evaluation ───
    out_tag = "-".join(needed_ds)        # e.g. "cloud-exchange_rate"

    # ════════════════════════════════════════════════════════════════════
    # Score extraction — one call per detector on its own training paths.
    # The MAD-β block below consumes these scores.
    # ════════════════════════════════════════════════════════════════════
    from src.metrics.eval_metrics import (
        SystemicLoadDetector, SharpeMVDetector,
    )

    all_scores = OrderedDict()
    with torch.no_grad():
        for cens_label, method, det_raw, ds_key in CENSORINGS:
            st = dataset_state[ds_key]
            if method == "kernel":
                probe = make_tail_detector(
                    method="kernel", sigma=st['sigma'], quantile=0.5)
                probe.fit(st['train_aug_norm'])
                scores = probe._embedding_dist(st['train_aug_norm'])
            elif method == "systemic":
                probe = SystemicLoadDetector(quantile=0.5)
                probe.fit(st['train_aug_raw'])
                scores = probe._peak_aggregate_load(st['train_aug_raw'])
            elif method == "sharpe":
                probe = SharpeMVDetector(quantile=0.5)
                probe.fit(st['train_aug_raw'])
                scores = probe._mv_utility(st['train_aug_raw'])
            else:
                raise ValueError(f"Unknown method: {method}")
            all_scores[cens_label] = scores.flatten().float().cpu()
            del probe
            _purge_cuda()

    # ════════════════════════════════════════════════════════════════════
    # ADAPTIVE β* via MAD — chosen as the project default after the
    # earlier MCD / MAD / local-var convergence comparison; MAD gave
    # the cleanest CSig → SigMMD curve as q is relaxed.
    #
    #   β* = 1 / (κ · MAD(scores)),  κ = 1.4826 ≈ 1/Φ⁻¹(0.75)
    #
    # κ is the consistency constant that makes MAD a robust σ estimator
    # under normality (so κ·MAD ≈ σ̂). β* therefore has units of
    # 1/[score]. Fixed per detector — does not depend on q.
    # ════════════════════════════════════════════════════════════════════
    MAD_CONST = 1.4826
    print(f"\n  [Adaptive-β pass: per-detector β* = 1 / ({MAD_CONST} · "
          f"MAD(scores))]", flush=True)

    def _mad(x):
        med = x.median()
        return float((x - med).abs().median().item())

    mad_steepnesses = OrderedDict()       # {cens_label: β}
    mad_sigma_hats = OrderedDict()        # {cens_label: σ̂_MAD}

    for cens_label, _, _, ds_key in CENSORINGS:
        scores_flat = all_scores[cens_label]
        mad = _mad(scores_flat)
        sigma_hat = MAD_CONST * mad
        steep = 1.0 / max(sigma_hat, 1e-12)
        mad_steepnesses[cens_label] = steep
        mad_sigma_hats[cens_label] = sigma_hat
        print(f"      {cens_label:>9s}[{ds_key:<14s}]: "
              f"MAD={mad:.4e}  σ̂={sigma_hat:.4e}  →  β* = {steep:.4e}",
              flush=True)

    mad_table = []
    for q in cquants:
        row = {"Model": name, "CQuant": q}
        for cens_label, mtd, det_raw, ds_key in CENSORINGS:
            s = mad_steepnesses[cens_label]
            st = dataset_state[ds_key]
            if mtd == "kernel":
                det = make_tail_detector(
                    method="kernel", sigma=st['sigma'],
                    quantile=q, steepness=s)
                det.fit(st['train_aug_norm'])
            else:
                det = make_tail_detector(
                    method=mtd, quantile=q, steepness=s)
                det.fit(st['train_aug_raw'])
            csig_val = _accumulate_csigmmd(
                st['batches'], det, det_raw,
                st['train_aug_norm'], st['sigma'], device)
            row[f"CSig({cens_label})"]   = csig_val
            row[f"SigMMD({cens_label})"] = st['sigmmd']
            row[f"Dataset({cens_label})"] = ds_key
            print(f"    [MAD-β]  q={q:.2f}  "
                  f"CSig({cens_label})[{ds_key}] = {csig_val:.6f}  "
                  f"β={s:.3e}", flush=True)
            del det
            _purge_cuda()
        mad_table.append(row)

    df_mad = pd.DataFrame(mad_table).sort_values('CQuant')

    # Per-detector palette + neutral SigMMD reference colour.
    panel_colors = {
        "Kernel":   '#1f77b4',   # blue
        "Systemic": '#d62728',   # red
        "Sharpe":   '#2ca02c',   # green
    }
    sig_color = '#444444'

    # Free cached batches before plotting (plotting is CPU-only).
    for st in dataset_state.values():
        del st['batches']
    _purge_cuda()

    # ════════════════════════════════════════════════════════════════════
    # PLOT — 1×3 grid (one panel per detector), MAD-derived β*.
    # ════════════════════════════════════════════════════════════════════
    fig_c, axes_c = plt.subplots(
        1, 3, figsize=(7.5, 2.8),
        sharey=False, constrained_layout=True)
    for c_idx, (cens_label, _, _, ds_key) in enumerate(CENSORINGS):
        ax = axes_c[c_idx]
        ds_display = "exchange" if ds_key == "exchange_rate" else ds_key
        xs = df_mad['CQuant']
        ys = df_mad[f"CSig({cens_label})"]
        sig_val = float(df_mad[f"SigMMD({cens_label})"].iloc[0])
        beta_str = f"$\\beta^* = {mad_steepnesses[cens_label]:.2e}$"
        ax.plot(xs, ys, color=panel_colors[cens_label],
                marker='o', markersize=4, linewidth=1.8,
                label=f"CSig({cens_label})")
        ax.axhline(sig_val, color=sig_color, linestyle='--',
                   linewidth=1.2, alpha=0.85,
                   label=f"SigMMD ({ds_display})")
        ax.set_title(
            f"CSig({cens_label}) — {ds_display}\n{beta_str}",
            fontsize=8)
        ax.invert_xaxis()
        ax.set_xlabel('Censoring Quantile  ($q$)')
        if c_idx == 0:
            ax.set_ylabel('MMD Distance')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.legend(loc='best', fontsize=7, frameon=True, framealpha=0.9)
    fig_c.suptitle(
        "Convergence of Metrics via Censoring Quantile with Adaptive β (MAD)",
        fontsize=10, y=1.02)
    out_path_c = os.path.join(
        ABLATION_DIR, f"{out_tag}_mad_beta_plot.pdf")
    fig_c.savefig(out_path_c, dpi=300, bbox_inches='tight')
    plt.close(fig_c)
    print(f"\nPlot   : {out_path_c}")


def load_probabilistic_models(config_path):
    """Load *only the first active model* from the config.

    The ablation studies score behaviour as a function of q on a single
    forecaster, so any further active entries are skipped here — at
    config-parse time, before any model is instantiated. Constructing
    extra adapters has real cost (download + GPU placement) and a
    failure in any of them (e.g. wrong device, missing file) would
    abort the run before the chosen model gets to do anything.
    """
    with open(config_path, 'r') as f:
        cfg_yaml = yaml.safe_load(f)
    active = [m for m in cfg_yaml['models'] if m.get('active', True)]
    if not active:
        raise ValueError(
            f"No active model in {config_path} — set `active: true` on "
            f"exactly one entry under `models`.")
    if len(active) > 1:
        chosen = active[0]['name']
        skipped = [m['name'] for m in active[1:]]
        print(f"  [Note] {len(active)} active models in config; "
              f"loading only `{chosen}`. Skipping: {skipped}")
    model_cfg = active[0]
    mod = importlib.import_module(f"src.models.{model_cfg['module']}")
    model_class = getattr(mod, model_cfg['class'])
    merged = {**cfg_yaml['data_config'], **model_cfg.get('params', {})}
    experiments_list = [{
        'name':  model_cfg['name'],
        'model': model_class(Namespace(**merged)),
    }]
    return cfg_yaml, experiments_list


if __name__ == "__main__":
    conf, exps = load_probabilistic_models(
        os.path.join(os.path.dirname(__file__), "config_fm.yml"))
    run_benchmark(conf, exps,
                  cquants=[0.01, 0.1, 0.25, 0.5, 0.75, 0.85, 0.9, 0.95])
