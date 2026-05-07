"""
agent_decision_experiment.py
============================

Two scenarios — exchange (Markowitz mean-variance) and cloud (newsvendor
capacity provisioning). Per scenario, each foundation model produces a
forecast; an agent computes the optimal action under that forecast and
applies it to the ground-truth path to obtain a realised reward. The
realised reward defines the *oracle ranking* of forecasters.

For each scenario × sweep value (γ for exchange, α for cloud) we report
how well each metric's ranking matches the oracle: a top-weighted
Kendall τ (scipy.stats.weightedtau, hyperbolic weigher) so disagreements
at the top of the ranking dominate, plus the top-1 selection (does the
metric pick the same forecaster the agent would have wanted?). Repeated
across seeds [2021..2025] and aggregated to mean ± std (τ) and total hit
rate (top-1).

Sweep:
  Exchange (Markowitz MV):  γ ∈ {0.5, 1.0, 2.0, 5.0}
  Cloud    (Newsvendor)  :  α ∈ {0.7, 0.8, 0.9, 0.95}
                            (α = c_under / (c_under + c_over))

Forecasts are run once per (seed, scenario, model); only the utility
evaluation iterates over the sweep, so per-seed compute matches a
single-horizon benchmark and the multi-seed run is ~|SEEDS|× that.

Usage:
    python -u src/experiments/agent_decision_experiment.py
    python -u src/experiments/agent_decision_experiment.py --seeds 2021,2022
    python -u src/experiments/agent_decision_experiment.py --only_scenario exchange
"""

import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*weights_only=False.*")
warnings.filterwarnings("ignore", message=".*non-tuple sequence.*")
warnings.filterwarnings("ignore", message=".*prediction length.*")

import os
import sys
import json
import time
import argparse
from collections import OrderedDict

import torch
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_bundle_root = os.path.abspath(os.path.join(_here, "../.."))
_sigrisk_root = _bundle_root        # bundle root doubles as the data root
_esk_root = os.path.join(_bundle_root, "ESK")
_sigkernel_root = os.path.join(_bundle_root, "sigkernel")
for _p in [_bundle_root, _esk_root, _sigkernel_root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.metrics.eval_metrics import (
    ESKSigMMD,
    CensoredESKSigMMD,
    make_tail_detector,
    AugmentPipeline,
    PathReVIN,
    SigmaEstimator,
)
from src.experiments.benchmark_fm_multi import (
    load_dataset,
    create_nonoverlapping_windows,
    create_overlapping_windows,
    compute_probabilistic_metrics,
    compute_variogram_score,
    get_model,
    _drop_model_attrs,
    purge_cuda,
    set_seed,
)


# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

SEEDS = [2021, 2022, 2023, 2024, 2025]
MODELS = ["Chronos", "Chronos-2", "Moirai", "MoiraiMoE", "Sundial", "TimesFM"]
N_SAMPLES = 100
MAX_TEST_WINDOWS = 100
CALIB_RATIO = 0.2
CENSORING_QUANTILE = 0.95
BATCH_SIZE = 8

GAMMAS = [0.5, 1.0, 2.0, 5.0]            # Markowitz risk aversion
ALPHAS = [0.7, 0.8, 0.9, 0.95]           # Newsvendor under-cost share

SCENARIOS = OrderedDict([
    ("exchange", {
        "path": os.path.join(
            _sigrisk_root, "data/long_term_forecast/exchange_rate/exchange_rate.csv"),
        "lookback": 512, "horizon": 32, "max_variates": 0,
        "censoring": ["sharpe", "kernel"],
        "agent": "markowitz_mv",
        "skip_models": [],
    }),
    ("cloud", {
        "path": os.path.join(_sigrisk_root, "data/cloud/cloud.csv"),
        "lookback": 512, "horizon": 96, "max_variates": 50,
        "censoring": ["systemic", "kernel"],
        "agent": "newsvendor",
        "skip_models": [],
    }),
])


# ═══════════════════════════════════════════════════════════════════════════
# Agent utility under truth (oracle reward)
# ═══════════════════════════════════════════════════════════════════════════

def realised_mv_utility(forecasts, targets, last_ctx, gamma, ridge=1e-4):
    """Markowitz mean-variance realised utility per window.

    forecasts : (N, D, K, H) numpy  — forecast price-level paths
    targets   : (N, H, D)    numpy  — realised price-level paths
    last_ctx  : (N, D)       numpy  — last context price (t = -1)
    gamma     : float                — risk aversion

    Returns ndarray (N,) of realised MV utility for the agent's optimal
    weights w* derived from each window's forecast distribution and
    applied to the realised return path.

    w*(forecast) = (1/(2γ)) Σ̂_f⁻¹ μ̂_f
    U_real(w*)  = w*ᵀ μ_r − (γ/2) w*ᵀ Σ_r w*
    """
    N, D, K, H = forecasts.shape
    eps = 1e-8

    # Forecast → per-step simple returns. Previous price for t=0 is the
    # last observed context price.
    last_ctx_b = np.broadcast_to(
        last_ctx[:, :, None, None], (N, D, K, 1))
    fcst_prev = np.concatenate(
        [last_ctx_b, forecasts[:, :, :, :-1]], axis=-1)
    fcst_returns = (forecasts - fcst_prev) / (np.abs(fcst_prev) + eps)
    # → flatten K and H into one sample axis; D stays as the variate axis.
    fr = fcst_returns.transpose(0, 2, 3, 1).reshape(N, K * H, D)

    mu_f = fr.mean(axis=1)                                     # (N, D)
    cf = fr - mu_f[:, None, :]
    Sigma_f = np.einsum('nij,nik->njk', cf, cf) / max(K * H - 1, 1)
    Sigma_f_reg = Sigma_f + ridge * np.eye(D)[None, :, :]
    w_star = (1.0 / (2.0 * gamma)) * np.linalg.solve(
        Sigma_f_reg, mu_f[..., None])[..., 0]                  # (N, D)

    # Realised returns from the target path.
    tgt_prev = np.concatenate(
        [last_ctx[:, None, :], targets[:, :-1, :]], axis=1)
    real_returns = (targets - tgt_prev) / (np.abs(tgt_prev) + eps)
    mu_r = real_returns.mean(axis=1)                           # (N, D)
    cr = real_returns - mu_r[:, None, :]
    Sigma_r = np.einsum('nij,nik->njk', cr, cr) / max(H - 1, 1)

    term_lin = np.einsum('ni,ni->n', w_star, mu_r)
    term_quad = np.einsum('ni,nij,nj->n', w_star, Sigma_r, w_star)
    return term_lin - 0.5 * gamma * term_quad                  # (N,)


def realised_newsvendor_utility(forecasts, targets, alpha):
    """Newsvendor realised reward (negative mean cost) per window.

    forecasts : (N, D, K, H) numpy
    targets   : (N, H, D)    numpy
    alpha     : float in (0, 1) — under-cost share, c_under / (c_over + c_under)

    Aggregate-load formulation: per time-step, the agent provisions
    capacity c*(t) = quantile_α over k of Σ_d forecasts[n, d, k, t].
    The realised cost penalises both directions:
        cost(t) = (1 − α) · max(c* − L_real, 0) + α · max(L_real − c*, 0)
    where L_real(t) = Σ_d targets[n, t, d]. Reward = −mean_t cost(t).
    """
    fcst_agg = forecasts.sum(axis=1)                           # (N, K, H)
    real_agg = targets.sum(axis=-1)                            # (N, H)
    c_star = np.quantile(fcst_agg, alpha, axis=1)              # (N, H)
    overage = np.maximum(c_star - real_agg, 0.0)
    underage = np.maximum(real_agg - c_star, 0.0)
    cost_t = (1.0 - alpha) * overage + alpha * underage        # (N, H)
    return -cost_t.mean(axis=1)                                # (N,)


# ═══════════════════════════════════════════════════════════════════════════
# Ranking helpers
# ═══════════════════════════════════════════════════════════════════════════

from scipy.stats import weightedtau


def weighted_tau_lower_better(metric_vals, utility_vals):
    """Top-weighted Kendall τ between (metric, lower=better) and
    (utility, higher=better).

    Uses ``scipy.stats.weightedtau`` with its default hyperbolic weigher
    ``1/(rank+1)`` (Vigna 2015), so disagreements at the top of the
    ranking cost much more than disagreements among the worst
    forecasters — exactly what we want when judging "does the metric
    pick the best forecaster".

    Sign-aligned so τ = +1 means the lowest-metric forecaster is also
    the highest-utility one. Returns 0.0 if τ is undefined (e.g. all
    values tied).
    """
    tau, _ = weightedtau(-np.asarray(metric_vals, dtype=float),
                         np.asarray(utility_vals, dtype=float))
    return 0.0 if np.isnan(tau) else float(tau)


def top1_agreement(metric_vals, utility_vals):
    """1 if argmin(metric) == argmax(utility), else 0."""
    return int(int(np.argmin(metric_vals)) == int(np.argmax(utility_vals)))


# ═══════════════════════════════════════════════════════════════════════════
# Per-scenario forecast + metric pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_scenario(name, cfg, device, seed, log_fn):
    """Execute the forecast / metric pipeline once per model.

    Returns:
        forecasts : {model_name: ndarray (N, D, K, H)} — None on failure
        metrics   : {model_name: {metric: float}}      — None on failure
        targets   : ndarray (N, H, D)
        last_ctx  : ndarray (N, D)
        n_windows : int
    """
    log_fn(f"\n{'#' * 95}")
    log_fn(f"# SCENARIO: {name}  (agent: {cfg['agent']})")
    log_fn(f"{'#' * 95}")
    log_fn(f"  Path: {cfg['path']}")

    if not os.path.isfile(cfg["path"]):
        log_fn(f"  MISSING FILE — skipping {name}.")
        return None

    data = load_dataset(cfg["path"], max_v=cfg["max_variates"])
    T, D = data.shape
    lookback, horizon = cfg["lookback"], cfg["horizon"]
    log_fn(f"  Data shape: ({T}, {D})  lookback={lookback}  horizon={horizon}")

    if T < lookback + 2 * horizon + 10:
        log_fn(f"  Not enough rows for split — abort.")
        return None

    calib_end = max(int(T * CALIB_RATIO), lookback + 5 * horizon)
    calib_end = min(calib_end, T - lookback - horizon)
    test_start = max(calib_end, lookback)

    contexts, targets = create_nonoverlapping_windows(
        data, lookback, horizon, test_start)
    if len(contexts) == 0:
        log_fn(f"  Zero test windows — abort.")
        return None

    if len(contexts) > MAX_TEST_WINDOWS:
        keep = np.linspace(0, len(contexts) - 1, MAX_TEST_WINDOWS, dtype=int)
        contexts = contexts[keep]
        targets = targets[keep]
        log_fn(f"  Test windows: {MAX_TEST_WINDOWS} (sub-sampled from {len(contexts)})")
    else:
        log_fn(f"  Test windows: {len(contexts)}")
    N_windows = len(contexts)
    last_ctx = contexts[:, -1, :].astype(np.float32)            # (N, D)

    # ── Calibration ─────────────────────────────────────────────────
    calib_step = max(1, horizon // 4)
    calib_targets = create_overlapping_windows(
        data[:calib_end], lookback, horizon, step=calib_step)
    if len(calib_targets) < 3:
        log_fn(f"  Too few calibration windows ({len(calib_targets)}).")
        return None
    log_fn(f"  Calibration windows: {len(calib_targets)} (step={calib_step})")

    calib_t = torch.tensor(calib_targets, dtype=torch.float32).to(device)
    revin = PathReVIN(D).to(device)
    revin.fit(calib_t)
    calib_norm = revin(calib_t)
    aug = AugmentPipeline(append_time=True).to(device)
    calib_aug = aug(calib_norm)
    calib_aug_raw = aug(calib_t)
    sigma = SigmaEstimator.fit(calib_norm)
    log_fn(f"  Sigma: {sigma:.4f}")

    # `censoring` may be a single method or a list of methods. We always
    # build a dict { display_label: (detector, det_input_is_raw) } so the
    # downstream metric pass can iterate over multiple CSig variants in
    # one shot (e.g. both CSig(Sharpe) and CSig(Kernel) on exchange).
    cens_methods = cfg["censoring"]
    if isinstance(cens_methods, str):
        cens_methods = [cens_methods]

    detectors = OrderedDict()           # {label: (detector, raw_aug_input)}
    for m in cens_methods:
        if m == "kernel":
            d = make_tail_detector(
                "kernel", sigma=sigma, quantile=CENSORING_QUANTILE)
            d.fit(calib_aug)            # fits on the *normalised* aug paths
            detectors["Kernel"] = (d, False)
        elif m == "sharpe":
            d = make_tail_detector("sharpe", quantile=CENSORING_QUANTILE)
            d.fit(calib_aug_raw)
            detectors["Sharpe"] = (d, True)
        elif m == "systemic":
            d = make_tail_detector("systemic", quantile=CENSORING_QUANTILE)
            d.fit(calib_aug_raw)
            detectors["Systemic"] = (d, True)
        else:
            raise ValueError(f"Unknown censoring method: {m}")
    cens_labels = list(detectors.keys())
    log_fn(f"  Censoring: " + ", ".join(f"CSig({l})" for l in cens_labels)
           + f"  (q={CENSORING_QUANTILE})")

    calib_aug_cpu = calib_aug.cpu()
    del calib_t, calib_norm, calib_aug, calib_aug_raw
    purge_cuda()

    targets_t = torch.tensor(targets, dtype=torch.float32)      # CPU

    # ── Loop over models ────────────────────────────────────────────
    forecasts_all = {}
    metrics_all = {}
    skip_models = set(cfg.get("skip_models", []))
    for model_name in MODELS:
        if model_name in skip_models:
            log_fn(f"\n  [{model_name}] SKIPPED on {name}.")
            forecasts_all[model_name] = None
            metrics_all[model_name] = None
            continue
        log_fn(f"\n  [{model_name}] Loading model...")
        t0 = time.time()
        try:
            fcst_np, row = _run_one_model(
                model_name, device, seed, lookback, horizon,
                contexts, targets_t, N_windows, D, sigma,
                revin, aug, detectors,
                calib_aug_cpu, log_fn)
            forecasts_all[model_name] = fcst_np
            metrics_all[model_name] = row
            cens_str = "  ".join(
                f"CSig({l})={row[f'CSig({l})']:.4f}" for l in cens_labels)
            log_fn(f"    OK ({time.time() - t0:.1f}s)  "
                   f"CRPS={row['CRPS']:.4f}  ES={row['ES']:.4f}  "
                   f"QL={row['QL']:.4f}  VS={row['VS']:.4f}  "
                   f"SigMMD={row['SigMMD']:.4f}  {cens_str}")
        except torch.cuda.OutOfMemoryError as exc:
            log_fn(f"    OOM: {str(exc)[:200]}")
            forecasts_all[model_name] = None
            metrics_all[model_name] = None
        except Exception as exc:
            log_fn(f"    FAILED: {type(exc).__name__}: {str(exc)[:200]}")
            forecasts_all[model_name] = None
            metrics_all[model_name] = None
        finally:
            purge_cuda()

    del calib_aug_cpu, targets_t, revin, aug, detectors
    purge_cuda()

    return {
        "forecasts": forecasts_all,
        "metrics": metrics_all,
        "targets": targets.astype(np.float32),
        "last_ctx": last_ctx,
        "n_windows": N_windows,
        "horizon": horizon,
        "D": D,
        "cens_labels": cens_labels,
    }


def _run_one_model(model_name, device, seed, lookback, horizon,
                   contexts, targets_t, N_windows, D, sigma,
                   revin, aug, detectors,
                   calib_aug_cpu, log_fn):
    """Forecast → per-window metrics, returning the forecast array too.

    `detectors` is an OrderedDict { display_label: (detector, raw_aug) }
    so this runs all CSig variants in one batched pass over the windows.
    """
    model = None
    try:
        model = get_model(model_name, device, lookback, horizon, N_SAMPLES, D)
        set_seed(seed)
        forecasts = model.forecast(
            contexts, horizon, N_SAMPLES, device=device, batch_size=BATCH_SIZE)
        fcst_np = forecasts.numpy()                              # (N, D, K, H)
        del forecasts

        n_nan = int(np.isnan(fcst_np).sum())
        if n_nan > 0:
            last = contexts[:, -1, :]
            fcst_np = np.where(
                np.isnan(fcst_np),
                np.broadcast_to(last[:, :, None, None], fcst_np.shape),
                fcst_np,
            ).astype(np.float32)
            log_fn(f"    NaN samples scrubbed: {n_nan}")

        preds_bslc = torch.from_numpy(
            np.ascontiguousarray(fcst_np.transpose(0, 2, 3, 1))
        )                                                        # (N, K, H, D)

        # Standard metrics
        es_vals, crps_vals, ql_vals, vs_vals = [], [], [], []
        for i in range(0, N_windows, BATCH_SIZE):
            j = min(i + BATCH_SIZE, N_windows)
            p_batch = preds_bslc[i:j]
            t_batch = targets_t[i:j]
            es, crps, ql = compute_probabilistic_metrics(p_batch, t_batch)
            vs = compute_variogram_score(p_batch, t_batch)
            es_vals.append(es); crps_vals.append(crps)
            ql_vals.append(ql); vs_vals.append(vs)

        _drop_model_attrs(model)
        model = None
        purge_cuda()

        # Horizon-aware micro-batch (matches benchmark_fm_multi)
        if horizon <= 96:
            mb_sig, mb_cen = 4, 2
        elif horizon <= 192:
            mb_sig, mb_cen = 2, 1
        else:
            mb_sig, mb_cen = 1, 1

        with torch.no_grad():
            calib_aug_device = calib_aug_cpu.to(device)
            sig_mmd = ESKSigMMD(
                sigma=sigma, dyadic_order=1, max_batch=mb_sig).to(device)
            cens_mmds = OrderedDict()
            for d_name in detectors:
                c = CensoredESKSigMMD(
                    sigma=sigma, dyadic_order=1, max_batch=mb_cen).to(device)
                c.set_pivot(calib_aug_device)
                cens_mmds[d_name] = c
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
                    for d_name, (det, raw_input) in detectors.items():
                        if raw_input:
                            ty, px = t_aug_raw, p_aug_raw_flat
                        else:
                            ty, px = t_aug, p_aug_flat
                        mask_y = det.get_mask(ty)
                        mask_x = det.get_mask(px).view(B, S)
                        cens_mmds[d_name].update(p_aug, t_aug, mask_x, mask_y)

                    del (p_batch, t_batch, t_norm, t_aug, p_flat, p_norm,
                         p_aug_flat, p_aug, t_aug_raw, p_aug_raw_flat)

                sig_val = sig_mmd.compute().item()
                cen_vals = {d: c.compute().item()
                            for d, c in cens_mmds.items()}
            finally:
                sig_mmd = None
                for k in list(cens_mmds.keys()):
                    cens_mmds[k] = None
                cens_mmds = None
                calib_aug_device = None
                preds_bslc = None

        row = {
            "CRPS": float(np.mean(crps_vals)),
            "ES":   float(np.mean(es_vals)),
            "QL":   float(np.mean(ql_vals)),
            "VS":   float(np.mean(vs_vals)),
            "SigMMD": sig_val,
        }
        for d_name, v in cen_vals.items():
            row[f"CSig({d_name})"] = v
        return fcst_np, row
    finally:
        _drop_model_attrs(model)
        model = None


# ═══════════════════════════════════════════════════════════════════════════
# Reporting
# ═══════════════════════════════════════════════════════════════════════════

METRIC_KEYS_BASE = ["CRPS", "ES", "QL", "VS", "SigMMD"]   # CSig(<label>) appended per scenario


def print_sweep_table(scenario_name, sweep_label, sweep_val,
                      utility, metrics_all, cens_labels, log_fn):
    """One block: per-forecaster row table + ranking-agreement summary."""
    log_fn(f"\n  {sweep_label} = {sweep_val}")
    log_fn(f"  {'─' * 92}")

    metric_keys = METRIC_KEYS_BASE + [f"CSig({l})" for l in cens_labels]
    hdr = f"  {'Forecaster':<12}"
    for k in metric_keys:
        hdr += f" | {k:>13}"
    hdr += f" | {'Realised Ū':>13}"
    log_fn(hdr)
    log_fn("  " + "-" * (len(hdr) - 2))

    valid_models = [m for m in MODELS
                    if metrics_all.get(m) is not None and m in utility]
    if not valid_models:
        log_fn("  (no valid forecasters)")
        return None

    best_u = max(utility[m] for m in valid_models)
    for m in MODELS:
        row_str = f"  {m:<12}"
        if metrics_all.get(m) is None:
            row_str += f" | {'—':>13}" * len(metric_keys) + f" | {'—':>13}"
            log_fn(row_str)
            continue
        for k in metric_keys:
            row_str += f" | {metrics_all[m][k]:>13.5f}"
        u = utility[m]
        marker = " *" if abs(u - best_u) < 1e-12 else "  "
        row_str += f" | {u:>+12.5f}{marker}"
        log_fn(row_str)

    metric_vals_dict = {
        k: [metrics_all[m][k] for m in valid_models] for k in metric_keys
    }
    util_vals = [utility[m] for m in valid_models]

    log_fn("")
    log_fn(f"  Ranking agreement (top-weighted Kendall τ vs realised Ū, "
           f"top-1 hit):")
    tau_top = OrderedDict()
    for k in metric_keys:
        tau = weighted_tau_lower_better(metric_vals_dict[k], util_vals)
        top = top1_agreement(metric_vals_dict[k], util_vals)
        tau_top[k] = (tau, top)
        winner_idx = int(np.argmin(metric_vals_dict[k]))
        winner = valid_models[winner_idx]
        check = "✓" if top else "✗"
        log_fn(f"    {k:<14} : τ={tau:>+6.3f}   top-1: {check} (picks {winner})")
    return {
        "valid_models": valid_models,
        "utility": {m: float(utility[m]) for m in valid_models},
        "metrics": {k: {m: float(metrics_all[m][k]) for m in valid_models}
                    for k in metric_keys},
        "tau_top": {k: {"tau": float(v[0]), "top1": int(v[1])}
                    for k, v in tau_top.items()},
    }


def print_aggregate_summary(per_seed_tau_top, log_fn):
    """Aggregate ranking agreement across seeds.

    per_seed_tau_top : {seed: {scen: {sweep_val: {metric: {"tau", "top1"}}}}}

    Two views per scenario:
      1. Per-sweep:  mean ± std of τ across seeds, hit rate = Σ top1 / n_seeds
      2. Across-sweep: same, pooling seeds × sweep values

    τ is a *top-weighted* Kendall correlation (scipy.stats.weightedtau)
    so disagreements at the top of the ranking dominate.
    """
    if not per_seed_tau_top:
        log_fn("\n  (no seeds completed)")
        return
    seeds = list(per_seed_tau_top.keys())

    # Reorganise to {scen: {sweep_val: {metric: {"taus": [...], "tops": [...]}}}}
    scen_set = OrderedDict()
    for s in seeds:
        for scen, by_sweep in per_seed_tau_top[s].items():
            scen_set.setdefault(scen, OrderedDict())
            for sv, by_metric in by_sweep.items():
                scen_set[scen].setdefault(sv, OrderedDict())
                for k, rt in by_metric.items():
                    scen_set[scen][sv].setdefault(
                        k, {"taus": [], "tops": []})
                    scen_set[scen][sv][k]["taus"].append(rt["tau"])
                    scen_set[scen][sv][k]["tops"].append(rt["top1"])

    log_fn(f"\n{'═' * 95}")
    log_fn(f"AGGREGATE across {len(seeds)} seeds: {seeds}")
    log_fn(f"{'═' * 95}")

    aggregate_dump = OrderedDict()
    for scen, by_sweep in scen_set.items():
        sweep_vals = list(by_sweep.keys())
        if not sweep_vals:
            continue
        metric_keys = list(next(iter(by_sweep.values())).keys())
        sweep_label = "γ" if scen == "exchange" else "α"

        log_fn(f"\n  {scen}  ({sweep_label} ∈ {sweep_vals})")
        log_fn(f"  {'─' * 92}")
        log_fn(f"    {'Sweep':>8}  {'Metric':<14} | "
               f"{'weighted-τ (mean ± std)':>26} | {'top-1 hit rate':>17}")
        log_fn(f"    {'-' * 8}  {'-' * 14}-+-{'-' * 26}-+-{'-' * 17}")
        scen_dump = OrderedDict()
        for sv in sweep_vals:
            sv_dump = OrderedDict()
            for k in metric_keys:
                taus = by_sweep[sv][k]["taus"]
                tops = by_sweep[sv][k]["tops"]
                mu = float(np.mean(taus))
                sd = float(np.std(taus, ddof=0))
                hit = f"{sum(tops)}/{len(tops)}"
                log_fn(f"    {sv:>8}  {k:<14} | "
                       f"{mu:>+10.3f} ± {sd:>10.3f} | {hit:>17}")
                sv_dump[k] = {
                    "tau_mean": mu, "tau_std": sd,
                    "taus_per_seed": [float(x) for x in taus],
                    "top1_hits": int(sum(tops)),
                    "top1_total": int(len(tops)),
                }
            scen_dump[str(sv)] = sv_dump

        # Cross-sweep pool
        log_fn(f"\n    {'pooled':>8}  {'Metric':<14} | "
               f"{'weighted-τ (mean ± std)':>26} | {'top-1 hit rate':>17}")
        log_fn(f"    {'-' * 8}  {'-' * 14}-+-{'-' * 26}-+-{'-' * 17}")
        across_dump = OrderedDict()
        for k in metric_keys:
            taus_all = sum(
                (by_sweep[sv][k]["taus"] for sv in sweep_vals), [])
            tops_all = sum(
                (by_sweep[sv][k]["tops"] for sv in sweep_vals), [])
            mu = float(np.mean(taus_all))
            sd = float(np.std(taus_all, ddof=0))
            hit = f"{sum(tops_all)}/{len(tops_all)}"
            log_fn(f"    {'pooled':>8}  {k:<14} | "
                   f"{mu:>+10.3f} ± {sd:>10.3f} | {hit:>17}")
            across_dump[k] = {
                "tau_mean": mu, "tau_std": sd,
                "top1_hits": int(sum(tops_all)),
                "top1_total": int(len(tops_all)),
            }

        aggregate_dump[scen] = {
            "by_sweep": scen_dump,
            "pooled": across_dump,
        }

    return aggregate_dump


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _parse_seeds(s):
    if s is None:
        return list(SEEDS)
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=str, default=None,
                   help="Comma-separated seeds; default = SEEDS constant.")
    p.add_argument("--only_scenario", type=str, default=None,
                   choices=list(SCENARIOS.keys()))
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device
    seeds = _parse_seeds(args.seeds)

    writeable = os.environ.get(
        "WRITEABLE_DIR", os.path.join(_sigrisk_root, "results"))
    out_dir = args.output_dir or os.path.join(
        writeable, "agent_decision_experiment")
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "agent_decision.log")
    json_path = os.path.join(out_dir, "raw_results.json")
    if os.path.exists(log_path):
        os.remove(log_path)

    def log_fn(*s):
        msg = " ".join(str(x) for x in s)
        print(msg, flush=True)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    log_fn("Agent decision experiment")
    log_fn(f"  Seeds:      {seeds}")
    log_fn(f"  Device:     {device}")
    log_fn(f"  Models:     {MODELS}")
    log_fn(f"  Sweep γ:    {GAMMAS}")
    log_fn(f"  Sweep α:    {ALPHAS}")

    scenarios_to_run = (
        [args.only_scenario] if args.only_scenario else list(SCENARIOS.keys()))
    log_fn(f"  Scenarios:  {scenarios_to_run}")

    per_seed_tau_top = OrderedDict()      # {seed: {scen: {sv: {metric: {tau,top1}}}}}
    per_seed_raw = OrderedDict()          # JSON dump per seed

    for seed in seeds:
        log_fn(f"\n\n{'#' * 95}")
        log_fn(f"# SEED {seed}")
        log_fn(f"{'#' * 95}")
        set_seed(seed)
        per_seed_tau_top[seed] = OrderedDict()
        per_seed_raw[seed] = OrderedDict()

        for scen_name in scenarios_to_run:
            cfg = SCENARIOS[scen_name]
            result = run_scenario(scen_name, cfg, device, seed, log_fn)
            if result is None:
                continue
            forecasts = result["forecasts"]
            metrics = result["metrics"]
            targets_np = result["targets"]
            last_ctx = result["last_ctx"]
            cens_labels = result["cens_labels"]

            sweep_vals = GAMMAS if scen_name == "exchange" else ALPHAS
            sweep_label = "γ" if scen_name == "exchange" else "α"
            log_fn(f"\n  Sweeping {sweep_label} ∈ {sweep_vals}")

            tau_top_by_sweep = OrderedDict()
            per_sweep_dump = OrderedDict()
            for sv in sweep_vals:
                utility = OrderedDict()
                for m in MODELS:
                    if forecasts.get(m) is None:
                        continue
                    if scen_name == "exchange":
                        u_per_window = realised_mv_utility(
                            forecasts[m], targets_np, last_ctx, gamma=sv)
                    else:
                        u_per_window = realised_newsvendor_utility(
                            forecasts[m], targets_np, alpha=sv)
                    utility[m] = float(np.mean(u_per_window))

                block = print_sweep_table(
                    scen_name, sweep_label, sv,
                    utility, metrics, cens_labels, log_fn)
                if block is not None:
                    tau_top_by_sweep[sv] = block["tau_top"]
                    per_sweep_dump[str(sv)] = block

            per_seed_tau_top[seed][scen_name] = tau_top_by_sweep
            per_seed_raw[seed][scen_name] = {
                "config": {k: v for k, v in cfg.items() if k != "path"},
                "n_windows": result["n_windows"],
                "horizon": result["horizon"],
                "D": result["D"],
                "cens_labels": cens_labels,
                "by_sweep": per_sweep_dump,
            }

    aggregate = print_aggregate_summary(per_seed_tau_top, log_fn)

    raw_dump = {
        "seeds": seeds,
        "per_seed": {str(s): per_seed_raw[s] for s in seeds},
        "aggregate": aggregate,
    }
    with open(json_path, "w") as f:
        json.dump(raw_dump, f, indent=2)
    log_fn(f"\n  Raw results: {json_path}")
    log_fn(f"  Log:         {log_path}")


if __name__ == "__main__":
    main()