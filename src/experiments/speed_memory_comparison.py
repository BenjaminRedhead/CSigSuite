"""
Per-sequence speed and memory comparison: ESK vs baseline SigKernel,
*as a function of batch size*. Both backends use a **Laplacian** static
kernel (k(x, y) = exp(−‖x − y‖ / σ)) so the comparison reflects the
project's current default static kernel rather than legacy RBF.

The baseline `sigkernel.SigKernel` was designed for one path-pair set
at a time — it has no native multi-window batch dimension, so the only
honest way to evaluate it on B sequences is a Python `for` loop:

    for b in range(B):
        kernel.compute_Gram(X[b], X[b], sym=True)

Each call peaks at the per-sequence Gram allocation; subsequent calls
release it before the next iteration, so the peak doesn't accumulate.

`sigkernel_mmd` (ESK) was designed *for* this batched workload — it
processes all B windows in a single call:

    kernel.compute_diag_Gram((B, S, L, D), (B, S, L, D))

→ produces (B, S, S) block-diagonal Gram in one shot, paying one
allocation that scales linearly in B.

Per-sequence quantities (the things that should be compared):
    speed-per-sequence  = total wall time / B
    memory-per-sequence : ESK   = peak_total_alloc / B   (amortised)
                          Base  = peak_per_call          (loop is sequential)

We sweep B over BATCH_SIZES on real Exchange and Weather windows. Same
sigma (median heuristic), same Laplacian static kernel, same
dyadic_order, same S forecast samples per window. Output is a 2-panel
plot per dataset of per-sequence wall time and per-sequence memory vs B
(log-log) plus a CSV.
"""

import os
import sys
import gc
import time
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", message=".*weights_only=False.*")
warnings.filterwarnings("ignore", message=".*non-tuple sequence.*")


# ── Path setup ─────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
_bundle_root = os.path.abspath(os.path.join(_here, "../.."))
_sigrisk_root = _bundle_root        # bundle root doubles as the data root
_esk_root = os.path.join(_bundle_root, "ESK")
_sigkernel_root = os.path.join(_bundle_root, "sigkernel")
for p in [_bundle_root, _esk_root, _sigkernel_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── ESK ─────────────────────────────────────────────────────────────────
from sigkernel_mmd import (
    SigKernel as ESKSigKernelMod,
    PathReVIN, AugmentPipeline, SigmaEstimator,
)
from src.metrics.eval_metrics import LaplacianKernel as ESKLaplacian

# ── Baseline SigKernel ─────────────────────────────────────────────────
try:
    from sigkernel.sigkernel import SigKernel as BaseSigKernel
except ImportError as e:
    print(f"FATAL: could not import baseline sigkernel package "
          f"(checked {_sigkernel_root}). Did you `pip install -e` it "
          f"into the venv? Underlying error:\n  {e}", flush=True)
    sys.exit(2)


# ── Laplacian static kernel matching Salvi's static-kernel API ────────
# Salvi's `sigkernel` does not ship a Laplacian (only RBF / RBF_CEXP /
# RBF_SQR). We define one here with the exact `batch_kernel` /
# `Gram_matrix` signatures `BaseSigKernel` consumes so the baseline runs
# under the same static kernel as ESK.
class BaseLaplacianKernel:
    """Laplacian static kernel  k(x, y) = exp(−‖x − y‖ / σ)  for Salvi's
    SigKernel. `σ` is the same length-scale parameter as Salvi's RBF
    (which is `exp(−‖x − y‖² / σ)`)."""

    def __init__(self, sigma):
        self.sigma = sigma

    @staticmethod
    def _sq_dists_batch(X, Y):
        Xs = torch.sum(X ** 2, dim=2)
        Ys = torch.sum(Y ** 2, dim=2)
        d = -2.0 * torch.bmm(X, Y.permute(0, 2, 1))
        d = d + Xs.unsqueeze(-1) + Ys.unsqueeze(-2)
        return d.clamp(min=0.0)

    @staticmethod
    def _sq_dists_gram(X, Y):
        Xs = torch.sum(X ** 2, dim=2)
        Ys = torch.sum(Y ** 2, dim=2)
        d = -2.0 * torch.einsum('ipk,jqk->ijpq', X, Y)
        d = d + Xs.reshape(X.shape[0], 1, X.shape[1], 1) \
              + Ys.reshape(1, Y.shape[0], 1, Y.shape[1])
        return d.clamp(min=0.0)

    def batch_kernel(self, X, Y):
        dist = torch.sqrt(self._sq_dists_batch(X, Y) + 1e-12)
        return torch.exp(-dist / self.sigma)

    def Gram_matrix(self, X, Y):
        dist = torch.sqrt(self._sq_dists_gram(X, Y) + 1e-12)
        return torch.exp(-dist / self.sigma)


# ── Settings ──────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(
    os.environ.get("WRITEABLE_DIR",
                   os.path.join(_sigrisk_root, "Results")),
    "speed_memory_comparison")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASETS = {
    "exchange_rate": os.path.join(
        _sigrisk_root,
        "data/long_term_forecast/exchange_rate/exchange_rate.csv"),
    "weather": os.path.join(
        _sigrisk_root,
        "data/long_term_forecast/weather/weather.csv"),
    "cloud": os.path.join(
        _sigrisk_root, "data/cloud/cloud.csv"),
}
HORIZON = 96
N_SAMPLES = 10                                # forecast samples per window
BATCH_SIZES = [1, 8, 16, 32, 64]
N_TOTAL_WINDOWS = max(BATCH_SIZES)            # need at least max(B) windows
DYADIC_ORDER = 1
WARMUP_BS = [1]                               # warm up at B=1 first
N_REPEATS = 3                                 # repeat each B point and average
SEED = 2026
PAPER_COLORS = ['#76b5c5', '#eab676', '#87ace1', '#b39eb5', '#9fd9b3']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════

def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df = df.drop(columns=["date"])
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])
    df = (df.select_dtypes(include=[np.number])
            .dropna(axis=1, how="all").ffill().bfill())
    return df.values.astype(np.float32)


def make_windows(data, horizon, n_windows, seed):
    T = data.shape[0]
    rng = np.random.default_rng(seed)
    starts = rng.choice(T - horizon, size=min(n_windows, T - horizon),
                        replace=False)
    return np.stack([data[s:s + horizon] for s in starts], axis=0)


def perturbed_samples(target_np, n_samples, scale=0.05, seed=0):
    rng = np.random.default_rng(seed)
    sigma_per = np.std(target_np, axis=0, keepdims=True) * scale + 1e-8
    noise = rng.normal(
        0, 1, size=(n_samples, *target_np.shape)).astype(np.float32)
    return target_np[None, :, :] + noise * sigma_per


# ══════════════════════════════════════════════════════════════════════
# Timing helpers
# ══════════════════════════════════════════════════════════════════════

def _sync():
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def _reset_peak():
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

def _peak_mb():
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    return 0.0

def _purge():
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


@torch.no_grad()
def time_esk(preds_BSLD, sigma, dyadic_order):
    """ESK: ONE call to compute_diag_Gram on a B-batched input.
    `preds_BSLD` shape (B, S, L_aug, D_aug). Returns (val, time_s, peak_mb)."""
    static = ESKLaplacian(sigma=sigma)
    kernel = ESKSigKernelMod(static, dyadic_order=dyadic_order)

    _sync(); _reset_peak()
    t0 = time.time()
    K = kernel.compute_diag_Gram(preds_BSLD, preds_BSLD)   # (B, S, S)
    _sync()
    elapsed = time.time() - t0
    peak = _peak_mb()

    val = K.mean().item()
    del K, kernel, static
    _purge()
    return val, elapsed, peak


@torch.no_grad()
def time_baseline_loop(preds_BSLD, sigma, dyadic_order):
    """Baseline: Python loop over the B windows, calling compute_Gram
    once per window. Total time is the sum, peak memory is the max
    across single-call peaks (loop is sequential so they don't pile up).
    """
    static = BaseLaplacianKernel(sigma=sigma)
    kernel = BaseSigKernel(static, dyadic_order=dyadic_order)
    B = preds_BSLD.shape[0]

    total_time = 0.0
    peak_per_call = 0.0
    last_val = 0.0
    for b in range(B):
        X = preds_BSLD[b].contiguous()                 # (S, L_aug, D_aug)
        _sync(); _reset_peak()
        t0 = time.time()
        K = kernel.compute_Gram(X, X, sym=True, max_batch=X.size(0))
        _sync()
        total_time += time.time() - t0
        peak_per_call = max(peak_per_call, _peak_mb())
        last_val = K.mean().item()
        del K
        _purge()

    del kernel, static
    _purge()
    return last_val, total_time, peak_per_call


# ══════════════════════════════════════════════════════════════════════
# Per-dataset experiment: B sweep
# ══════════════════════════════════════════════════════════════════════

def run_dataset(name, csv_path):
    print(f"\n=== {name} ===", flush=True)
    data = load_dataset(csv_path)
    T, D = data.shape
    print(f"  raw shape ({T}, {D})", flush=True)

    # Generate the union of windows once.
    windows_np = make_windows(data, HORIZON, N_TOTAL_WINDOWS, seed=SEED)
    print(f"  prepared {len(windows_np)} windows × H={HORIZON} × D={D}",
          flush=True)

    # Calibration tensors on the full window set.
    train_t = torch.tensor(windows_np, dtype=torch.float32, device=device)
    revin = PathReVIN(D).to(device)
    revin.fit(train_t)
    train_norm = revin(train_t)
    aug = AugmentPipeline(append_time=True).to(device)
    sigma = SigmaEstimator.fit(train_norm)
    print(f"  sigma={sigma:.4f}", flush=True)

    # Pre-build the (N_TOTAL, S, L_aug, D_aug) tensor of normalised+aug
    # forecast samples — batched slices come from this.
    rng = np.random.default_rng(SEED)
    all_preds_aug = []
    for w_np in windows_np:
        preds_raw = perturbed_samples(
            w_np, N_SAMPLES, scale=0.05,
            seed=int(rng.integers(1 << 30)))
        preds_t = torch.tensor(preds_raw, dtype=torch.float32, device=device)
        preds_aug = aug(revin(preds_t))                # (S, L_aug, D_aug)
        all_preds_aug.append(preds_aug)
    all_preds_aug = torch.stack(all_preds_aug, dim=0)  # (N_TOTAL, S, L', D')
    print(f"  preds_aug stack shape {tuple(all_preds_aug.shape)}",
          flush=True)

    # Warm-up at B=1 to JIT-compile both backends.
    print(f"  [warmup at B={WARMUP_BS}]", flush=True)
    for B in WARMUP_BS:
        chunk = all_preds_aug[:B]
        _ = time_esk(chunk, sigma, DYADIC_ORDER)
        _ = time_baseline_loop(chunk, sigma, DYADIC_ORDER)

    # Sweep.
    rows = []
    for B in BATCH_SIZES:
        chunk = all_preds_aug[:B]
        for rep in range(N_REPEATS):
            try:
                _, t_e, m_e = time_esk(chunk, sigma, DYADIC_ORDER)
            except torch.cuda.OutOfMemoryError as exc:
                print(f"    B={B:>3d}  ESK OOM: {str(exc)[:100]}", flush=True)
                t_e, m_e = float('nan'), float('nan')
                _purge()
            try:
                _, t_b, m_b = time_baseline_loop(chunk, sigma, DYADIC_ORDER)
            except torch.cuda.OutOfMemoryError as exc:
                print(f"    B={B:>3d}  Base OOM: {str(exc)[:100]}", flush=True)
                t_b, m_b = float('nan'), float('nan')
                _purge()

            rows.append({
                "dataset":       name,
                "batch_size":    B,
                "repeat":        rep,
                "esk_time_s":    t_e,
                "esk_peak_mb":   m_e,                 # total peak across all B
                "esk_per_seq_time_ms": (t_e / B) * 1000 if t_e == t_e else float('nan'),
                "esk_per_seq_mb":      m_e / B       if m_e == m_e else float('nan'),
                "base_time_s":   t_b,
                "base_peak_mb":  m_b,                 # max single-call peak
                "base_per_seq_time_ms": (t_b / B) * 1000 if t_b == t_b else float('nan'),
                "base_per_seq_mb":     m_b           if m_b == m_b else float('nan'),
            })
            print(f"    B={B:>3d} rep={rep}  "
                  f"ESK: {t_e * 1000:>8.1f}ms total, "
                  f"{m_e:>7.1f}MB peak  →  "
                  f"{(t_e / B) * 1000:>7.2f}ms/seq, "
                  f"{m_e / B:>7.2f}MB/seq    "
                  f"|  Base: {t_b * 1000:>8.1f}ms total, "
                  f"{m_b:>7.1f}MB peak  →  "
                  f"{(t_b / B) * 1000:>7.2f}ms/seq, "
                  f"{m_b:>7.2f}MB/seq", flush=True)

    df = pd.DataFrame(rows)
    return df


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("Per-sequence speed/memory vs batch-size: ESK vs SigKernel",
          flush=True)
    print(f"  device         = {device}", flush=True)
    print(f"  horizon        = {HORIZON}", flush=True)
    print(f"  n_samples (S)  = {N_SAMPLES}", flush=True)
    print(f"  batch sizes    = {BATCH_SIZES}", flush=True)
    print(f"  n_repeats      = {N_REPEATS}", flush=True)
    print(f"  dyadic_order   = {DYADIC_ORDER}", flush=True)
    print(f"  output dir     = {OUTPUT_DIR}", flush=True)

    for name, p in DATASETS.items():
        if not os.path.isfile(p):
            print(f"FATAL: dataset CSV not found: {p}", flush=True)
            sys.exit(2)

    all_dfs = []
    for name, path in DATASETS.items():
        all_dfs.append(run_dataset(name, path))
    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(
        os.path.join(OUTPUT_DIR, "speed_memory_per_window.csv"),
        index=False)

    # Summarise: mean over repeats per (dataset, B).
    summary = (combined.groupby(['dataset', 'batch_size'])
               .agg(esk_per_seq_time_ms=('esk_per_seq_time_ms', 'mean'),
                    esk_per_seq_mb     =('esk_per_seq_mb',      'mean'),
                    esk_peak_mb        =('esk_peak_mb',         'mean'),
                    base_per_seq_time_ms=('base_per_seq_time_ms', 'mean'),
                    base_per_seq_mb    =('base_per_seq_mb',     'mean'),
                    base_peak_mb       =('base_peak_mb',        'mean'))
               .reset_index())
    summary['speedup_x'] = (summary['base_per_seq_time_ms']
                            / summary['esk_per_seq_time_ms'])
    summary['mem_ratio_x'] = (summary['base_per_seq_mb']
                              / summary['esk_per_seq_mb'])
    summary.to_csv(
        os.path.join(OUTPUT_DIR, "speed_memory_summary.csv"), index=False)

    print("\n" + "=" * 110)
    print("SUMMARY (mean over repeats)".center(110))
    print("=" * 110)
    print(summary.to_string(index=False))
    print("=" * 110)

    # ── Plot — 2 columns × 2 rows: per-seq time and per-seq memory, ──
    # one row per dataset.
    n_ds = len(DATASETS)
    fig, axes = plt.subplots(
        n_ds, 2, figsize=(7.5, 2.6 * n_ds),
        sharex=True, constrained_layout=True)
    if n_ds == 1:
        axes = axes.reshape(1, 2)

    for row_i, name in enumerate(DATASETS):
        sub = summary[summary['dataset'] == name].sort_values('batch_size')

        # Left: per-seq wall time vs B
        ax = axes[row_i, 0]
        ax.plot(sub['batch_size'], sub['esk_per_seq_time_ms'],
                color=PAPER_COLORS[0], marker='o', markersize=4,
                linewidth=1.8, label='ESK (Laplacian)')
        ax.plot(sub['batch_size'], sub['base_per_seq_time_ms'],
                color=PAPER_COLORS[1], marker='s', markersize=4,
                linewidth=1.8, label='SigKernel (Laplacian)')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('Batch size $B$  (windows per call)')
        ax.set_ylabel('Per-sequence wall time (ms)')
        ax.set_title(f"{name} — per-sequence time")
        ax.grid(True, which='both', linestyle=':', alpha=0.5)
        ax.legend(fontsize=8, loc='best')
        # Annotate the speed-up at the largest B that has finite values.
        finite = sub[np.isfinite(sub['speedup_x'])]
        if len(finite):
            B_last = int(finite['batch_size'].iloc[-1])
            sp_last = float(finite['speedup_x'].iloc[-1])
            ax.text(0.95, 0.05,
                    f"@ B={B_last}: {sp_last:.1f}× speed-up",
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white', alpha=0.7,
                              edgecolor='gray', linewidth=0.5))

        # Right: per-seq memory vs B
        ax = axes[row_i, 1]
        ax.plot(sub['batch_size'], sub['esk_per_seq_mb'],
                color=PAPER_COLORS[0], marker='o', markersize=4,
                linewidth=1.8, label='ESK (peak / B)')
        ax.plot(sub['batch_size'], sub['base_per_seq_mb'],
                color=PAPER_COLORS[1], marker='s', markersize=4,
                linewidth=1.8, label='SigKernel (per-call peak)')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('Batch size $B$  (windows per call)')
        ax.set_ylabel('Per-sequence peak memory (MB)')
        ax.set_title(f"{name} — per-sequence memory")
        ax.grid(True, which='both', linestyle=':', alpha=0.5)
        ax.legend(fontsize=8, loc='best')
        finite = sub[np.isfinite(sub['mem_ratio_x'])]
        if len(finite):
            B_last = int(finite['batch_size'].iloc[-1])
            mr_last = float(finite['mem_ratio_x'].iloc[-1])
            ax.text(0.95, 0.05,
                    f"@ B={B_last}: {mr_last:.2f}× SigKernel/ESK",
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='white', alpha=0.7,
                              edgecolor='gray', linewidth=0.5))

    fig.suptitle(
        f"Per-sequence speed & memory vs batch size — Laplacian static kernel  "
        f"(S={N_SAMPLES}, L={HORIZON}, dyadic={DYADIC_ORDER})",
        fontsize=10, y=1.02)
    plt_path = os.path.join(OUTPUT_DIR, "speed_memory_vs_batch.pdf")
    plt.savefig(plt_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"\nPer-window CSV : "
          f"{os.path.join(OUTPUT_DIR, 'speed_memory_per_window.csv')}")
    print(f"Summary CSV    : "
          f"{os.path.join(OUTPUT_DIR, 'speed_memory_summary.csv')}")
    print(f"Plot           : {plt_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
