"""
Dependency sensitivity experiment — final version (GP only).

Standard metrics (ES, CRPS, QL, VS), flat RBF-MMD, Sig-MMD, and three
censored Sig-MMD variants:
  - CSig(Kernel)   — kernel mean embedding censoring
  - CSig(Sharpe)   — Markowitz mean-variance utility censoring
                     (U* = (1/2γ) μ^T Σ^{-1} μ)
  - CSig(Systemic) — aggregate systemic load exceedance

All censored variants use the mean training path as pivot.

Seeds: [2021, 2022, 2023, 2024, 2025] — mean ± std reported across seeds.
"""

import torch
import numpy as np
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_bundle_root = os.path.abspath(os.path.join(_here, "../.."))
_esk_root = os.path.join(_bundle_root, "ESK")
_sigkernel_root = os.path.join(_bundle_root, "sigkernel")
for _p in [_bundle_root, _esk_root, _sigkernel_root]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.metrics.eval_metrics import (
    ESKSigMMD, CensoredESKSigMMD, make_tail_detector,
    AugmentPipeline, PathReVIN, SigmaEstimator,
)

SEEDS = [2021, 2022, 2023, 2024, 2025]


# ═══════════════════════════════════════════════════════════════════════
# GP Generator
# ═══════════════════════════════════════════════════════════════════════

class GPGenerator:
    def __init__(self, n_features=3, length=30, length_scale=5.0):
        self.D, self.L = n_features, length
        t = torch.arange(length).float().view(-1, 1)
        dists = torch.cdist(t, t)**2
        self.K_t = torch.exp(-dists / (2 * length_scale**2)) + 1e-4 * torch.eye(length)
        temp = torch.randn(n_features, n_features); Sigma = temp @ temp.T
        d = torch.sqrt(torch.diag(Sigma))
        self.Sigma_s = Sigma / (d.unsqueeze(1) @ d.unsqueeze(0))
        self.Cov_full = torch.kron(self.K_t, self.Sigma_s) + 1e-6 * torch.eye(self.L * self.D)
        self.dist_gt = torch.distributions.MultivariateNormal(
            torch.zeros(self.L * self.D), self.Cov_full)

    def get_sample(self, bs, mode='gt'):
        if mode == 'gt':
            return self.dist_gt.sample((bs,)).view(bs, self.L, self.D)
        elif mode == 'no_temp':
            d = torch.distributions.MultivariateNormal(torch.zeros(self.D), self.Sigma_s)
            return d.sample((bs, self.L))
        elif mode == 'indep':
            stds = torch.sqrt(torch.diag(self.Sigma_s)).view(1, 1, -1)
            return torch.randn(bs, self.L, self.D) * stds
        else:
            raise ValueError(f"Unknown mode: {mode}")


# ═══════════════════════════════════════════════════════════════════════
# Traditional metrics
# ═══════════════════════════════════════════════════════════════════════

def compute_traditional_metrics(preds, target):
    preds, target = preds.cpu(), target.cpu()
    target_exp = target.unsqueeze(1)
    preds_roll = torch.roll(preds, 1, dims=1)
    d_t = torch.norm(preds - target_exp, dim=-1).mean(-1).mean(1)
    d_i = torch.norm(preds - preds_roll, dim=-1).mean(-1).mean(1)
    es = (d_t - 0.5 * d_i).mean().item()
    diff = torch.abs(preds - target_exp).mean(1)
    intr = torch.abs(preds - preds_roll).mean(1)
    crps = (diff - 0.5 * intr).mean().item()
    ql = sum(torch.max((q-1)*(target_exp - torch.quantile(preds, q, dim=1, keepdim=True)),
                        q*(target_exp - torch.quantile(preds, q, dim=1, keepdim=True))).mean().item()
             for q in [0.1, 0.5, 0.9])
    td = torch.abs(target.unsqueeze(-1) - target.unsqueeze(-2)).pow(0.5)
    pd = torch.abs(preds.unsqueeze(-1) - preds.unsqueeze(-2)).pow(0.5)
    vs = (td - pd.mean(1)).pow(2).sum((-1, -2)).mean().item()
    return es, crps, ql, vs


def compute_rbf_mmd(x, y, tile_size=4096, sigma_subsample=4096):
    """O(N²)-time, O(tile_size²)-peak-memory RBF MMD.

    Equivalent to the naive cdist version up to float32 rounding: pairwise
    distance matrices are built in `tile_size × tile_size` blocks, kernel
    contributions are accumulated in a fp64 scalar, and σ is estimated via
    the median heuristic on a sub-sample (default 4096 paths from each of
    x and y, matching the full-Gram median to within sub-sampling
    variance).
    """
    device = x.device
    x_f = x.view(x.size(0), -1)
    y_f = y.view(y.size(0), -1)
    Nx, Ny = x_f.size(0), y_f.size(0)

    # Small-input fast path — original O(N²) memory is fine.
    if max(Nx, Ny) <= tile_size:
        with torch.no_grad():
            xx = torch.cdist(x_f, x_f, p=2) ** 2
            yy = torch.cdist(y_f, y_f, p=2) ** 2
            xy = torch.cdist(x_f, y_f, p=2) ** 2
            combined = torch.cat([xx.view(-1), yy.view(-1), xy.view(-1)])
            sig = (torch.median(combined[combined > 0])
                   if combined.max() > 0
                   else torch.tensor(1.0, device=device))
            g = 1.0 / (2 * sig + 1e-8)
            return (torch.exp(-g * xx).mean()
                    + torch.exp(-g * yy).mean()
                    - 2 * torch.exp(-g * xy).mean()).item()

    # ── 1. Estimate σ via median heuristic on a sub-sample ──
    with torch.no_grad():
        n_sub = min(sigma_subsample, Nx, Ny)
        ix = torch.randperm(Nx, device=device)[:n_sub]
        iy = torch.randperm(Ny, device=device)[:n_sub]
        d_xx = torch.cdist(x_f[ix], x_f[ix], p=2) ** 2
        d_yy = torch.cdist(y_f[iy], y_f[iy], p=2) ** 2
        d_xy = torch.cdist(x_f[ix], y_f[iy], p=2) ** 2
        combined = torch.cat([d_xx.view(-1), d_yy.view(-1), d_xy.view(-1)])
        sig = (torch.median(combined[combined > 0])
               if combined.max() > 0
               else torch.tensor(1.0, device=device))
    g = 1.0 / (2 * sig + 1e-8)
    del d_xx, d_yy, d_xy, combined

    # ── 2. Accumulate kernel sums tile-by-tile in fp64 ──
    def _tiled_mean(a, b):
        Na, Nb = a.size(0), b.size(0)
        total = torch.zeros((), device=a.device, dtype=torch.float64)
        with torch.no_grad():
            for i in range(0, Na, tile_size):
                ai = a[i:i + tile_size]
                for j in range(0, Nb, tile_size):
                    bj = b[j:j + tile_size]
                    d2 = torch.cdist(ai, bj, p=2) ** 2
                    total = total + torch.exp(-g * d2).sum().double()
        return (total / float(Na * Nb)).item()

    K_xx = _tiled_mean(x_f, x_f)
    K_yy = _tiled_mean(y_f, y_f)
    K_xy = _tiled_mean(x_f, y_f)

    # Hand back a clean GPU state to the next caller. The tight tiled loop
    # produces hundreds of small caching-allocator blocks; without an
    # explicit sync + empty_cache, a subsequent large allocation (like
    # SigMMD's PDE solver) can land on a fragmented heap and trigger
    # "illegal memory access" rather than a clean OOM.
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return K_xx + K_yy - 2.0 * K_xy


# ═══════════════════════════════════════════════════════════════════════
# SigMMD helpers
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_sig_mmd(X_aug, Y_aug, sigma, device):
    """Compute SigMMD with the same breakdown applied in
    ``benchmark_fm_multi._run_one_model``:

    1. ``torch.no_grad()`` decorator suppresses
       ``SigKernelDiagFn.forward``'s ``ctx.save_for_backward(M, K, …)``,
       so the float64 PDE intermediates are freed at forward exit
       instead of accumulating across update chunks.
    2. ``max_batch=1`` keeps per-chunk memory of the (mb, S², L², 8 B)
       Gram bounded — safe even on a fragmented heap.
    3. Synchronize + empty_cache before returning so the next caller
       (CSigMMD / next seed) starts on a clean GPU heap.
    """
    m = ESKSigMMD(sigma=sigma, dyadic_order=1, max_batch=1).to(device)
    m.update(X_aug, Y_aug)
    val = m.compute().item()
    m = None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return val


@torch.no_grad()
def eval_csig_mmd(X_aug, Y_aug, mask_x, mask_y, sigma, train_aug, device):
    """Censored SigMMD with mean training path as pivot.

    Same breakdown as ``eval_sig_mmd`` above: no_grad wrap + max_batch=1
    + post-call sync/empty_cache. The censored variant computes K_X0 /
    K_Y0 / K_00 on top of K_XX / K_XY / K_YY, so it has higher per-chunk
    memory than plain SigMMD — keeping mb=1 here is essential.
    """
    m = CensoredESKSigMMD(sigma=sigma, max_batch=1).to(device)
    m.set_pivot(train_aug)
    m.update(X_aug, Y_aug, mask_x, mask_y)
    val = m.compute().item()
    m = None
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return val


# ═══════════════════════════════════════════════════════════════════════
# Single experiment run (GP)
# ═══════════════════════════════════════════════════════════════════════

def run_single(gen, device, B_TRAIN, B_TEST, SAMPLES, L, D):
    """Run the dependency experiment on GP paths."""
    process = "GP"

    path_revin = PathReVIN(D).to(device)
    aug_pipe = AugmentPipeline(append_time=True).to(device)

    # Calibration
    train_raw_base = gen.get_sample(B_TRAIN, 'gt').to(device)
    train_mask = (torch.rand(B_TRAIN, L, D, device=device) < 0.02).float()
    train_noise = torch.randn(B_TRAIN, L, D, device=device) * 20.0
    train_raw = train_raw_base + (train_noise * train_mask)
    path_revin.fit(train_raw)
    train_norm = path_revin(train_raw)
    train_aug = aug_pipe(train_norm)

    sigma = SigmaEstimator.fit(train_norm)
    print(f"  sigma={sigma:.4f}")

    # Censoring detectors — VaR-MDD replaced by Markowitz Sharpe MV.
    detectors = {
        "Kernel": make_tail_detector("kernel", quantile=0.95, sigma=sigma),
        "Sharpe": make_tail_detector("sharpe", quantile=0.95),
        "Systemic": make_tail_detector("systemic", quantile=0.95),
    }
    for det in detectors.values():
        det.fit(train_aug)

    # GT test data
    gt_base = gen.get_sample(B_TEST, 'gt').to(device)
    gt_jm = (torch.rand(B_TEST, L, D, device=device) < 0.02).float()
    gt_jn = torch.randn(B_TEST, L, D, device=device) * 20.0
    gt_raw = gt_base + (gt_jn * gt_jm)
    gt_aug = aug_pipe(path_revin(gt_raw))

    # Forecast models
    def gen_preds(mode):
        preds = []
        for _ in range(SAMPLES):
            if mode == 'gt':       p = gen.get_sample(B_TEST, 'gt').to(device) + (gt_jn * gt_jm)
            elif mode == 'bad_tail': p = gen.get_sample(B_TEST, 'gt').to(device) + (-gt_jn * gt_jm)
            elif mode == 'no_temp': p = gen.get_sample(B_TEST, 'no_temp').to(device) + (gt_jn * gt_jm)
            elif mode == 'indep':   p = gen.get_sample(B_TEST, 'indep').to(device) + (gt_jn * gt_jm)
            preds.append(p)
        return torch.stack(preds, dim=1)

    models = {
        f"F1. Match ({process})": gen_preds('gt'),
        "F2. No Time Corr":      gen_preds('no_temp'),
        "F3. Independent":       gen_preds('indep'),
        "F4. Jump Diffusion":    gen_preds('bad_tail'),
    }

    # Pre-augment
    augmented = {}
    for fname, preds in models.items():
        preds = preds.to(device)
        B, S, Lp, Dp = preds.shape
        flat = aug_pipe(path_revin(preds.view(-1, Lp, Dp)))
        L_aug, D_aug = flat.shape[1], flat.shape[2]
        augmented[fname] = flat.view(B, S, L_aug, D_aug)

    L_aug = gt_aug.shape[1]
    D_aug = gt_aug.shape[2]

    # Column names
    trad_cols = ['ES', 'CRPS', 'QL', 'VS']
    mmd_cols = ['RbfMMD', 'SigMMD']
    csig_cols = [f"CSig({dn})" for dn in detectors]
    all_cols = trad_cols + mmd_cols + csig_cols

    # ── Table 1: Scores ──
    print(f"\n{'=' * 180}")
    print(f"{f'DEPENDENCY SENSITIVITY ({process}): Standard + SigMMD + Censored SigMMD':^180}")
    print(f"{'=' * 180}")

    header = f"{'Model':<22}"
    for c in all_cols:
        header += f" | {c:>14}"
    print(header)
    print("-" * 180)

    results = {}
    for fname, X_aug in augmented.items():
        preds_raw = models[fname].to(device)
        es, crps, ql, vs = compute_traditional_metrics(preds_raw, gt_raw)

        gt_flat = gt_aug.unsqueeze(1).repeat(1, SAMPLES, 1, 1).view(-1, L_aug, D_aug)
        preds_flat = X_aug.view(-1, L_aug, D_aug)
        r_mmd = compute_rbf_mmd(preds_flat, gt_flat)

        s_mmd = eval_sig_mmd(X_aug, gt_aug, sigma, device)

        row_data = {
            'ES': es, 'CRPS': crps, 'QL': ql, 'VS': vs,
            'RbfMMD': r_mmd, 'SigMMD': s_mmd,
        }

        for dn, detector in detectors.items():
            mask_y = detector.get_mask(gt_aug)
            mask_x = detector.get_mask(
                X_aug.reshape(-1, L_aug, D_aug)).reshape(B_TEST, SAMPLES)
            row_data[f"CSig({dn})"] = eval_csig_mmd(
                X_aug, gt_aug, mask_x, mask_y, sigma, train_aug, device)

        results[fname] = row_data
        row = f"{fname:<22}"
        for c in all_cols:
            row += f" | {row_data[c]:>14.6f}"
        print(row)

    # ── Table 2: Relative error ──
    f1_key = list(results.keys())[0]
    base = results[f1_key]
    print()
    print(f"{'Relative error (% vs ' + f1_key + ')':^180}")
    print("-" * 180)
    for fname, scores in results.items():
        row = f"{fname:<22}"
        for c in all_cols:
            diff = (scores[c] - base[c]) / (abs(base[c]) + 1e-9) * 100
            row += f" | {diff:>+13.1f}%"
        print(row)

    # ── Discrimination ratios ──
    print()
    print(f"{'Discrimination ratio (F_i / F1)':^180}")
    print("-" * 180)
    for c in mmd_cols + csig_cols:
        f1_val = base[c]
        row = f"  {c:<18}"
        for fname in results:
            ratio = results[fname][c] / (f1_val + 1e-12)
            row += f" | {fname}: {ratio:.2f}x"
        print(row)

    # ── F2 vs F3 ordering check ──
    print()
    print(f"{'F2 vs F3 ORDERING (F2 > F3 = correlations detected)':^180}")
    print("-" * 180)
    print(f"  {'Metric':<18} {'F2 (spatial only)':>18} {'F3 (independent)':>18} {'F2 > F3?':>10}")
    print(f"  {'-'*18} {'-'*18} {'-'*18} {'-'*10}")
    f2 = results["F2. No Time Corr"]
    f3 = results["F3. Independent"]
    for c in all_cols:
        correct = "YES" if f2[c] > f3[c] else "no"
        print(f"  {c:<18} {f2[c]:>18.6f} {f3[c]:>18.6f} {correct:>10}")

    print(f"\n{'=' * 180}")
    return results


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def aggregate_dependency_results(seed_results, process_label):
    """Mean ± std across seeds.

    seed_results : list of dicts, each keyed by model name, values are
                   {metric: scalar}.
    """
    if not seed_results:
        return
    model_names = list(seed_results[0].keys())
    metrics = list(seed_results[0][model_names[0]].keys())
    n_seeds = len(seed_results)

    print(f"\n{'=' * 180}")
    print(f"{f'AGGREGATE ({process_label}) ACROSS {n_seeds} SEEDS — mean ± std':^180}")
    print(f"{'=' * 180}")

    header = f"  {'Model':<22}"
    for m in metrics:
        header += f" | {m:>18}"
    print(header)
    print("  " + "-" * (22 + 21 * len(metrics)))

    for model in model_names:
        row = f"  {model:<22}"
        for m in metrics:
            vals = [sr[model][m] for sr in seed_results]
            mu = float(np.mean(vals))
            sd = float(np.std(vals, ddof=0))
            row += f" | {mu:>8.4f} ± {sd:>7.4f}"
        print(row)

    # F2-vs-F3 ordering tally per metric.
    f2_candidates = [k for k in model_names if "No Time" in k]
    f3_candidates = [k for k in model_names if "Independent" in k]
    if f2_candidates and f3_candidates:
        f2_key, f3_key = f2_candidates[0], f3_candidates[0]
        print(f"\n  F2 > F3 tally ({process_label}):")
        print(f"  {'Metric':<18} {'correct seeds / total':>24}")
        print(f"  {'-'*18} {'-'*24}")
        for m in metrics:
            correct = sum(1 for sr in seed_results
                          if sr[f2_key][m] > sr[f3_key][m])
            print(f"  {m:<18} {f'{correct}/{n_seeds}':>24}")


def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    B_TRAIN, B_TEST, SAMPLES, L, D = 500, 200, 50, 30, 5

    gp_seed_results = []

    for seed in SEEDS:
        print(f"\n\n{'#' * 180}")
        print(f"{f'SEED {seed}':^180}")
        print(f"{'#' * 180}")
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # A fresh GP with a fresh cross-variate covariance per seed — this is
        # where the bulk of the seed-level randomness comes from.
        gen = GPGenerator(n_features=D, length=L)

        print(f"\n{'#' * 180}")
        print(f"{f'GAUSSIAN PROCESS (seed {seed})':^180}")
        print(f"{'#' * 180}")
        gp_seed_results.append(
            run_single(gen, device, B_TRAIN, B_TEST, SAMPLES, L, D))

    # ── Aggregate across seeds ──
    print(f"\n\n{'#' * 180}")
    print(f"{f'AGGREGATE RESULTS ACROSS {len(SEEDS)} SEEDS':^180}")
    print(f"{'#' * 180}")
    aggregate_dependency_results(gp_seed_results, "GP")

    print(f"\n{'=' * 180}")
    print("Done.")


if __name__ == "__main__":
    run_experiment()
