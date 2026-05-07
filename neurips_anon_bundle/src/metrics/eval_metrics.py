import torch
import numpy as np
import collections as co
import os
import sys
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
try:
    from torchmetrics import Metric
except Exception:
    # torchmetrics pulls in torchvision and can fail on torch/torchvision
    # ABI mismatches. Only the (legacy) SigMMD / CensoredSigMMD classes
    # below use Metric — the ESK-based metrics used by trading_experiment
    # inherit from torch.nn.Module and do not need torchmetrics.
    Metric = torch.nn.Module
import sigkernel

# Add ESK library to path
_esk_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../ESK"))
if _esk_root not in sys.path:
    sys.path.insert(0, _esk_root)
from sigkernel_mmd import RBFKernel as ESKRBFKernel, SigKernel as ESKSigKernel
from sigkernel_mmd import SigMMD as _ESKSigMMDLibBase


class _ESKSigMMDLib(_ESKSigMMDLibBase):
    """Numerically robust subclass of the upstream ESK SigMMD.

    Replaces ``sqrt(diag_X + 1e-8)`` and ``sqrt(diag_X * K_YY + 1e-8)``
    with ``sqrt(clamp(.., min=1e-8))``. The upstream additive shift only
    catches small-positive roundoff; it produces NaN whenever the PDE
    output goes macroscopically negative (observed for some Moirai paths
    at long horizons on FX). The clamp-floor is mathematically a no-op
    when the radicand is positive (verified to within float-rounding for
    every well-behaved case in the benchmark) and turns the negative-
    diagonal pathology into a finite, sensibly-bounded value instead of
    NaN.
    """
    def _update_chunk(self, preds, target):
        B, S, L, D = preds.shape
        target_batched = target.unsqueeze(1)

        K_XX = self.kernel.compute_diag_Gram(preds, preds)
        K_YY = self.kernel.compute_diag_Gram(
            target_batched, target_batched).squeeze(-1).squeeze(-1)
        K_XY = self.kernel.compute_diag_Gram(
            preds, target_batched).squeeze(-1)

        # ── Saturation guard ──────────────────────────────────────
        # When forecast paths exit the static kernel's effective domain
        # (e.g. ‖x−y‖²/σ ≫ 1 everywhere), the static Gram is uniformly
        # ~0, the PDE driver M_inc → 0, and the recurrence collapses to
        # the boundary attractor K(x, y) ≡ 1 for every path pair. The
        # metric value would then be the constant ≈1.01 boundary value
        # regardless of forecast content — improper. We refuse to score
        # in that regime: emit NaN and a printed message. Tolerance is
        # tight (1e-7) so it fires only on the *exact* attractor, not on
        # near-misses (≈1.005) which can still reflect real differences.
        if _is_kernel_saturated(K_XX, K_YY, K_XY):
            print(
                "[SigMMD] saturation detected — all kernel evaluations "
                "collapsed to the PDE boundary attractor (K = 1). Path "
                "pairs are kernel-orthogonal at this σ; metric is "
                "undefined for this input. Returning NaN.",
                flush=True)
            nan_val = torch.tensor(float('nan'), device=preds.device,
                                   dtype=preds.dtype)
            if self.total_mmd.device != nan_val.device:
                self.total_mmd = self.total_mmd.to(nan_val.device)
                self.total_count = self.total_count.to(nan_val.device)
            self.total_mmd += nan_val
            self.total_count += B
            return

        if self.normalize:
            diag_X = torch.diagonal(K_XX, dim1=-2, dim2=-1)
            sqrt_diag_X = torch.sqrt(torch.clamp(diag_X, min=1e-8))
            norm_XX = sqrt_diag_X.unsqueeze(-1) * sqrt_diag_X.unsqueeze(-2)
            K_XX_norm = K_XX / (norm_XX + 1e-8)
            norm_XY = torch.sqrt(
                torch.clamp(diag_X * K_YY.unsqueeze(-1), min=1e-8))
            K_XY_norm = K_XY / (norm_XY + 1e-8)
            term_YY = torch.ones(B, device=preds.device, dtype=preds.dtype)
        else:
            K_XX_norm = K_XX
            K_XY_norm = K_XY
            term_YY = K_YY

        term_XX = K_XX_norm.sum(dim=(-2, -1)) / (S * S)
        term_XY = K_XY_norm.mean(dim=-1)
        mmd_sq = torch.relu(term_XX - 2 * term_XY + term_YY)

        if self.total_mmd.device != mmd_sq.device:
            self.total_mmd = self.total_mmd.to(mmd_sq.device)
            self.total_count = self.total_count.to(mmd_sq.device)
        self.total_mmd += mmd_sq.sum()
        self.total_count += B


# ═══════════════════════════════════════════════════════════════════════
# Static kernels (drop-in replacements for ESKRBFKernel)
#
# All implement the `batch_kernel(X, Y)` and `batch_kernel_diag(X, Y)`
# API consumed by ESK's SigKernel PDE solver. Heavier-tailed kernels
# (Matern, Laplacian) are useful when forecast paths have extreme values
# that push the RBF kernel `exp(-‖x-y‖²/σ)` to underflow and trigger the
# saturation regime. Implementations originally lived in
# `src/experiments/ablation_static_kernel.py`; moved here so they can
# be selected as `static_kernel="…"` on `ESKSigMMD` / `CensoredESKSigMMD`.
# ═══════════════════════════════════════════════════════════════════════

def _sq_dists_cross(X, Y):
    """(BxNx, Lx, D) × (ByNy, Ly, D) → (BxNx, ByNy, Lx, Ly)"""
    Xsq = (X ** 2).sum(-1).unsqueeze(1).unsqueeze(3)
    Ysq = (Y ** 2).sum(-1).unsqueeze(0).unsqueeze(2)
    XY  = torch.einsum('bld,cde->bcle', X, Y.transpose(1, 2))
    return torch.clamp(Xsq + Ysq - 2 * XY, min=0.0)


def _sq_dists_diag(X, Y):
    """(B, Nx, Lx, D) × (B, Ny, Ly, D) → (B, Nx, Ny, Lx, Ly)"""
    Xsq = (X ** 2).sum(-1)
    Ysq = (Y ** 2).sum(-1)
    XY  = torch.einsum('bild,bjkd->bijlk', X, Y)
    return torch.clamp(
        Xsq[:, :, None, :, None] + Ysq[:, None, :, None, :] - 2 * XY,
        min=0.0,
    )


class LaplacianKernel(torch.nn.Module):
    r"""Laplacian kernel  k(x,y) = exp(−‖x−y‖ / σ)."""

    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma

    def batch_kernel(self, X, Y):
        Bx, Nx, Lx, D = X.shape
        By, Ny, Ly, _ = Y.shape
        dist = torch.sqrt(_sq_dists_cross(
            X.reshape(Bx * Nx, Lx, D), Y.reshape(By * Ny, Ly, D)) + 1e-12)
        return torch.exp(-dist / self.sigma).reshape(Bx, Nx, By, Ny, Lx, Ly)

    def batch_kernel_diag(self, X, Y):
        dist = torch.sqrt(_sq_dists_diag(X, Y) + 1e-12)
        return torch.exp(-dist / self.sigma)


class RationalQuadraticKernel(torch.nn.Module):
    r"""Rational Quadratic kernel  k(x,y) = (1 + ‖x−y‖² / (2 α σ))^(−α).

    Parameters
    ----------
    sigma : float   Lengthscale parameter (same units as RBF).
    alpha : float   Shape parameter; as α → ∞ this converges to RBF.
    """

    def __init__(self, sigma: float = 1.0, alpha: float = 1.0):
        super().__init__()
        self.sigma = sigma
        self.alpha = alpha

    def _eval(self, sq_dist):
        return (1.0 + sq_dist / (2.0 * self.alpha * self.sigma)).pow(-self.alpha)

    def batch_kernel(self, X, Y):
        Bx, Nx, Lx, D = X.shape
        By, Ny, Ly, _ = Y.shape
        sq = _sq_dists_cross(
            X.reshape(Bx * Nx, Lx, D), Y.reshape(By * Ny, Ly, D))
        return self._eval(sq).reshape(Bx, Nx, By, Ny, Lx, Ly)

    def batch_kernel_diag(self, X, Y):
        return self._eval(_sq_dists_diag(X, Y))


class MaternKernel(torch.nn.Module):
    r"""Matern kernel (closed-form for nu = 0.5, 1.5, 2.5).

    The Matern kernel has a heavier tail than RBF (k(r) decays
    polynomially-times-exponentially for ν=1.5/2.5 vs purely
    exponentially for RBF), making it more robust to forecast paths
    whose normalised values are extreme relative to the calibration σ.
    """

    def __init__(self, sigma: float = 1.0, nu: float = 1.5):
        super().__init__()
        self.sigma = sigma
        assert nu in (0.5, 1.5, 2.5), "nu must be 0.5, 1.5, or 2.5"
        self.nu = nu

    def _eval(self, sq_dist):
        r = torch.sqrt(sq_dist + 1e-12) / self.sigma
        if self.nu == 0.5:
            return torch.exp(-r)
        elif self.nu == 1.5:
            s3 = np.sqrt(3.0) * r
            return (1.0 + s3) * torch.exp(-s3)
        else:  # 2.5
            s5 = np.sqrt(5.0) * r
            return (1.0 + s5 + s5 ** 2 / 3.0) * torch.exp(-s5)

    def batch_kernel(self, X, Y):
        Bx, Nx, Lx, D = X.shape
        By, Ny, Ly, _ = Y.shape
        sq = _sq_dists_cross(
            X.reshape(Bx * Nx, Lx, D), Y.reshape(By * Ny, Ly, D))
        return self._eval(sq).reshape(Bx, Nx, By, Ny, Lx, Ly)

    def batch_kernel_diag(self, X, Y):
        return self._eval(_sq_dists_diag(X, Y))


# Tolerance for declaring all kernel evaluations to have hit the boundary
# attractor (K = 1). Tighter than 1e-6 so we fire only on the *exact*
# attractor, not near-misses that may still reflect real distributional
# differences.
_SATURATION_TOL = 1e-7


def _is_kernel_saturated(K_XX, K_YY, K_XY) -> bool:
    """Return True iff every entry of all three Gram tensors is within
    `_SATURATION_TOL` of 1.0 — the PDE's boundary-attractor regime
    where the metric becomes a constant independent of input."""
    return bool(
        ((K_XX - 1.0).abs() < _SATURATION_TOL).all().item() and
        ((K_YY - 1.0).abs() < _SATURATION_TOL).all().item() and
        ((K_XY - 1.0).abs() < _SATURATION_TOL).all().item()
    )


_KERNEL_REGISTRY = {
    "rbf":       lambda sigma: ESKRBFKernel(sigma=sigma),
    "laplacian": lambda sigma: LaplacianKernel(sigma=sigma),
    "matern":    lambda sigma: MaternKernel(sigma=sigma, nu=1.5),
}


def _resolve_static_kernel(static_kernel, sigma):
    """Map ``static_kernel`` (string or kernel module) to a kernel
    instance. String options: 'laplacian' (default — exp(-r/σ); cusp
    at zero preserves dependency-scale sensitivity in the bulk while
    the heavier exponential tail resists the RBF saturation seen on
    extreme-path forecasts), 'matern' (Matern ν=1.5; smoother bulk
    response), 'rbf'. Anything else with a ``batch_kernel`` /
    ``batch_kernel_diag`` API is passed through unchanged."""
    if static_kernel is None or static_kernel == "laplacian":
        return LaplacianKernel(sigma=sigma)
    if isinstance(static_kernel, str):
        key = static_kernel.lower()
        if key not in _KERNEL_REGISTRY:
            raise ValueError(
                f"Unknown static_kernel='{static_kernel}'. "
                f"Options: {sorted(_KERNEL_REGISTRY.keys())}, "
                f"or pass a module exposing batch_kernel / "
                f"batch_kernel_diag.")
        return _KERNEL_REGISTRY[key](sigma)
    if hasattr(static_kernel, "batch_kernel") and \
       hasattr(static_kernel, "batch_kernel_diag"):
        return static_kernel
    raise TypeError(
        f"static_kernel must be a string ({sorted(_KERNEL_REGISTRY.keys())}) "
        f"or a module with batch_kernel/batch_kernel_diag methods, "
        f"got {type(static_kernel).__name__}.")


# --- 1. Path Normalization & Augmentation ---

class PathReVIN(torch.nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer('mean', torch.zeros(1, 1, dim))
        self.register_buffer('std', torch.ones(1, 1, dim))

    def fit(self, x):
        self.mean = x.mean(dim=(0, 1), keepdim=True)
        std = x.std(dim=(0, 1), keepdim=True)
        
        # === FIX: Robustness for Constant Columns ===
        # If std is close to 0 (constant column), set scale to 1.0 to prevent explosion
        # Otherwise, use the actual std
        self.std = torch.where(std < 1e-1, torch.ones_like(std), std)

    def forward(self, x):
        return (x - self.mean) / self.std

class AugmentPipeline(torch.nn.Module):
    def __init__(self, append_time=True):
        super().__init__()
        self.append_time = append_time

    def forward(self, x):
        B, L, C = x.shape
        device = x.device
        if self.append_time:
            time_channel = torch.linspace(0, 1, L, device=device).reshape(1, L, 1).expand(B, L, 1)
            x = torch.cat([x, time_channel], dim=2)
        curr_C = x.shape[-1]
        zero_start = torch.zeros(B, 1, curr_C, device=device) 
        zero_end = torch.zeros(B, 1, curr_C, device=device)
        zero_end[:, :, -1] = x[:, -1:, -1] 
        return torch.cat([zero_start, x, zero_end], dim=1)

# --- 2. Parameter Estimation ---

class SigmaEstimator:
    @staticmethod
    def fit(train_data_norm, max_samples=250):
        N = train_data_norm.shape[0]
        x_flat = train_data_norm.reshape(N, -1)
        idx = torch.randperm(N)[:max_samples]
        subset = x_flat[idx]
        dists = torch.cdist(subset, subset, p=2)
        valid_dists = dists[dists > 1e-6]
        if len(valid_dists) > 0:
            sigma = torch.median(valid_dists)
        else:
            sigma = torch.tensor(1.0)
        return sigma.item()

# --- 3. Feature Extraction ---

_Pair = co.namedtuple('Pair', ('start', 'end'))
class DyadicWindow:
    def __init__(self, depth): self.depth = depth
    def __call__(self, length): return self._recursive_call(float(length))
    def _recursive_call(self, length, _offset=0.0, _depth=0, _out=None):
        if _out is None: _out = [] 
        start, end = int(_offset), int(_offset + length)
        if end - start >= 2: _out.append(_Pair(start, end))
        if _depth < self.depth:
            half = length / 2
            self._recursive_call(half, _offset, _depth + 1, _out)
            self._recursive_call(half, _offset + half, _depth + 1, _out)
        return _out

class SigFeatureExtractor(torch.nn.Module):
    def __init__(self, sig_depth=3, window_depth=2):
        super().__init__()
        self.sig_depth = sig_depth
        self.window_gen = DyadicWindow(window_depth)

    def forward(self, x_aug):
        import signatory
        all_sigs = []
        for win in self.window_gen(x_aug.shape[1]):
            path_slice = x_aug[:, win.start : win.end, :]
            sig = signatory.signature(path_slice, self.sig_depth)
            all_sigs.append(sig)
        return torch.cat(all_sigs, dim=1)

# --- 4. Metric Computation (Fast / Per-Instance) ---

class BaseSigKernelMetric(Metric):
    def __init__(self, sigma=1.0):
        super().__init__()
        self.static = sigkernel.RBFKernel(sigma=sigma)
        self.model = sigkernel.SigKernel(self.static, dyadic_order=1)

    def _compute_gram_matrix(self, X, Y):
        # 1. Compute Raw Gram Matrix
        K_xy = self.model.compute_Gram(X, Y, sym=(X is Y))
        # 2. Compute Diagonals for Normalization
        diag_X = self.model.compute_kernel(X, X)
        diag_Y = self.model.compute_kernel(Y, Y)
        # 3. Normalize — clamp the radicand at +1e-8 instead of adding
        # 1e-8 outside sqrt, since PDE roundoff can make the outer product
        # negative on extreme paths and `sqrt(negative) = NaN`.
        denom = torch.sqrt(torch.clamp(torch.outer(diag_X, diag_Y), min=1e-8))
        return K_xy / denom

class SigMMD(BaseSigKernelMetric):
    def __init__(self, sigma=1.0):
        super().__init__(sigma)
        self.add_state("px", default=[], dist_reduce_fx="cat")
        self.add_state("py", default=[], dist_reduce_fx="cat")

    def update(self, x, y):
        self.px.append(x); self.py.append(y)

    def compute(self):
        X = torch.cat(self.px)
        Y = torch.cat(self.py)
        
        K_xx = self._compute_gram_matrix(X, X)
        K_yy = self._compute_gram_matrix(Y, Y)
        K_xy = self._compute_gram_matrix(X, Y)
        
        return K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()

class CensoredSigMMD(BaseSigKernelMetric):
    def __init__(self, sigma=1.0):
        super().__init__(sigma)
        self.add_state("px", default=[], dist_reduce_fx="cat")
        self.add_state("py", default=[], dist_reduce_fx="cat")
        self.add_state("mask_x", default=[], dist_reduce_fx="cat")
        self.add_state("mask_y", default=[], dist_reduce_fx="cat")

    def update(self, x, y, mask_x, mask_y):
        if x.shape[0] != mask_x.shape[0]:
             K_samples = x.shape[0] // mask_x.shape[0]
             mask_x = torch.repeat_interleave(mask_x, K_samples)
        if y.shape[0] != mask_y.shape[0]:
             K_samples = y.shape[0] // mask_y.shape[0]
             mask_y = torch.repeat_interleave(mask_y, K_samples)
        self.px.append(x); self.py.append(y)
        self.mask_x.append(mask_x); self.mask_y.append(mask_y)

    def compute(self):
        X = torch.cat(self.px)
        Y = torch.cat(self.py)
        Wx = torch.cat(self.mask_x).squeeze()
        Wy = torch.cat(self.mask_y).squeeze()

        K_xx = self._compute_gram_matrix(X, X)
        K_yy = self._compute_gram_matrix(Y, Y)
        K_xy = self._compute_gram_matrix(X, Y)

        # Zero Path for Pivot
        Z = torch.zeros((1, X.shape[1], X.shape[2]), device=X.device)
        K_x0 = self._compute_gram_matrix(X, Z).squeeze()
        K_y0 = self._compute_gram_matrix(Y, Z).squeeze()

        def weighted_score(K_mat, K_a0, K_b0, w_a, w_b):
            W_mat = torch.outer(w_a, w_b)
            term1 = (W_mat * K_mat).mean()
            term2 = (w_a * K_a0).mean() * (1 - w_b).mean()
            term3 = (1 - w_a).mean() * (w_b * K_b0).mean()
            term4 = (1 - w_a).mean() * (1 - w_b).mean()
            return term1 + term2 + term3 + term4

        score_xx = weighted_score(K_xx, K_x0, K_x0, Wx, Wx)
        score_yy = weighted_score(K_yy, K_y0, K_y0, Wy, Wy)
        score_xy = weighted_score(K_xy, K_x0, K_y0, Wx, Wy)

        return score_xx + score_yy - 2 * score_xy

# --- 4b. ESK-based Metric Computation (Batched) ---
#
# These take (B, S, L, D) preds and (B, L, D) target — no sample-repeating
# of Y needed.  ESK's compute_diag_Gram computes block-diagonal grams across
# the batch in one kernel call, eliminating the per-instance Python loop.

class ESKSigMMD(torch.nn.Module):
    """Batched SigMMD using the ESK (sigkernel_mmd) library.

    Wraps ESK's SigMMD which uses compute_diag_Gram for efficient
    block-diagonal kernel computation over (B, S, L, D) inputs.

    Parameters
    ----------
    sigma : float
        Kernel length-scale parameter.
    dyadic_order : int
        PDE dyadic refinement depth.
    static_kernel : str or module
        Static kernel choice. ``"laplacian"`` (default — exp(-r/σ);
        chosen as project-wide default because its cusp at zero
        preserves dependency-scale sensitivity in the bulk while the
        heavier exponential tail resists the boundary-attractor
        saturation that plain RBF hits on long-horizon / extreme-path
        forecasts), ``"matern"`` (Matern ν=1.5; smoother bulk
        response), or ``"rbf"`` (legacy). May also be an arbitrary
        kernel module exposing the ``batch_kernel`` /
        ``batch_kernel_diag`` API consumed by ``ESKSigKernel``.
    """
    def __init__(self, sigma=1.0, dyadic_order=1, max_batch=4,
                 static_kernel="laplacian"):
        super().__init__()
        static = _resolve_static_kernel(static_kernel, sigma)
        kernel = ESKSigKernel(static, dyadic_order=dyadic_order)
        self.metric = _ESKSigMMDLib(kernel, normalize=True, max_batch=max_batch)

    def update(self, preds, target):
        """preds: (B, S, L, D), target: (B, L, D)"""
        self.metric.update(preds, target)

    def compute(self):
        return self.metric.compute()

    def reset(self):
        self.metric.reset()


class CensoredESKSigMMD(torch.nn.Module):
    """Batched censored SigMMD using the ESK library.

    Uses compute_diag_Gram for block-diagonal kernel computation and
    vectorises the weighted pivot-path censoring across the batch.

    Parameters
    ----------
    sigma : float
        Kernel length-scale parameter.
    dyadic_order : int
        PDE dyadic refinement depth.
    static_kernel : str or module
        Static-kernel choice; see :class:`ESKSigMMD`. Default
        ``"laplacian"`` (project-wide default).
    """
    def __init__(self, sigma=1.0, dyadic_order=1, max_batch=2,
                 static_kernel="laplacian"):
        super().__init__()
        static = _resolve_static_kernel(static_kernel, sigma)
        self.kernel = ESKSigKernel(static, dyadic_order=dyadic_order)
        self.max_batch = max_batch
        self.pivot = None  # (1, L, D) mean training path; None = zero path
        self.register_buffer('total_mmd', torch.tensor(0.0))
        self.register_buffer('total_count', torch.tensor(0))

    def set_pivot(self, train_data_aug):
        """Set the pivot path to the mean of training data.

        train_data_aug : (N, L, D) augmented training paths.
        """
        self.pivot = train_data_aug.mean(dim=0, keepdim=True).detach()  # (1, L, D)

    def _update_chunk(self, preds, target, mask_x, mask_y):
        """Process a single micro-batch."""
        B, S, L, D = preds.shape
        target_b = target.unsqueeze(1)

        K_XX = self.kernel.compute_diag_Gram(preds, preds)
        K_YY = self.kernel.compute_diag_Gram(
            target_b, target_b).squeeze(-1).squeeze(-1)
        K_XY = self.kernel.compute_diag_Gram(
            preds, target_b).squeeze(-1)

        # Saturation guard — see ESKSigMMD/_ESKSigMMDLib._update_chunk for
        # rationale. If every kernel evaluation has collapsed to the PDE
        # boundary attractor (K = 1) then the metric becomes a constant
        # independent of forecast content and we refuse to score.
        if _is_kernel_saturated(K_XX, K_YY, K_XY):
            print(
                "[CSigMMD] saturation detected — all kernel evaluations "
                "collapsed to the PDE boundary attractor (K = 1). Path "
                "pairs are kernel-orthogonal at this σ; metric is "
                "undefined for this input. Returning NaN.",
                flush=True)
            nan_val = torch.tensor(float('nan'), device=preds.device,
                                   dtype=preds.dtype)
            if self.total_mmd.device != nan_val.device:
                self.total_mmd = self.total_mmd.to(nan_val.device)
                self.total_count = self.total_count.to(nan_val.device)
            self.total_mmd += nan_val
            self.total_count += B
            return

        # Pivot path: mean training path if set, otherwise mean of current batch
        if self.pivot is not None:
            pv = self.pivot.to(preds.device)              # (1, L, D)
        else:
            pv = target.mean(dim=0, keepdim=True)         # (1, L, D)
        Z = pv.unsqueeze(0).expand(B, 1, L, D)            # (B, 1, L, D)
        K_X0 = self.kernel.compute_diag_Gram(preds, Z).squeeze(-1)
        K_Y0 = self.kernel.compute_diag_Gram(
            target_b, Z).squeeze(-1).squeeze(-1)
        K_00 = self.kernel.compute_Gram(pv, pv).squeeze()

        diag_X  = torch.diagonal(K_XX, dim1=-2, dim2=-1)
        # Numerical-robustness fix: clamp the radicand at +1e-8 instead of
        # additively shifting. No-op when diag_X / K_YY / K_00 are positive
        # (the typical case); prevents NaN when the float64 PDE solver
        # produces macroscopically negative diagonal entries on extreme
        # input paths.
        sqrt_dX = torch.sqrt(torch.clamp(diag_X, min=1e-8))
        sqrt_dY = torch.sqrt(torch.clamp(K_YY,   min=1e-8))
        sqrt_d0 = torch.sqrt(torch.clamp(K_00,   min=1e-8))

        K_XX = K_XX / (sqrt_dX.unsqueeze(-1) * sqrt_dX.unsqueeze(-2) + 1e-8)
        K_XY = K_XY / (sqrt_dX * sqrt_dY.unsqueeze(-1) + 1e-8)
        K_X0 = K_X0 / (sqrt_dX * sqrt_d0 + 1e-8)
        K_Y0 = K_Y0 / (sqrt_dY * sqrt_d0 + 1e-8)

        wx = mask_x
        wy = mask_y

        W_XX = wx.unsqueeze(-1) * wx.unsqueeze(-2)
        wx_K_X0 = (wx * K_X0).mean(dim=-1)
        mean_1mx = (1 - wx).mean(dim=-1)

        score_xx = ((W_XX * K_XX).mean(dim=(-2, -1))
                    + 2 * wx_K_X0 * mean_1mx
                    + mean_1mx * mean_1mx)
        score_yy = (wy * wy
                    + 2 * wy * (1 - wy) * K_Y0
                    + (1 - wy) * (1 - wy))
        score_xy = (wy * (wx * K_XY).mean(dim=-1)
                    + (1 - wy) * wx_K_X0
                    + mean_1mx * wy * K_Y0
                    + mean_1mx * (1 - wy))

        mmd_sq = torch.relu(score_xx + score_yy - 2 * score_xy)

        if self.total_mmd.device != mmd_sq.device:
            self.total_mmd  = self.total_mmd.to(mmd_sq.device)
            self.total_count = self.total_count.to(mmd_sq.device)
        self.total_mmd  += mmd_sq.sum()
        self.total_count += B

    def update(self, preds, target, mask_x, mask_y):
        """
        preds  : (B, S, L, D)  prediction samples
        target : (B, L, D)     ground-truth paths
        mask_x : (B, S)        per-sample censoring weight for predictions
        mask_y : (B,)          censoring weight for ground truth

        Automatically micro-batches to stay within GPU memory.
        """
        B, S, L, D = preds.shape

        if L > 150 and self.kernel.dyadic_order > 0:
            self.kernel.dyadic_order = 0

        mb = self.max_batch
        for i in range(0, B, mb):
            j = min(i + mb, B)
            self._update_chunk(
                preds[i:j], target[i:j], mask_x[i:j], mask_y[i:j])

    def compute(self):
        return self.total_mmd / self.total_count

    def reset(self):
        self.total_mmd.zero_()
        self.total_count.zero_()

# --- 5. Censoring Mechanisms ---

class MahalanobisTailDetector:
    """Tail detection via Mahalanobis distance on truncated signature features.

    Uses MinCovDet on signature features extracted from augmented paths.
    """
    def __init__(self, quantile=0.95, feature_extractor=None, steepness=15.0):
        self.quantile = quantile
        self.feature_extractor = feature_extractor
        self.steepness = steepness
        self.mcd = None
        self.pca = None
        self.threshold = None

    def fit(self, train_data_aug):
        self.feature_extractor.eval()
        with torch.no_grad():
            feats = self.feature_extractor(train_data_aug).cpu().numpy()

        if feats.shape[1] > 32:
            self.pca = PCA(n_components=32, whiten=False)
            feats = self.pca.fit_transform(feats)

        self.mcd = MinCovDet(support_fraction=0.9, random_state=42).fit(feats)
        dists = self.mcd.mahalanobis(feats)
        self.threshold = np.quantile(dists, self.quantile)

    def get_mask(self, data_aug):
        if self.mcd is None:
            raise RuntimeError("Detector must be fit() before get_mask().")
        with torch.no_grad():
            feats = self.feature_extractor(data_aug).cpu().numpy()
        if self.pca:
            feats = self.pca.transform(feats)
        dists = torch.tensor(self.mcd.mahalanobis(feats), device=data_aug.device)
        return torch.sigmoid(self.steepness * (dists - self.threshold))


class KernelTailDetector:
    """Tail detection via kernel mean embedding distance.

    Measures how far each test path is from the *distribution* of training
    paths in the signature kernel RKHS, not just the geometric mean:

        d(x) = k_norm(x,x) - (2/M) Σ_i k_norm(x, z_i) + C_train

    where z_i are representative training paths (subsampled for speed),
    k_norm is the normalised signature kernel, and C_train is the constant
    mean self-similarity of the training set (precomputed at fit time).

    This captures distributional structure (correlations, temporal dynamics)
    that a simple distance-from-mean approach misses.

    Parameters
    ----------
    quantile : float
        Quantile of training distances used to set the threshold.
    sigma : float or None
        Static-kernel length-scale. If None, estimated via the median
        heuristic on training paths.
    dyadic_order : int
        PDE dyadic refinement depth.
    steepness : float
        Sigmoid steepness for soft thresholding.
    max_batch : int
        Micro-batch size for kernel computation (memory control).
    n_representatives : int
        Number of training paths to keep as representative set.
    static_kernel : str or module
        Static-kernel choice. Default ``"laplacian"`` (exp(-r/σ)) —
        matches the project-wide default for ``ESKSigMMD`` and avoids
        the boundary-attractor saturation that plain RBF hits on
        extreme-path forecasts. Other options: ``"matern"``, ``"rbf"``,
        or any module exposing ``batch_kernel`` / ``batch_kernel_diag``.
    """

    def __init__(self, quantile=0.95, sigma=None, dyadic_order=1,
                 steepness=15.0, max_batch=32, n_representatives=200,
                 static_kernel="laplacian"):
        self.quantile = quantile
        self.sigma_init = sigma
        self.dyadic_order = dyadic_order
        self.steepness = steepness
        self.max_batch = max_batch
        self.n_representatives = n_representatives
        self.static_kernel_choice = static_kernel
        self.reps = None           # (M, L, D) representative training paths
        self.C_train = None        # scalar: (1/M²) Σ_{i,j} k_norm(z_i, z_j)
        self.threshold = None
        self.kernel = None

    def _build_kernel(self, sigma):
        static = _resolve_static_kernel(self.static_kernel_choice, sigma)
        self.kernel = ESKSigKernel(static, dyadic_order=self.dyadic_order)

    @torch.no_grad()
    def _compute_gram_normalised(self, X, Y):
        """Compute normalised Gram matrix between 3D path tensors.

        X: (A, L, D), Y: (B, L, D) → (A, B) normalised kernel values.
        Micro-batched to avoid OOM on large A, B.

        Uses compute_diag_Gram (batch_kernel_diag) — the same kernel code
        path as CensoredESKSigMMD, which is numerically proven stable for
        this workload. Full float64 to match the PDE solver's internal dtype.
        """
        A, B = X.shape[0], Y.shape[0]
        device = X.device
        mb = self.max_batch

        # Self-kernel diagonals k(x_i, x_i). compute_diag_Gram auto-handles
        # 3D inputs by unsqueezing Nx=Ny=1 internally, returning (mb,) — the
        # per-path self-kernels.
        diag_X = torch.zeros(A, device=device, dtype=torch.float64)
        for i in range(0, A, mb):
            j = min(i + mb, A)
            chunk = X[i:j]
            diag_X[i:j] = self.kernel.compute_diag_Gram(chunk, chunk)

        diag_Y = torch.zeros(B, device=device, dtype=torch.float64)
        for i in range(0, B, mb):
            j = min(i + mb, B)
            chunk = Y[i:j]
            diag_Y[i:j] = self.kernel.compute_diag_Gram(chunk, chunk)

        # Cross-Gram A×B. batch_kernel_diag supports Nx != Ny; put A/B into
        # the Nx/Ny dims by unsqueezing a singleton B dim so the call goes
        # through the proven-stable block-diagonal path.
        K = torch.zeros(A, B, device=device, dtype=torch.float64)
        for i in range(0, A, mb):
            ie = min(i + mb, A)
            X_sub = X[i:ie].unsqueeze(0)   # (1, ie-i, L, D)
            for j2 in range(0, B, mb):
                je = min(j2 + mb, B)
                Y_sub = Y[j2:je].unsqueeze(0)  # (1, je-j2, L, D)
                K[i:ie, j2:je] = self.kernel.compute_diag_Gram(
                    X_sub, Y_sub).squeeze(0)

        # Normalise (same pattern as CensoredESKSigMMD._update_chunk).
        # Clamp the radicand at +1e-8 instead of additively shifting — the
        # PDE solver can produce macroscopically negative diag(K) entries
        # on extreme input paths, which would otherwise NaN the sqrt and
        # poison every detector mask downstream.
        denom = torch.sqrt(torch.clamp(torch.outer(diag_X, diag_Y), min=1e-8))
        return K / (denom + 1e-8)

    @torch.no_grad()
    def _embedding_dist(self, paths):
        """Kernel mean embedding distance from paths to the training distribution.

        d(x) = 1 - (2/M) Σ_i k_norm(x, z_i) + C_train

        (k_norm(x,x) = 1 after normalisation, so the first term is always 1.)

        Returns (N,) tensor.
        """
        N = paths.shape[0]
        M = self.reps.shape[0]
        device = paths.device

        dists = []
        for i in range(0, N, self.max_batch):
            j = min(i + self.max_batch, N)
            chunk = paths[i:j]  # (mb, L, D)

            # k_norm(chunk, reps): (mb, M)
            # Compute in sub-batches over reps too if M is large. Keep
            # float64 to match _compute_gram_normalised's output precision.
            k_xz_sum = torch.zeros(
                chunk.shape[0], device=device, dtype=torch.float64)
            rep_bs = max(self.max_batch, 64)
            for ri in range(0, M, rep_bs):
                rj = min(ri + rep_bs, M)
                K_block = self._compute_gram_normalised(chunk, self.reps[ri:rj])
                k_xz_sum += K_block.sum(dim=1)

            mean_k_xz = k_xz_sum / M
            d = torch.relu(1.0 - 2 * mean_k_xz + self.C_train)
            dists.append(d)

        return torch.cat(dists)

    def fit(self, train_data_aug):
        """Calibrate on training paths (N, L, D)."""
        device = train_data_aug.device
        N, L, D = train_data_aug.shape

        if self.sigma_init is None:
            sigma = SigmaEstimator.fit(train_data_aug)
        else:
            sigma = self.sigma_init
        self._build_kernel(sigma)

        if L > 150 and self.kernel.dyadic_order > 0:
            self.kernel.dyadic_order = 0

        # Store mean training path for use as pivot in censored MMD
        self.mean_path = train_data_aug.mean(dim=0, keepdim=True)  # (1, L, D)

        # Subsample representative set
        M = min(self.n_representatives, N)
        idx = torch.randperm(N)[:M]
        self.reps = train_data_aug[idx].clone()  # (M, L, D)

        # Precompute C_train = (1/M²) Σ_{i,j} k_norm(z_i, z_j)
        K_zz = self._compute_gram_normalised(self.reps, self.reps)  # (M, M)
        self.C_train = K_zz.mean().item()

        # Compute embedding distances for all training data
        train_dists = self._embedding_dist(train_data_aug)
        self.threshold = torch.quantile(train_dists, self.quantile).item()

    def get_mask(self, data_aug):
        """Compute soft censoring mask (N,) for test paths."""
        if self.kernel is None:
            raise RuntimeError("Detector must be fit() before get_mask().")
        dists = self._embedding_dist(data_aug)
        return torch.sigmoid(self.steepness * (dists - self.threshold))


class SystemicLoadDetector:
    """Aggregate capacity exceedance censoring for systemic risk.

    Weights paths by whether the peak aggregate load across all variates
    approaches or exceeds a critical systemic threshold.  Designed for
    safety-critical systems (power grids, traffic networks) where the
    operational bottleneck is the cumulative sum of interacting components
    and under-prediction carries heavily asymmetric costs.

    For a path x with D variates, the aggregate systemic load at time t is:

        S_t(x) = sum_d x_{t,d}

    The weight is:

        w(x) = sigmoid(steepness * (max_t S_t(x) - tau))

    Under the censored MMD framework with a safe-state pivot, this naturally
    induces asymmetric penalties: if the ground truth is an extreme systemic
    event (w -> 1) but the forecast under-predicts aggregate load (w -> 0),
    the forecast mass collapses to the pivot and the signature kernel MMD
    severely penalises the distance between the extreme GT path and the
    safe pivot — including missed cross-variate positive correlations that
    would overwhelm aggregate capacity.

    Parameters
    ----------
    threshold : float or None
        Critical systemic capacity tau.  If None, calibrated from training
        data via quantile.
    steepness : float
        Sigmoid steepness (higher = sharper transition around tau).
    quantile : float or None
        If provided and threshold is None, fit() sets tau as this quantile
        of training peak aggregate loads.
    """

    def __init__(self, threshold=None, steepness=15.0, quantile=None):
        self.threshold = threshold
        self.steepness = steepness
        self.quantile = quantile

    def _peak_aggregate_load(self, paths):
        """Compute peak aggregate load per path. paths: (N, L, D) -> (N,)."""
        # S_t = sum_d x_{t,d}  then  max_t S_t
        return paths.sum(dim=-1).max(dim=-1).values  # (N,)

    def fit(self, train_data_aug):
        """Calibrate threshold from training data if quantile is set."""
        if self.quantile is not None and self.threshold is None:
            loads = self._peak_aggregate_load(train_data_aug)
            self.threshold = torch.quantile(loads, self.quantile).item()

    def get_mask(self, data_aug):
        """
        data_aug : (N, L, D)
        Returns  : (N,) soft mask in [0, 1]
        """
        if self.threshold is None:
            raise RuntimeError(
                "Threshold not set. Either pass threshold= to constructor "
                "or pass quantile= and call fit().")
        loads = self._peak_aggregate_load(data_aug)
        return torch.sigmoid(self.steepness * (loads - self.threshold))


class VaRExceedanceDetector:
    """Value-at-Risk censoring based on a path risk score.

    Assigns weight based on whether a path's risk score exceeds an
    empirical VaR threshold.  "Boring" everyday paths are zeroed out;
    the metric only evaluates extreme-event accuracy.

        w(x) = sigmoid(steepness * (R(x) - VaR_alpha))

    where R(x) is a configurable risk-scoring function applied to each path.

    Parameters
    ----------
    threshold : float
        VaR threshold. Paths with risk score above this are weighted.
        Can be set manually or calibrated from training data via fit().
    steepness : float
        Sigmoid steepness.
    risk_fn : str
        Risk scoring function:
        - 'max_drawdown'  : Maximum peak-to-trough decline across the path.
        - 'peak'          : Maximum absolute value (same as utility detector).
        - 'volatility'    : Standard deviation across timesteps.
        - 'tail_ratio'    : Fraction of timesteps exceeding 2 std from mean.
    quantile : float or None
        If provided, fit() calibrates the threshold as this quantile of
        training risk scores (overriding the manual threshold).
    dim : int or None
        If specified, compute risk on this dimension only.
    """

    def __init__(self, threshold=None, steepness=15.0, risk_fn='max_drawdown',
                 quantile=None, dim=None):
        self.threshold = threshold
        self.steepness = steepness
        self.risk_fn = risk_fn
        self.quantile = quantile
        self.dim = dim

    def _compute_risk(self, paths):
        """Compute risk score per path. paths: (N, L, D) -> (N,)."""
        if self.dim is not None:
            paths = paths[:, :, self.dim:self.dim + 1]  # (N, L, 1)

        if self.risk_fn == 'max_drawdown':
            # Max peak-to-trough decline: max_t(cummax(t) - x(t))
            # Compute per-dimension, take max across dims
            cummax = paths.cummax(dim=1).values        # (N, L, D)
            drawdown = (cummax - paths).amax(dim=1)    # (N, D)
            return drawdown.amax(dim=1)                # (N,)

        elif self.risk_fn == 'peak':
            return paths.abs().amax(dim=(1, 2))        # (N,)

        elif self.risk_fn == 'volatility':
            return paths.std(dim=1).amax(dim=1)        # (N,)

        elif self.risk_fn == 'tail_ratio':
            # Fraction of timesteps exceeding 2 std from mean (per path)
            mu = paths.mean(dim=1, keepdim=True)       # (N, 1, D)
            sd = paths.std(dim=1, keepdim=True) + 1e-8  # (N, 1, D)
            extreme = ((paths - mu).abs() > 2 * sd).float()
            return extreme.mean(dim=(1, 2))            # (N,)

        elif self.risk_fn == 'period_return':
            # Absolute start→end move per path. Designed to focus evaluation
            # on "decisive" paths (large realised directional move), where
            # trading PnL is concentrated. Sign-blind: |end - start|.
            # Paths from AugmentPipeline are zero-padded at boundaries, so
            # index 1 and -2 are the real endpoints. If dim was not specified
            # and a time channel is present (last channel), drop it.
            if paths.shape[1] >= 3:
                start_val = paths[:, 1, :]
                end_val = paths[:, -2, :]
            else:
                start_val = paths[:, 0, :]
                end_val = paths[:, -1, :]
            diff = (end_val - start_val).abs()
            if self.dim is None and diff.shape[-1] > 1:
                diff = diff[..., :-1]                  # drop time channel
            return diff.amax(dim=-1)                    # (N,)

        else:
            raise ValueError(f"Unknown risk_fn: {self.risk_fn!r}. "
                             f"Choose 'max_drawdown', 'peak', 'volatility', "
                             f"'tail_ratio', or 'period_return'.")

    def fit(self, train_data_aug):
        """Optionally calibrate threshold from training data."""
        if self.quantile is not None:
            risk_scores = self._compute_risk(train_data_aug)
            self.threshold = torch.quantile(risk_scores, self.quantile).item()

    def get_mask(self, data_aug):
        """
        data_aug : (N, L, D)
        Returns  : (N,) soft mask in [0, 1]
        """
        if self.threshold is None:
            raise RuntimeError(
                "Threshold not set. Either pass threshold= to constructor "
                "or pass quantile= and call fit().")
        risk = self._compute_risk(data_aug)
        return torch.sigmoid(self.steepness * (risk - self.threshold))


class SharpeMVDetector:
    """Markowitz mean-variance utility censoring.

    Scores a path by the *optimal* mean-variance utility achievable on that
    path, U*(x) = (1/2γ) · μ^T Σ^{-1} μ, where
        μ_d = mean per-step simple return of channel d along the path,
        Σ   = empirical covariance of per-step returns across channels.

    This is the unconstrained-weight maximiser of
        U(w) = w^T μ − γ/2 · w^T Σ w,
    i.e. the Sharpe-squared (up to scaling) of a path. Financially:
    up-weights paths where a Markowitz-optimising allocator would see a
    large risk-adjusted opportunity (high mean with low / favourably-
    correlated variance). Censoring by this score focuses the metric on
    paths the allocator would actually have acted on.

    Parameters
    ----------
    threshold : float or None
        Critical utility threshold. If None, calibrated from training data
        at `quantile`.
    steepness : float
        Sigmoid steepness for the soft mask.
    quantile : float or None
        If set and threshold is None, fit() sets threshold = this quantile
        of training utilities.
    gamma : float
        Risk-aversion parameter in U* = (1/2γ) μ^T Σ^{-1} μ. Rescales the
        overall utility; does not change ranking under quantile calibration.
    eps : float
        Tikhonov regularisation added to the diagonal of Σ before solve.
    skip_last_channel : bool
        If True (default), drops the last channel before computing returns.
        Matches `AugmentPipeline(append_time=True)`, whose last channel is
        a time index rather than a financial asset value.
    """

    def __init__(self, threshold=None, steepness=15.0, quantile=None,
                 gamma=1.0, eps=1e-6, skip_last_channel=True):
        self.threshold = threshold
        self.steepness = steepness
        self.quantile = quantile
        self.gamma = gamma
        self.eps = eps
        self.skip_last_channel = skip_last_channel

    def _mv_utility(self, paths):
        """U*(x) = (1/2γ) μ^T Σ^{-1} μ per path.  paths: (N, L, D) → (N,)."""
        if self.skip_last_channel and paths.shape[-1] > 1:
            paths = paths[..., :-1]
        # Skip AugmentPipeline zero-padding at start/end if present.
        if paths.shape[1] >= 4:
            paths = paths[:, 1:-1, :]
        if paths.shape[1] < 2:
            return torch.zeros(paths.shape[0], device=paths.device,
                               dtype=paths.dtype)

        # Per-step simple returns: (N, L-1, D)
        denom = paths[:, :-1, :].abs() + 1e-10
        returns = (paths[:, 1:, :] - paths[:, :-1, :]) / denom
        N_, T_, D_ = returns.shape

        mu = returns.mean(dim=1)                            # (N, D)
        centered = returns - mu.unsqueeze(1)                # (N, L-1, D)
        cov = (centered.transpose(1, 2) @ centered) / max(T_ - 1, 1)   # (N, D, D)
        cov = cov + self.eps * torch.eye(D_, device=cov.device,
                                         dtype=cov.dtype).unsqueeze(0)
        try:
            inv_mu = torch.linalg.solve(cov, mu.unsqueeze(-1)).squeeze(-1)
        except Exception:
            # Fall back to pseudo-inverse on singular cov.
            inv_mu = torch.linalg.pinv(cov) @ mu.unsqueeze(-1)
            inv_mu = inv_mu.squeeze(-1)
        u = (mu * inv_mu).sum(dim=-1) / (2.0 * self.gamma)  # (N,)
        # Clamp pathological blow-ups (rare but can happen on near-singular Σ).
        u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        return u

    def fit(self, train_data_aug):
        """Calibrate threshold = quantile of training MV utilities."""
        if self.quantile is not None and self.threshold is None:
            utils = self._mv_utility(train_data_aug)
            self.threshold = torch.quantile(utils, self.quantile).item()

    def get_mask(self, data_aug):
        """data_aug : (N, L, D) → (N,) soft mask in [0, 1]."""
        if self.threshold is None:
            raise RuntimeError(
                "Threshold not set. Either pass threshold= to constructor "
                "or pass quantile= and call fit().")
        utils = self._mv_utility(data_aug)
        return torch.sigmoid(self.steepness * (utils - self.threshold))


def make_tail_detector(method="mahalanobis", quantile=0.95, steepness=15.0,
                       feature_extractor=None, sigma=None, dyadic_order=1,
                       max_batch=32, n_representatives=200, device=None,
                       threshold=None, risk_fn='max_drawdown', dim=None,
                       gamma=1.0, static_kernel="laplacian"):
    """Factory for censoring detectors.

    Parameters
    ----------
    method : str
        Tail-based (data-driven threshold from quantile):
          'mahalanobis'  — MCD on truncated signature features.
          'kernel'       — Kernel mean embedding distance (distributional).

        Risk-based:
          'var'          — Value-at-Risk: weight paths whose risk score exceeds threshold.
          'systemic'     — Aggregate capacity: weight paths whose peak sum-of-variates
                           load exceeds a systemic threshold.
          'sharpe'       — Markowitz mean-variance utility U* = (1/2γ)μ^T Σ^{-1} μ.
                           Up-weights paths with large risk-adjusted return
                           (Sharpe-squared) opportunities.

    threshold : float or None
        For 'var'/'systemic'/'sharpe': manual threshold. If None, calibrated via quantile.
    risk_fn : str
        For 'var' only: 'max_drawdown', 'peak', 'volatility', 'tail_ratio',
        or 'period_return' (|end - start|, focuses on decisive directional
        moves — PnL-aligned).
    dim : int or None
        For 'var': restrict risk computation to this dimension.
    gamma : float
        For 'sharpe' only: risk-aversion parameter.
    """
    if method == "mahalanobis":
        if feature_extractor is None:
            feature_extractor = SigFeatureExtractor(3, 2)
            if device:
                feature_extractor = feature_extractor.to(device)
        return MahalanobisTailDetector(
            quantile=quantile, feature_extractor=feature_extractor,
            steepness=steepness)
    elif method == "kernel":
        return KernelTailDetector(
            quantile=quantile, sigma=sigma, dyadic_order=dyadic_order,
            steepness=steepness, max_batch=max_batch,
            n_representatives=n_representatives,
            static_kernel=static_kernel)
    elif method == "var":
        return VaRExceedanceDetector(
            threshold=threshold, steepness=steepness, risk_fn=risk_fn,
            quantile=quantile if threshold is None else None, dim=dim)
    elif method == "systemic":
        return SystemicLoadDetector(
            threshold=threshold, steepness=steepness,
            quantile=quantile if threshold is None else None)
    elif method == "sharpe":
        return SharpeMVDetector(
            threshold=threshold, steepness=steepness,
            quantile=quantile if threshold is None else None,
            gamma=gamma)
    else:
        raise ValueError(
            f"Unknown method: {method!r}. Choose from: "
            f"'mahalanobis', 'kernel', 'var', 'systemic', 'sharpe'.")