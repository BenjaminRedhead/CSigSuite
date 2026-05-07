"""
Signature kernel — real-space Padé PDE solver with optional Triton acceleration.

Matches the sigkernel library's numerics:
  • RBF static kernel:  k(x,y) = exp(-‖x-y‖²/σ)   (NOT 1/(2σ²))
  • Padé(1,1) update:   K[i,j] = (K[i-1,j]+K[i,j-1])·A - K[i-1,j-1]·C
                         A = 1 + dt/2 + dt²/12,  C = 1 - dt²/12
  • Dyadic refinement:  tile M_inc by 2^n in each dim, divide by (2^n)²
  • Anti-diagonal sweep: 2L-1 vectorised steps (vs L² scalar steps)
"""
import torch
import torch.nn as nn
from typing import Optional

__all__ = ['RBFKernel', 'SigKernel', 'SigKernelDiagFn']

# Try to import triton; fall back gracefully
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ════════════════════════════════════════════════════════════════════════════
# Helper: tile a tensor along a dimension (replicates each element n times)
# ════════════════════════════════════════════════════════════════════════════

def _tile(A: torch.Tensor, dim: int, n: int) -> torch.Tensor:
    """Repeat each slice of *A* along *dim* exactly *n* times."""
    return A.repeat_interleave(n, dim=dim)


# ════════════════════════════════════════════════════════════════════════════
# Real-space PDE solver  (PyTorch, anti-diagonal sweep)
# ════════════════════════════════════════════════════════════════════════════

def _solve_pde(M_inc: torch.Tensor) -> torch.Tensor:
    """
    Solve the signature-kernel Goursat PDE in real space.

    Parameters
    ----------
    M_inc : (..., M, N)   increment tensor  (∂²k_static / ∂s∂t)

    Returns
    -------
    K : (..., M+1, N+1)   PDE solution grid,  K[..., -1, -1] is the kernel value.
    """
    *batch, M, N = M_inc.shape
    device = M_inc.device
    # Use float64 to avoid overflow — sig-kernel values grow factorially.
    M_inc = M_inc.to(torch.float64)

    K = torch.ones(*batch, M + 1, N + 1, device=device, dtype=torch.float64)

    for d in range(2, M + N + 1):
        i0 = max(1, d - N)
        i1 = min(M, d - 1)
        i = torch.arange(i0, i1 + 1, device=device)
        j = d - i

        dt    = M_inc[..., i - 1, j - 1]
        k10   = K[..., i - 1, j]
        k01   = K[..., i, j - 1]
        k00   = K[..., i - 1, j - 1]

        dt_sq = dt * dt
        A = 1.0 + 0.5 * dt + (1.0 / 12.0) * dt_sq
        C = 1.0 - (1.0 / 12.0) * dt_sq

        K[..., i, j] = (k10 + k01) * A - k00 * C

    return K


# ════════════════════════════════════════════════════════════════════════════
# Triton PDE kernel  (one program per batch element)
# ════════════════════════════════════════════════════════════════════════════

if HAS_TRITON:
    # Cap BLOCK to avoid register pressure / illegal memory access on large grids.
    # 256 threads × float64 is well within A100 register limits.
    _TRITON_MAX_BLOCK = 256

    @triton.jit
    def _sigkernel_diag_step(
        M_ptr, K_ptr,
        M_stride_b, M_stride_i, M_stride_j,
        K_stride_b, K_stride_i, K_stride_j,
        i0, n_cells, d,
        BLOCK: tl.constexpr,
    ):
        """Process one chunk of an anti-diagonal across all batch elements.

        Grid: (B, n_chunks) where n_chunks = ceil(n_cells / BLOCK).
        program_id(0) = batch index, program_id(1) = chunk index.
        """
        bid = tl.program_id(0)
        cid = tl.program_id(1)
        offs = cid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_cells

        # Clamp so masked-off threads never compute OOB addresses
        i = tl.where(mask, i0 + offs, 1)
        j = tl.where(mask, d - (i0 + offs), 1)
        mi = tl.where(mask, i - 1, 0)
        mj = tl.where(mask, j - 1, 0)

        dt  = tl.load(M_ptr + bid * M_stride_b + mi * M_stride_i + mj * M_stride_j, mask=mask, other=0.0)
        k10 = tl.load(K_ptr + bid * K_stride_b + (i - 1) * K_stride_i + j * K_stride_j, mask=mask, other=1.0)
        k01 = tl.load(K_ptr + bid * K_stride_b + i * K_stride_i + (j - 1) * K_stride_j, mask=mask, other=1.0)
        k00 = tl.load(K_ptr + bid * K_stride_b + (i - 1) * K_stride_i + (j - 1) * K_stride_j, mask=mask, other=1.0)

        dt_sq = dt * dt
        A = 1.0 + 0.5 * dt + (1.0 / 12.0) * dt_sq
        C = 1.0 - (1.0 / 12.0) * dt_sq
        val = (k10 + k01) * A - k00 * C

        tl.store(K_ptr + bid * K_stride_b + i * K_stride_i + j * K_stride_j, val, mask=mask)

    def _solve_pde_triton(M_inc: torch.Tensor) -> torch.Tensor:
        """GPU-accelerated PDE solve via Triton (float64, chunked anti-diagonal sweep).

        Each anti-diagonal is split into chunks of _TRITON_MAX_BLOCK threads.
        Grid is (B, n_chunks) so large anti-diagonals use multiple programs
        instead of a single oversized BLOCK that exhausts registers.
        """
        *batch, M, N = M_inc.shape
        flat = M_inc.reshape(-1, M, N).to(torch.float64).contiguous()
        B = flat.shape[0]

        K = torch.ones(B, M + 1, N + 1, device=flat.device, dtype=torch.float64)
        BLOCK = min(triton.next_power_of_2(max(M, N)), _TRITON_MAX_BLOCK)

        ms_b, ms_i, ms_j = flat.stride(0), flat.stride(1), flat.stride(2)
        ks_b, ks_i, ks_j = K.stride(0), K.stride(1), K.stride(2)

        for d in range(2, M + N + 1):
            i0 = max(1, d - N)
            i1 = min(M, d - 1)
            n_cells = i1 - i0 + 1
            n_chunks = (n_cells + BLOCK - 1) // BLOCK

            _sigkernel_diag_step[(B, n_chunks)](
                flat, K,
                ms_b, ms_i, ms_j,
                ks_b, ks_i, ks_j,
                i0, n_cells, d,
                BLOCK,
            )

        return K.reshape(*batch, M + 1, N + 1)


def solve_pde(M_inc: torch.Tensor) -> torch.Tensor:
    """Dispatch to Triton (GPU, float64) or PyTorch (CPU) PDE solver.

    Falls back to PyTorch if Triton fails (e.g. stale CUDA context error
    from a prior async operation, or kernel compilation failure).
    """
    if HAS_TRITON and M_inc.is_cuda:
        try:
            # Sync first to surface any stale async CUDA errors before
            # Triton tries to load/compile a kernel binary
            torch.cuda.synchronize(M_inc.device)
            return _solve_pde_triton(M_inc)
        except RuntimeError as e:
            if "illegal memory access" in str(e) or "CUDA" in str(e):
                import warnings
                warnings.warn(
                    f"Triton PDE solver failed ({e}), falling back to PyTorch. "
                    f"This may indicate a prior CUDA error.",
                    RuntimeWarning, stacklevel=2)
                # Fall back to PyTorch (runs on same device, no Triton)
                return _solve_pde(M_inc)
            raise
    return _solve_pde(M_inc)


# ════════════════════════════════════════════════════════════════════════════
# Static kernel
# ════════════════════════════════════════════════════════════════════════════

class RBFKernel(nn.Module):
    r"""RBF kernel  k(x,y) = exp(-\|x-y\|^2 / σ).

    NOTE: the denominator is σ, **not** 2σ².  This matches the sigkernel
    library and means σ has units of squared-distance.
    """

    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma

    # -- full cross-batch kernel (used by compute_Gram) -------------------
    def batch_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """(B_x, N_x, L_x, D) × (B_y, N_y, L_y, D) → (B_x, N_x, B_y, N_y, L_x, L_y)"""
        Bx, Nx, Lx, D = X.shape
        By, Ny, Ly, _  = Y.shape
        Xf = X.reshape(Bx * Nx, Lx, D)
        Yf = Y.reshape(By * Ny, Ly, D)
        Xsq = (Xf ** 2).sum(-1).unsqueeze(1).unsqueeze(3)   # (BxNx,1,Lx,1)
        Ysq = (Yf ** 2).sum(-1).unsqueeze(0).unsqueeze(2)   # (1,ByNy,1,Ly)
        XY  = torch.einsum('bld,cde->bcle', Xf, Yf.transpose(1, 2))
        dist = torch.clamp(Xsq + Ysq - 2 * XY, min=0.0)
        return torch.exp(-dist / self.sigma).reshape(Bx, Nx, By, Ny, Lx, Ly)

    # -- block-diagonal kernel (used by compute_diag_Gram) ----------------
    def batch_kernel_diag(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """(B, N_x, L_x, D) × (B, N_y, L_y, D) → (B, N_x, N_y, L_x, L_y)"""
        Xsq = (X ** 2).sum(-1)                                       # (B,Nx,Lx)
        Ysq = (Y ** 2).sum(-1)                                       # (B,Ny,Ly)
        XY  = torch.einsum('bild,bjkd->bijlk', X, Y)                 # (B,Nx,Ny,Lx,Ly)
        dist = torch.clamp(
            Xsq[:, :, None, :, None] + Ysq[:, None, :, None, :] - 2 * XY,
            min=0.0,
        )
        return torch.exp(-dist / self.sigma)


# ════════════════════════════════════════════════════════════════════════════
# Autograd functions
# ════════════════════════════════════════════════════════════════════════════

def _compute_M_inc(G_static: torch.Tensor, dyadic_order: int) -> torch.Tensor:
    """Double-difference of static kernel + optional dyadic tiling.

    Matches sigkernel: NO padding.  The double-difference gives the discrete
    mixed second derivative ∂²k/∂s∂t, producing an (L-1)×(L-1) tensor from
    an L×L static kernel.  Padding with zeros would inject spurious O(1)
    increments in the first row/column (equal to raw kernel values instead
    of small second differences), causing the PDE to overflow.
    """
    M = (G_static[..., 1:, 1:] + G_static[..., :-1, :-1]
         - G_static[..., 1:, :-1] - G_static[..., :-1, 1:])
    if dyadic_order > 0:
        n = 2 ** dyadic_order
        M = _tile(_tile(M, -2, n), -1, n) / float(n * n)
    return M


class SigKernelFn(torch.autograd.Function):
    """Full cross-batch signature kernel (autograd)."""

    @staticmethod
    def forward(ctx, X, Y, static_kernel, dyadic_order):
        G = static_kernel.batch_kernel(X, Y)
        M = _compute_M_inc(G, dyadic_order)
        K = solve_pde(M)
        ctx.save_for_backward(M, K, X, Y)
        ctx.static_kernel = static_kernel
        ctx.dyadic_order = dyadic_order
        return K[..., -1, -1]

    @staticmethod
    def backward(ctx, grad_output):
        M, K, X, Y = ctx.saved_tensors
        M_rev = torch.flip(M, dims=[-1, -2])
        K_rev = solve_pde(M_rev)
        K_rev = torch.flip(K_rev, dims=[-1, -2])

        # Adjoint:  dK[L,L]/dM[i,j] ≈ K_fwd[i,j] · K_rev[i+1,j+1]
        grad_M = (K[..., :-1, :-1] * K_rev[..., 1:, 1:]
                  * grad_output[..., None, None])

        # Un-tile gradient if dyadic refinement was applied
        n = 2 ** ctx.dyadic_order
        if n > 1:
            *b, MM, NN = grad_M.shape
            grad_M = grad_M.reshape(*b, MM // n, n, NN // n, n).sum(dim=(-1, -3)) / float(n * n)

        # Back-prop through double difference
        gp = torch.nn.functional.pad(grad_M, (0, 1, 0, 1))
        grad_G = gp[..., :-1, :-1] + gp[..., 1:, 1:] - gp[..., :-1, 1:] - gp[..., 1:, :-1]

        with torch.enable_grad():
            Xg = X.detach().requires_grad_(True)
            Yg = Y.detach().requires_grad_(True)
            G_replay = ctx.static_kernel.batch_kernel(Xg, Yg)
            gX, gY = torch.autograd.grad(G_replay, [Xg, Yg], grad_G)
        return gX, gY, None, None


class SigKernelDiagFn(torch.autograd.Function):
    """Block-diagonal signature kernel (autograd)."""

    @staticmethod
    def forward(ctx, X, Y, static_kernel, dyadic_order):
        G = static_kernel.batch_kernel_diag(X, Y)
        M = _compute_M_inc(G, dyadic_order)
        K = solve_pde(M)
        ctx.save_for_backward(M, K, X, Y)
        ctx.static_kernel = static_kernel
        ctx.dyadic_order = dyadic_order
        return K[..., -1, -1]

    @staticmethod
    def backward(ctx, grad_output):
        M, K, X, Y = ctx.saved_tensors
        M_rev = torch.flip(M, dims=[-1, -2])
        K_rev = solve_pde(M_rev)
        K_rev = torch.flip(K_rev, dims=[-1, -2])

        grad_M = (K[..., :-1, :-1] * K_rev[..., 1:, 1:]
                  * grad_output[..., None, None])

        n = 2 ** ctx.dyadic_order
        if n > 1:
            *b, MM, NN = grad_M.shape
            grad_M = grad_M.reshape(*b, MM // n, n, NN // n, n).sum(dim=(-1, -3)) / float(n * n)

        gp = torch.nn.functional.pad(grad_M, (0, 1, 0, 1))
        grad_G = gp[..., :-1, :-1] + gp[..., 1:, 1:] - gp[..., :-1, 1:] - gp[..., 1:, :-1]

        with torch.enable_grad():
            Xg = X.detach().requires_grad_(True)
            Yg = Y.detach().requires_grad_(True)
            G_replay = ctx.static_kernel.batch_kernel_diag(Xg, Yg)
            gX, gY = torch.autograd.grad(G_replay, [Xg, Yg], grad_G)
        return gX, gY, None, None


# ════════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════════

class SigKernel(nn.Module):
    """Signature kernel with real-space Padé PDE solver.

    Parameters
    ----------
    static_kernel : RBFKernel
        Base kernel for lifting paths into RKHS.
    dyadic_order : int, default 0
        Dyadic refinement depth.  Order *n* subdivides each PDE cell into
        (2^n)² sub-cells, improving numerical accuracy at the cost of a
        (2^n)²-fold increase in PDE grid size.  ``dyadic_order=1`` matches
        the default of the ``sigkernel`` library.
    """

    def __init__(self, static_kernel: RBFKernel, dyadic_order: int = 0):
        super().__init__()
        self.static_kernel = static_kernel
        self.dyadic_order = dyadic_order

    # -- full Gram --------------------------------------------------------
    def compute_Gram(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        sq_x = sq_y = False
        if X.ndim == 3: X = X.unsqueeze(1); sq_x = True
        if Y.ndim == 3: Y = Y.unsqueeze(1); sq_y = True
        K = SigKernelFn.apply(X, Y, self.static_kernel, self.dyadic_order)
        if sq_x: K = K.squeeze(1)
        if sq_y: K = K.squeeze(-1)
        return K

    # -- block-diagonal Gram ----------------------------------------------
    def compute_diag_Gram(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        sq_x = sq_y = False
        if X.ndim == 3: X = X.unsqueeze(1); sq_x = True
        if Y.ndim == 3: Y = Y.unsqueeze(1); sq_y = True
        K = SigKernelDiagFn.apply(X, Y, self.static_kernel, self.dyadic_order)
        if sq_x: K = K.squeeze(1)
        if sq_y: K = K.squeeze(-1)
        return K

    # -- diagonal (self-kernel) -------------------------------------------
    def compute_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        K = self.compute_Gram(X, Y)
        return torch.diagonal(K)


# Backward-compatible aliases
LogSigKernel = SigKernel
LogSigKernelDiagFn = SigKernelDiagFn
