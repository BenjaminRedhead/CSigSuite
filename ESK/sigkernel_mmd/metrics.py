"""
Efficient Signature MMD implementation with batched computation.
"""
import torch
import torch.nn as nn
from typing import Literal, Optional
from .kernels import SigKernel, LogSigKernel

__all__ = ['SigMMD']


class SigMMD(nn.Module):
    """
    Efficient Signature kernel Maximum Mean Discrepancy.
    
    Computes MMD² = ||μ_P - μ_Q||²_H using signature kernel embeddings
    with efficient batched computation and optional normalization.
    
    Uses biased V-statistic estimator:
        MMD²_V = E[k(X,X)] - 2E[k(X,Y)] + E[k(Y,Y)]
    
    This estimator includes diagonal terms and has lower variance than
    the unbiased U-statistic, making it more stable for small sample sizes.
    
    Parameters
    ----------
    kernel : LogSigKernel
        Signature kernel instance
    normalize : bool, default=True
        Whether to normalize kernel by diagonal values:
            k_norm(x,y) = k(x,y) / sqrt(k(x,x) * k(y,y))
        
    Attributes
    ----------
    total_mmd : torch.Tensor
        Running sum of MMD² values
    total_count : torch.Tensor
        Running count of batches
        
    Examples
    --------
    >>> from sigkernel_mmd import SigMMD, LogSigKernel, RBFKernel
    >>> 
    >>> # Setup
    >>> static = RBFKernel(sigma=1.0)
    >>> kernel = LogSigKernel(static)
    >>> metric = SigMMD(kernel)
    >>> 
    >>> # Predictions: (batch, samples, length, dim)
    >>> preds = torch.randn(100, 20, 50, 5)
    >>> # Ground truth: (batch, length, dim)
    >>> target = torch.randn(100, 50, 5)
    >>> 
    >>> # Update metric
    >>> metric.update(preds, target)
    >>> 
    >>> # Get result
    >>> mmd_value = metric.compute()
    >>> print(f"Sig-MMD: {mmd_value:.6f}")
    >>> 
    >>> # Reset for new evaluation
    >>> metric.reset()
    
    Notes
    -----
    Block-diagonal computation:
    - Predictions maintain (B, S, L, D) structure
    - Target is treated as (B, 1, L, D) per-batch ground truth
    - Only within-batch kernel pairs are computed; cross-batch terms
      k(X[b,s], X[b',s']) for b≠b' are never materialised, preventing
      spurious inter-batch correlations in the ground truth.

    Memory complexity: O(B × S²) for Gram matrices
    Time complexity: O(B × S² × L² × D) for kernel computation
    (factor-B improvement over full cross-batch Gram matrices)
    
    References
    ----------
    Gretton et al. (2012). "A Kernel Two-Sample Test". JMLR.
    """
    
    def __init__(
        self,
        kernel: LogSigKernel,
        normalize: bool = True,
        max_batch: int = 4,
    ):
        super().__init__()
        self.kernel = kernel
        self.normalize = normalize
        self.max_batch = max_batch

        # State buffers for distributed computation
        self.register_buffer('total_mmd', torch.tensor(0.0))
        self.register_buffer('total_count', torch.tensor(0))

    def _update_chunk(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Process a single micro-batch without further splitting."""
        B, S, L, D = preds.shape
        target_batched = target.unsqueeze(1)  # (B, 1, L, D)

        K_XX = self.kernel.compute_diag_Gram(preds, preds)
        K_YY = self.kernel.compute_diag_Gram(target_batched, target_batched).squeeze(-1).squeeze(-1)
        K_XY = self.kernel.compute_diag_Gram(preds, target_batched).squeeze(-1)

        if self.normalize:
            diag_X = torch.diagonal(K_XX, dim1=-2, dim2=-1)
            sqrt_diag_X = torch.sqrt(diag_X + 1e-8)

            norm_XX = sqrt_diag_X.unsqueeze(-1) * sqrt_diag_X.unsqueeze(-2)
            K_XX_norm = K_XX / (norm_XX + 1e-8)

            norm_XY = torch.sqrt(diag_X * K_YY.unsqueeze(-1) + 1e-8)
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

    def update(
        self,
        preds: torch.Tensor,
        target: torch.Tensor
    ) -> None:
        """
        Update metric with a batch of predictions and targets.

        Automatically micro-batches to stay within GPU memory.  The
        ``max_batch`` constructor arg controls the chunk size (default 4).

        Parameters
        ----------
        preds : torch.Tensor
            Predictions with shape (B, S, L, D)
        target : torch.Tensor
            Ground truth with shape (B, L, D)
        """
        if preds.ndim != 4:
            raise ValueError(
                f"preds must be 4D (batch, samples, length, dim), "
                f"got shape {preds.shape}"
            )
        if target.ndim != 3:
            raise ValueError(
                f"target must be 3D (batch, length, dim), "
                f"got shape {target.shape}"
            )

        L = preds.shape[2]
        if L > 150 and self.kernel.dyadic_order > 0:
            self.kernel.dyadic_order = 0

        B, S, L, D = preds.shape

        if target.shape != (B, L, D):
            raise ValueError(
                f"target shape {target.shape} incompatible with "
                f"preds shape {preds.shape}"
            )

        # Micro-batch to bound peak GPU memory.
        # Peak allocation per element ≈ S² × L_dyadic² × 8 bytes (float64 PDE).
        mb = self.max_batch
        for i in range(0, B, mb):
            j = min(i + mb, B)
            self._update_chunk(preds[i:j], target[i:j])
    
    def compute(self) -> torch.Tensor:
        """
        Compute final MMD value.
        
        Returns
        -------
        mmd : torch.Tensor
            Scalar MMD value averaged over all batches
        """
        if self.total_count == 0:
            raise RuntimeError("No data has been added to metric")
        
        return self.total_mmd / self.total_count
    
    def reset(self) -> None:
        """Reset metric state for new evaluation."""
        self.total_mmd.zero_()
        self.total_count.zero_()