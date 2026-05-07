"""
Preprocessing utilities for time series paths.
"""
import torch
import torch.nn as nn

__all__ = ['PathReVIN', 'AugmentPipeline', 'SigmaEstimator']


class PathReVIN(nn.Module):
    """
    Reversible Instance Normalization for time series paths.
    
    Normalizes each path to zero mean and unit variance, with
    special handling for constant features.
    
    Parameters
    ----------
    dim : int
        Feature dimension
    eps : float, default=1e-5
        Small constant for numerical stability
        
    Examples
    --------
    >>> normalizer = PathReVIN(dim=5)
    >>> train_data = torch.randn(1000, 50, 5)
    >>> normalizer.fit(train_data)
    >>> 
    >>> test_data = torch.randn(100, 50, 5)
    >>> test_normalized = normalizer(test_data)
    """
    
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer('mean', torch.zeros(1, 1, dim))
        self.register_buffer('std', torch.ones(1, 1, dim))
    
    def fit(self, x: torch.Tensor) -> None:
        """
        Compute normalization statistics from training data.
        
        Parameters
        ----------
        x : torch.Tensor
            Training data of shape (batch, length, dim)
        """
        # Compute statistics over batch and time
        self.mean = x.mean(dim=(0, 1), keepdim=True)
        std = x.std(dim=(0, 1), keepdim=True)
        
        # Handle constant features: set std=1 to avoid division by zero
        self.std = torch.where(std < 1e-1, torch.ones_like(std), std)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize paths."""
        return (x - self.mean) / (self.std + self.eps)


class AugmentPipeline(nn.Module):
    """
    Path augmentation with time, basepoint, and endpoint.
    
    Augments paths to make signature kernel sensitive to:
    - Translation (basepoint at origin)
    - Time parameterization (time channel)
    - Duration (endpoint differs from last point)
    
    Parameters
    ----------
    append_time : bool, default=True
        Whether to append time channel
        
    Examples
    --------
    >>> augmenter = AugmentPipeline(append_time=True)
    >>> x = torch.randn(100, 50, 5)  # (batch, length, dim)
    >>> x_aug = augmenter(x)         # (batch, 52, 6)
    """
    
    def __init__(self, append_time: bool = True):
        super().__init__()
        self.append_time = append_time
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Augment paths.
        
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, length, dim)
            
        Returns
        -------
        x_aug : torch.Tensor
            Shape (batch, length+2, dim+1) if append_time=True
            Shape (batch, length+2, dim) otherwise
        """
        B, L, D = x.shape
        device = x.device
        
        # Append time channel if requested
        if self.append_time:
            time = torch.linspace(0, 1, L, device=device)
            time = time.reshape(1, L, 1).expand(B, L, 1)
            x = torch.cat([x, time], dim=2)
            D = D + 1
        
        # Basepoint: zero vector
        zero_start = torch.zeros(B, 1, D, device=device)
        
        # Endpoint: repeat last time coordinate, zero out features
        zero_end = torch.zeros(B, 1, D, device=device)
        if self.append_time:
            zero_end[:, :, -1] = x[:, -1:, -1]  # Keep last time value
        
        # Concatenate: [basepoint, path, endpoint]
        return torch.cat([zero_start, x, zero_end], dim=1)


class SigmaEstimator:
    """
    Median heuristic for RBF bandwidth estimation.
    
    Estimates σ as the median pairwise distance in a sample,
    providing a robust automatic bandwidth selection.
    
    Examples
    --------
    >>> train_data = torch.randn(1000, 50, 5)
    >>> sigma = SigmaEstimator.fit(train_data)
    >>> print(f"Estimated sigma: {sigma:.4f}")
    """
    
    @staticmethod
    def fit(
        data: torch.Tensor,
        max_samples: int = 500
    ) -> float:
        """
        Estimate sigma from data using median heuristic.
        
        Parameters
        ----------
        data : torch.Tensor
            Shape (batch, length, dim)
        max_samples : int, default=500
            Maximum number of samples to use (for efficiency)
            
        Returns
        -------
        sigma : float
            Estimated bandwidth
        """
        N = data.shape[0]

        # Flatten to vectors
        x_flat = data.reshape(N, -1)

        # Subsample if needed
        if N > max_samples:
            idx = torch.randperm(N)[:max_samples]
            subset = x_flat[idx]
        else:
            subset = x_flat
        
        # Compute pairwise distances
        dists = torch.cdist(subset, subset, p=2)
        
        # Filter out zeros (self-distances)
        valid_dists = dists[dists > 1e-6]
        
        if len(valid_dists) > 0:
            return torch.median(valid_dists).item()
        else:
            return 1.0