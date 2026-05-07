"""
Signature Kernel MMD
====================

Efficient implementation of signature kernel Maximum Mean Discrepancy
for probabilistic time series evaluation.

Basic Usage
-----------
>>> import torch
>>> from sigkernel_mmd import SigMMD, LogSigKernel, RBFKernel
>>> from sigkernel_mmd import PathReVIN, AugmentPipeline, SigmaEstimator
>>> 
>>> # Prepare data
>>> preds = torch.randn(100, 20, 50, 5)   # (batch, samples, length, dim)
>>> target = torch.randn(100, 50, 5)      # (batch, length, dim)
>>> 
>>> # Preprocessing
>>> normalizer = PathReVIN(dim=5)
>>> normalizer.fit(target)
>>> target_norm = normalizer(target)
>>> preds_norm = normalizer(preds.view(-1, 50, 5)).view(100, 20, 50, 5)
>>> 
>>> augmenter = AugmentPipeline(append_time=True)
>>> target_aug = augmenter(target_norm)
>>> preds_aug = augmenter(preds_norm.view(-1, 50, 5)).view(100, 20, -1, 6)
>>> 
>>> # Estimate sigma
>>> sigma = SigmaEstimator.fit(target_norm)
>>> 
>>> # Create metric
>>> static = RBFKernel(sigma=sigma)
>>> kernel = LogSigKernel(static)
>>> metric = SigMMD(kernel)
>>> 
>>> # Evaluate
>>> metric.update(preds_aug, target_aug)
>>> score = metric.compute()
>>> print(f"Sig-MMD: {score:.6f}")
"""

__version__ = '0.1.0'
__author__ = 'Anonymous'
__license__ = 'MIT'

from .kernels import RBFKernel, SigKernel, LogSigKernel
from .metrics import SigMMD
from .preprocessing import PathReVIN, AugmentPipeline, SigmaEstimator

__all__ = [
    # Kernels
    'RBFKernel',
    'SigKernel',
    'LogSigKernel',   # backward-compat alias for SigKernel

    # Metrics
    'SigMMD',

    # Preprocessing
    'PathReVIN',
    'AugmentPipeline',
    'SigmaEstimator',
]