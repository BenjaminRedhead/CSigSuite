# Signature Kernel MMD

Efficient implementation of signature kernel Maximum Mean Discrepancy (MMD) for
probabilistic time series forecasting evaluation.

## Features

- **Log-space computation** for numerical stability with long sequences
- **Efficient 4D batching** — no data duplication or flattening
- **Biased V-statistic** for lower variance estimation
- **GPU accelerated** via PyTorch
- **Fully differentiable** with custom autograd
- **Minimal dependencies** — just PyTorch and NumPy

## Installation

From the bundle root:
```bash
pip install -e ESK
```

## Quick Start

```python
import torch
from sigkernel_mmd import (
    SigMMD, LogSigKernel, RBFKernel,
    PathReVIN, AugmentPipeline, SigmaEstimator,
)

# Data: predictions with samples, ground truth
preds = torch.randn(100, 20, 50, 5)   # (batch, samples, length, dim)
target = torch.randn(100, 50, 5)      # (batch, length, dim)

# Preprocessing
normalizer = PathReVIN(dim=5)
normalizer.fit(target)
target_norm = normalizer(target)
preds_norm = normalizer(preds.view(-1, 50, 5)).view(100, 20, 50, 5)

augmenter = AugmentPipeline(append_time=True)
target_aug = augmenter(target_norm)
preds_aug = augmenter(preds_norm.view(-1, 50, 5)).view(100, 20, -1, 6)

# Estimate bandwidth
sigma = SigmaEstimator.fit(target_norm)

# Create metric
static = RBFKernel(sigma=sigma)
kernel = LogSigKernel(static)
metric = SigMMD(kernel)

# Evaluate
metric.update(preds_aug, target_aug)
score = metric.compute()
print(f"Sig-MMD: {score:.6f}")
```

## Why Signature Kernels?

Traditional metrics like CRPS assume independence across time steps and
variables. Signature kernels:

- Capture **temporal dependencies** naturally
- Handle **multivariate correlations** without assumptions
- Are **proper scoring rules** with unbiased estimators
- Scale **linearly** with sequence length (not exponentially)

## Performance

Compared to naive signature computation:

| Metric          | Naive               | This Library         |
|-----------------|---------------------|----------------------|
| Memory          | O(N · S · d^K)      | O(N² · S²)           |
| Time            | Exponential in dim  | Linear in dim        |
| Numerical Range | 10^-50 to 10^50     | Stable via log-space |

## License

MIT License — see LICENSE file.
