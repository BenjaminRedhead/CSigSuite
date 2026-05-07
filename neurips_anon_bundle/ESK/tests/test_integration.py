"""
Integration tests for full pipeline.
"""
import pytest
import torch
from sigkernel_mmd import (
    SigMMD,
    LogSigKernel,
    RBFKernel,
    PathReVIN,
    AugmentPipeline,
    SigmaEstimator,
)


class TestFullPipeline:
    """Test complete evaluation pipeline."""
    
    def test_end_to_end(self):
        """Test complete workflow from raw data to MMD."""
        # Raw data
        preds = torch.randn(50, 10, 30, 5)
        target = torch.randn(50, 30, 5)
        
        # Step 1: Normalize
        normalizer = PathReVIN(dim=5)
        normalizer.fit(target)
        target_norm = normalizer(target)
        preds_norm = normalizer(preds.view(-1, 30, 5)).view(50, 10, 30, 5)
        
        # Step 2: Augment
        augmenter = AugmentPipeline(append_time=True)
        target_aug = augmenter(target_norm)
        preds_aug = augmenter(preds_norm.view(-1, 30, 5)).view(50, 10, -1, 6)
        
        # Step 3: Estimate sigma
        sigma = SigmaEstimator.fit(target_norm)
        
        # Step 4: Create metric
        static = RBFKernel(sigma=sigma)
        kernel = LogSigKernel(static)
        metric = SigMMD(kernel)
        
        # Step 5: Compute MMD
        metric.update(preds_aug, target_aug)
        mmd = metric.compute()
        
        # Should complete successfully
        assert not torch.isnan(mmd)
        assert not torch.isinf(mmd)
        assert mmd >= -1e-6, f"MMD should be non-negative, got {mmd}"
    
    def test_multiple_batches(self):
        """Test evaluation over multiple batches."""
        normalizer = PathReVIN(dim=3)
        augmenter = AugmentPipeline(append_time=True)
        
        # Fit on train data
        train_data = torch.randn(100, 20, 3)
        normalizer.fit(train_data)
        sigma = SigmaEstimator.fit(normalizer(train_data))
        
        # Create metric
        kernel = LogSigKernel(RBFKernel(sigma=sigma))
        metric = SigMMD(kernel)
        
        # Evaluate on multiple test batches
        for _ in range(5):
            preds = torch.randn(10, 5, 20, 3)
            target = torch.randn(10, 20, 3)
            
            target_norm = normalizer(target)
            preds_norm = normalizer(preds.view(-1, 20, 3)).view(10, 5, 20, 3)
            
            target_aug = augmenter(target_norm)
            preds_aug = augmenter(preds_norm.view(-1, 20, 3)).view(10, 5, -1, 4)
            
            metric.update(preds_aug, target_aug)
        
        mmd = metric.compute()
        
        assert not torch.isnan(mmd)
    
    def test_different_sequence_lengths(self):
        """Test with different sequence lengths."""
        for length in [10, 30, 50]:
            preds = torch.randn(20, 5, length, 3)
            target = torch.randn(20, length, 3)
            
            normalizer = PathReVIN(dim=3)
            normalizer.fit(target)
            
            augmenter = AugmentPipeline(append_time=True)
            
            target_norm = normalizer(target)
            preds_norm = normalizer(preds.view(-1, length, 3)).view(20, 5, length, 3)
            
            target_aug = augmenter(target_norm)
            preds_aug = augmenter(preds_norm.view(-1, length, 3)).view(20, 5, -1, 4)
            
            sigma = SigmaEstimator.fit(target_norm)
            kernel = LogSigKernel(RBFKernel(sigma=sigma))
            metric = SigMMD(kernel)
            
            metric.update(preds_aug, target_aug)
            mmd = metric.compute()
            
            assert not torch.isnan(mmd)


@pytest.fixture
def device():
    """Get available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestGPUPipeline:
    """Test full pipeline on GPU."""
    
    def test_gpu_end_to_end(self, device):
        """Test complete pipeline on GPU."""
        if device.type == 'cpu':
            pytest.skip("GPU not available")
        
        # All on GPU
        preds = torch.randn(20, 5, 30, 3, device=device)
        target = torch.randn(20, 30, 3, device=device)
        
        normalizer = PathReVIN(dim=3).to(device)
        normalizer.fit(target)
        
        augmenter = AugmentPipeline(append_time=True).to(device)
        
        target_norm = normalizer(target)
        preds_norm = normalizer(preds.view(-1, 30, 3)).view(20, 5, 30, 3)
        
        target_aug = augmenter(target_norm)
        preds_aug = augmenter(preds_norm.view(-1, 30, 3)).view(20, 5, -1, 4)
        
        sigma = SigmaEstimator.fit(target_norm.cpu())  # Sigma est on CPU
        kernel = LogSigKernel(RBFKernel(sigma=sigma))
        metric = SigMMD(kernel).to(device)
        
        metric.update(preds_aug, target_aug)
        mmd = metric.compute()
        
        assert mmd.device.type == 'cuda'
        assert not torch.isnan(mmd)