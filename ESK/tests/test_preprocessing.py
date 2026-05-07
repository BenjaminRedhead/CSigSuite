"""
Tests for preprocessing utilities.
"""
import pytest
import torch
from sigkernel_mmd import PathReVIN, AugmentPipeline, SigmaEstimator


class TestPathReVIN:
    """Test path normalization."""
    
    def test_zero_mean_unit_var(self):
        """Normalized data should have zero mean and unit variance."""
        normalizer = PathReVIN(dim=5)
        data = torch.randn(100, 50, 5) * 10 + 5
        
        normalizer.fit(data)
        data_norm = normalizer(data)
        
        # Check mean ≈ 0
        assert torch.allclose(data_norm.mean(dim=(0, 1)), torch.zeros(5), atol=1e-5)
        
        # Check std ≈ 1
        assert torch.allclose(data_norm.std(dim=(0, 1)), torch.ones(5), atol=1e-2)
    
    def test_constant_feature_handling(self):
        """Should handle constant features gracefully."""
        normalizer = PathReVIN(dim=3)
        data = torch.randn(100, 50, 3)
        data[:, :, 1] = 5.0  # Make second feature constant
        
        normalizer.fit(data)
        data_norm = normalizer(data)
        
        # Should not produce NaN or Inf
        assert not torch.isnan(data_norm).any()
        assert not torch.isinf(data_norm).any()
    
    def test_fit_transform_consistency(self):
        """Transform should use statistics from fit."""
        normalizer = PathReVIN(dim=5)
        train_data = torch.randn(100, 50, 5)
        test_data = torch.randn(20, 50, 5)
        
        normalizer.fit(train_data)
        
        # Transform uses train statistics
        test_norm = normalizer(test_data)
        
        # Test mean should NOT be exactly zero
        test_mean = test_norm.mean(dim=(0, 1))
        assert not torch.allclose(test_mean, torch.zeros(5), atol=1e-2)


class TestAugmentPipeline:
    """Test path augmentation."""
    
    def test_time_augmentation(self):
        """Should append time channel."""
        augmenter = AugmentPipeline(append_time=True)
        x = torch.randn(10, 50, 5)
        
        x_aug = augmenter(x)
        
        # Should add basepoint, endpoint, and time channel
        assert x_aug.shape == (10, 52, 6)  # 50+2 length, 5+1 dim
    
    def test_no_time_augmentation(self):
        """Should work without time channel."""
        augmenter = AugmentPipeline(append_time=False)
        x = torch.randn(10, 50, 5)
        
        x_aug = augmenter(x)
        
        # Should only add basepoint and endpoint
        assert x_aug.shape == (10, 52, 5)  # 50+2 length, same dim
    
    def test_basepoint_is_zero(self):
        """First point should be zero."""
        augmenter = AugmentPipeline(append_time=True)
        x = torch.randn(10, 50, 5)
        
        x_aug = augmenter(x)
        
        # Basepoint should be all zeros
        assert torch.allclose(x_aug[:, 0, :], torch.zeros(10, 6))
    
    def test_time_channel_range(self):
        """Time channel should be in [0, 1]."""
        augmenter = AugmentPipeline(append_time=True)
        x = torch.randn(10, 50, 5)
        
        x_aug = augmenter(x)
        
        # Last dimension is time
        time_channel = x_aug[:, 1:-1, -1]  # Exclude basepoint and endpoint
        
        assert time_channel.min() >= 0.0
        assert time_channel.max() <= 1.0


class TestSigmaEstimator:
    """Test sigma estimation."""
    
    def test_positive_sigma(self):
        """Sigma should always be positive."""
        data = torch.randn(100, 50, 5)
        sigma = SigmaEstimator.fit(data)
        
        assert sigma > 0
    
    def test_reasonable_magnitude(self):
        """Sigma should be reasonable magnitude for unit variance data."""
        data = torch.randn(100, 50, 5)
        sigma = SigmaEstimator.fit(data)
        
        # For unit variance data, sigma should be O(1) to O(100)
        assert 0.1 < sigma < 1000
    
    def test_subsampling(self):
        """Should subsample large datasets."""
        data = torch.randn(10000, 50, 5)
        sigma = SigmaEstimator.fit(data, max_samples=100)
        
        # Should complete without error
        assert sigma > 0
    
    def test_scale_invariance(self):
        """Sigma should scale with data scale."""
        data = torch.randn(100, 50, 5)
        
        sigma1 = SigmaEstimator.fit(data)
        sigma10 = SigmaEstimator.fit(data * 10)
        
        # Sigma should scale roughly linearly
        assert 5 < sigma10 / sigma1 < 15
    
    def test_empty_or_zero_data(self):
        """Should handle edge cases gracefully."""
        # All zeros
        data_zero = torch.zeros(10, 50, 5)
        sigma = SigmaEstimator.fit(data_zero)
        
        # Should return default value
        assert sigma == 1.0


@pytest.fixture
def device():
    """Get available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestGPUCompatibility:
    """Test GPU compatibility for preprocessing."""
    
    def test_revin_on_gpu(self, device):
        """PathReVIN should work on GPU."""
        if device.type == 'cpu':
            pytest.skip("GPU not available")
        
        normalizer = PathReVIN(dim=5).to(device)
        data = torch.randn(100, 50, 5, device=device)
        
        normalizer.fit(data)
        data_norm = normalizer(data)
        
        assert data_norm.device.type == 'cuda'
    
    def test_augmenter_on_gpu(self, device):
        """AugmentPipeline should work on GPU."""
        if device.type == 'cpu':
            pytest.skip("GPU not available")
        
        augmenter = AugmentPipeline(append_time=True).to(device)
        data = torch.randn(10, 50, 5, device=device)
        
        data_aug = augmenter(data)
        
        assert data_aug.device.type == 'cuda'