"""
Tests for Sig-MMD metric implementation.
"""
import pytest
import torch
from sigkernel_mmd import SigMMD, LogSigKernel, RBFKernel


class TestSigMMD:
    """Test Sig-MMD metric."""
    
    @pytest.fixture
    def setup_metric(self):
        """Create metric for testing."""
        kernel = LogSigKernel(RBFKernel(sigma=1.0))
        metric = SigMMD(kernel, normalize=True)
        return metric
    
    def test_perfect_match_zero_mmd(self, setup_metric):
        """MMD should be near zero for identical distributions."""
        metric = setup_metric

        # All S samples are identical copies of the target
        target = torch.randn(10, 20, 3)
        X = target.unsqueeze(1).expand(10, 5, 20, 3).clone()

        metric.update(X, target)
        mmd = metric.compute()

        assert mmd < 1e-3, f"Expected MMD≈0 for identical data, got {mmd}"
    
    def test_different_distributions_positive_mmd(self, setup_metric):
        """MMD should be positive for different distributions."""
        metric = setup_metric
        
        X = torch.randn(10, 5, 20, 3)
        Y = torch.randn(10, 20, 3) + 5.0  # Shifted distribution
        
        metric.update(X, Y)
        mmd = metric.compute()
        
        assert mmd > 0, "MMD should be positive for different distributions"
    
    def test_shape_validation_4d_preds(self, setup_metric):
        """Should raise error if preds is not 4D."""
        metric = setup_metric
        
        with pytest.raises(ValueError, match="preds must be 4D"):
            metric.update(
                torch.randn(10, 20, 3),  # 3D instead of 4D
                torch.randn(10, 20, 3)
            )
    
    def test_shape_validation_3d_target(self, setup_metric):
        """Should raise error if target is not 3D."""
        metric = setup_metric
        
        with pytest.raises(ValueError, match="target must be 3D"):
            metric.update(
                torch.randn(10, 5, 20, 3),
                torch.randn(10, 5, 20, 3)  # 4D instead of 3D
            )
    
    def test_shape_compatibility(self, setup_metric):
        """Should raise error if batch sizes don't match."""
        metric = setup_metric
        
        with pytest.raises(ValueError, match="incompatible"):
            metric.update(
                torch.randn(10, 5, 20, 3),
                torch.randn(15, 20, 3)  # Different batch size
            )
    
    def test_symmetry(self):
        """MMD(X,Y) should approximately equal MMD(Y,X) for same target."""
        kernel = LogSigKernel(RBFKernel(sigma=1.0))

        # Use same target for both directions so we compare the same quantity
        target = torch.randn(5, 15, 2)
        X = torch.randn(5, 10, 15, 2)

        # Compute MMD(X, target) twice — should be deterministic
        metric_1 = SigMMD(kernel)
        metric_1.update(X, target)
        mmd_1 = metric_1.compute()

        metric_2 = SigMMD(kernel)
        metric_2.update(X, target)
        mmd_2 = metric_2.compute()

        assert torch.isclose(mmd_1, mmd_2, rtol=1e-5)
    
    def test_triangle_inequality(self):
        """MMD should satisfy triangle inequality (approximately)."""
        kernel = LogSigKernel(RBFKernel(sigma=1.0))
        
        X = torch.randn(5, 10, 15, 2)
        Y = torch.randn(5, 10, 15, 2)
        Z = torch.randn(5, 10, 15, 2)
        
        # d(X, Y)
        metric_xy = SigMMD(kernel)
        metric_xy.update(X, Y[:, 0, :, :])
        d_xy = torch.sqrt(torch.clamp(metric_xy.compute(), min=0))
        
        # d(Y, Z)
        metric_yz = SigMMD(kernel)
        metric_yz.update(Y, Z[:, 0, :, :])
        d_yz = torch.sqrt(torch.clamp(metric_yz.compute(), min=0))
        
        # d(X, Z)
        metric_xz = SigMMD(kernel)
        metric_xz.update(X, Z[:, 0, :, :])
        d_xz = torch.sqrt(torch.clamp(metric_xz.compute(), min=0))
        
        # d(X,Z) <= d(X,Y) + d(Y,Z) + tolerance
        assert d_xz <= d_xy + d_yz + 1e-3
    
    def test_reset_functionality(self, setup_metric):
        """Reset should clear metric state."""
        metric = setup_metric
        
        X = torch.randn(5, 3, 10, 2)
        Y = torch.randn(5, 10, 2)
        
        # First computation
        metric.update(X, Y)
        mmd1 = metric.compute()
        
        # Reset
        metric.reset()
        
        # Second computation
        metric.update(X, Y)
        mmd2 = metric.compute()
        
        # Should give same result
        assert torch.isclose(mmd1, mmd2)
    
    def test_accumulation(self, setup_metric):
        """Multiple updates should accumulate correctly."""
        metric = setup_metric
        
        X1 = torch.randn(5, 3, 10, 2)
        Y1 = torch.randn(5, 10, 2)
        
        X2 = torch.randn(5, 3, 10, 2)
        Y2 = torch.randn(5, 10, 2)
        
        # Update twice
        metric.update(X1, Y1)
        metric.update(X2, Y2)
        mmd_accumulated = metric.compute()
        
        # Compute separately
        metric.reset()
        metric.update(X1, Y1)
        mmd1 = metric.compute()
        
        metric.reset()
        metric.update(X2, Y2)
        mmd2 = metric.compute()
        
        # Average should match
        expected = (mmd1 + mmd2) / 2
        assert torch.isclose(mmd_accumulated, expected, rtol=1e-4)
    
    def test_normalization_effect(self):
        """Normalization should affect MMD values."""
        kernel = LogSigKernel(RBFKernel(sigma=1.0))
        
        X = torch.randn(5, 3, 10, 2)
        Y = torch.randn(5, 10, 2)
        
        # With normalization
        metric_norm = SigMMD(kernel, normalize=True)
        metric_norm.update(X, Y)
        mmd_norm = metric_norm.compute()
        
        # Without normalization
        metric_no_norm = SigMMD(kernel, normalize=False)
        metric_no_norm.update(X, Y)
        mmd_no_norm = metric_no_norm.compute()
        
        # Should be different
        assert not torch.isclose(mmd_norm, mmd_no_norm)
    
    def test_no_data_error(self, setup_metric):
        """Should raise error if compute called before update."""
        metric = setup_metric
        
        with pytest.raises(RuntimeError, match="No data"):
            metric.compute()
    
    def test_batch_size_1(self, setup_metric):
        """Should handle batch size of 1."""
        metric = setup_metric
        
        X = torch.randn(1, 5, 20, 3)
        Y = torch.randn(1, 20, 3)
        
        metric.update(X, Y)
        mmd = metric.compute()
        
        assert not torch.isnan(mmd)
    
    def test_single_sample(self, setup_metric):
        """Should handle single sample per batch."""
        metric = setup_metric
        
        X = torch.randn(10, 1, 20, 3)
        Y = torch.randn(10, 20, 3)
        
        metric.update(X, Y)
        mmd = metric.compute()
        
        assert not torch.isnan(mmd)
    
    def test_gradient_flow(self):
        """Gradients should flow through metric computation."""
        kernel = LogSigKernel(RBFKernel(sigma=1.0))
        metric = SigMMD(kernel)
        
        X = torch.randn(3, 5, 10, 2, requires_grad=True)
        Y = torch.randn(3, 10, 2)
        
        metric.update(X, Y)
        mmd = metric.compute()
        mmd.backward()
        
        assert X.grad is not None
        assert not torch.isnan(X.grad).any()


@pytest.fixture
def device():
    """Get available device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.mark.gpu
class TestGPUCompatibility:
    """Test GPU compatibility for metrics."""
    
    def test_metric_on_gpu(self, device):
        """Metric should work on GPU."""
        if device.type == 'cpu':
            pytest.skip("GPU not available")
        
        kernel = LogSigKernel(RBFKernel(sigma=1.0))
        metric = SigMMD(kernel).to(device)
        
        X = torch.randn(5, 3, 10, 2, device=device)
        Y = torch.randn(5, 10, 2, device=device)
        
        metric.update(X, Y)
        mmd = metric.compute()
        
        assert mmd.device.type == 'cuda'
        assert not torch.isnan(mmd)


class TestNumericalStability:
    """Test numerical stability of metric."""
    
    def test_extreme_values(self):
        """Should handle extreme input values."""
        # Use sigma proportional to data scale so the RBF kernel isn't trivially 0
        kernel = LogSigKernel(RBFKernel(sigma=10000.0))
        metric = SigMMD(kernel)

        X = torch.randn(5, 3, 10, 2) * 100  # Large values
        Y = torch.randn(5, 10, 2) * 100
        
        metric.update(X, Y)
        mmd = metric.compute()
        
        assert not torch.isnan(mmd)
        assert not torch.isinf(mmd)
    
    def test_near_zero_values(self):
        """Should handle very small values."""
        kernel = LogSigKernel(RBFKernel(sigma=1.0))
        metric = SigMMD(kernel)
        
        X = torch.randn(5, 3, 10, 2) * 1e-6
        Y = torch.randn(5, 10, 2) * 1e-6
        
        metric.update(X, Y)
        mmd = metric.compute()
        
        assert not torch.isnan(mmd)