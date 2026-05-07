"""
Tests for kernel implementations.
"""
import pytest
import torch
import math
from sigkernel_mmd.kernels import (
    RBFKernel,
    SigKernel,
    LogSigKernel,         # backward-compat alias
    _compute_M_inc,
    _solve_pde,
    solve_pde,
    _tile,
)


# ═══════════════════════════════════════════════════════════════════════════
# RBF static kernel
# ═══════════════════════════════════════════════════════════════════════════

class TestRBFKernel:
    """Test RBF kernel k(x,y) = exp(-||x-y||²/σ)."""

    def test_formula(self):
        """Verify the formula is exp(-d²/σ), NOT exp(-d²/(2σ²))."""
        sigma = 4.0
        kernel = RBFKernel(sigma=sigma)
        # Two single-timestep, 1-D "paths"
        X = torch.tensor([[[[1.0]]]]) # (1,1,1,1)
        Y = torch.tensor([[[[3.0]]]])
        K = kernel.batch_kernel(X, Y)
        d_sq = (1.0 - 3.0) ** 2  # = 4
        expected = math.exp(-d_sq / sigma)  # exp(-4/4) = exp(-1)
        assert torch.isclose(K.squeeze(), torch.tensor(expected), atol=1e-6)

    def test_self_similarity(self):
        """k(x, x) should equal 1."""
        kernel = RBFKernel(sigma=1.0)
        X = torch.randn(5, 3, 10, 4)
        K = kernel.batch_kernel(X, X)
        for i in range(5):
            for j in range(3):
                diag = torch.diagonal(K[i, j, i, j])
                assert torch.allclose(diag, torch.ones_like(diag), atol=1e-5)

    def test_symmetry(self):
        """k(x, y) should equal k(y, x)."""
        kernel = RBFKernel(sigma=1.0)
        X = torch.randn(3, 2, 10, 4)
        Y = torch.randn(3, 2, 10, 4)
        K_XY = kernel.batch_kernel(X, Y)
        K_YX = kernel.batch_kernel(Y, X)
        K_YX_T = K_YX.permute(2, 3, 0, 1, 5, 4)
        assert torch.allclose(K_XY, K_YX_T, atol=1e-5)

    def test_bandwidth_effect(self):
        """Larger sigma should give values closer to 1."""
        X = torch.randn(2, 1, 10, 3)
        Y = torch.randn(2, 1, 10, 3)
        K_small = RBFKernel(sigma=0.1).batch_kernel(X, Y)
        K_large = RBFKernel(sigma=100.0).batch_kernel(X, Y)
        assert K_large.mean() > K_small.mean()


class TestRBFKernelDiag:
    """Test block-diagonal RBF kernel."""

    def test_output_shape(self):
        kernel = RBFKernel(sigma=1.0)
        X = torch.randn(4, 3, 10, 5)
        Y = torch.randn(4, 2, 10, 5)
        K = kernel.batch_kernel_diag(X, Y)
        assert K.shape == (4, 3, 2, 10, 10)

    def test_consistency_with_batch_kernel(self):
        """Diagonal blocks of batch_kernel must equal batch_kernel_diag."""
        kernel = RBFKernel(sigma=1.0)
        B, Nx, Ny, L, D = 3, 4, 4, 8, 3
        X = torch.randn(B, Nx, L, D)
        Y = torch.randn(B, Ny, L, D)
        K_full = kernel.batch_kernel(X, Y)
        K_diag = kernel.batch_kernel_diag(X, Y)
        for b in range(B):
            assert torch.allclose(K_full[b, :, b, :, :, :], K_diag[b], atol=1e-5)

    def test_self_similarity(self):
        kernel = RBFKernel(sigma=1.0)
        X = torch.randn(3, 2, 8, 4)
        K = kernel.batch_kernel_diag(X, X)
        for b in range(3):
            for i in range(2):
                diag = torch.diagonal(K[b, i, i])
                assert torch.allclose(diag, torch.ones_like(diag), atol=1e-5)

    def test_symmetry(self):
        kernel = RBFKernel(sigma=1.0)
        X = torch.randn(3, 4, 8, 3)
        K = kernel.batch_kernel_diag(X, X)
        K_T = K.permute(0, 2, 1, 4, 3)
        assert torch.allclose(K, K_T, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════
# PDE solver
# ═══════════════════════════════════════════════════════════════════════════

class TestPDESolver:
    """Test the real-space Padé PDE solver."""

    def test_zero_increments_give_identity(self):
        """M_inc = 0 → K = 1 everywhere (PDE doesn't evolve)."""
        M = torch.zeros(5, 5)
        K = _solve_pde(M)
        assert K.shape == (6, 6)
        assert torch.allclose(K, torch.ones_like(K))

    def test_boundary_conditions(self):
        """K[0,:] = K[:,0] = 1."""
        M = torch.randn(8, 8)
        K = _solve_pde(M)
        assert torch.allclose(K[0, :], torch.ones(9))
        assert torch.allclose(K[:, 0], torch.ones(9))

    def test_positive_values_for_positive_increments(self):
        """PDE solution should be positive for small positive M_inc."""
        M = torch.rand(10, 10) * 0.1
        K = _solve_pde(M)
        assert (K > 0).all()

    def test_batch_dimensions(self):
        """Should handle arbitrary batch dimensions."""
        M = torch.randn(3, 4, 5, 5)
        K = _solve_pde(M)
        assert K.shape == (3, 4, 6, 6)

    def test_matches_naive_double_loop(self):
        """Anti-diagonal sweep must match naive i,j loop."""
        M = torch.randn(6, 6)
        L = 6
        # Naive double loop
        K_naive = torch.ones(L + 1, L + 1)
        for i in range(1, L + 1):
            for j in range(1, L + 1):
                dt = M[i - 1, j - 1]
                dt_sq = dt ** 2
                A = 1.0 + 0.5 * dt + (1.0 / 12.0) * dt_sq
                C = 1.0 - (1.0 / 12.0) * dt_sq
                K_naive[i, j] = (K_naive[i - 1, j] + K_naive[i, j - 1]) * A - K_naive[i - 1, j - 1] * C
        K_sweep = _solve_pde(M)
        assert torch.allclose(K_naive, K_sweep, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════
# Dyadic refinement (M_inc tiling)
# ═══════════════════════════════════════════════════════════════════════════

class TestDyadicRefinement:
    """Test that M_inc tiling matches sigkernel's approach."""

    def test_tile_shape(self):
        """Tiling should multiply dimensions by 2^n."""
        G = torch.randn(3, 4, 10, 10)  # batch static kernel
        M = _compute_M_inc(G, dyadic_order=0)
        assert M.shape == (3, 4, 10, 10)  # double-diff of padded G

        M1 = _compute_M_inc(G, dyadic_order=1)
        # Each original dim of M multiplied by 2
        assert M1.shape[-2] == M.shape[-2] * 2
        assert M1.shape[-1] == M.shape[-1] * 2

    def test_tile_preserves_sum(self):
        """Tiling + dividing should preserve total sum."""
        G = torch.randn(5, 5)
        M0 = _compute_M_inc(G, dyadic_order=0)
        M1 = _compute_M_inc(G, dyadic_order=1)
        assert torch.isclose(M0.sum(), M1.sum(), rtol=1e-5)

    def test_dyadic_order_0_is_identity(self):
        """Order 0 should not change M_inc."""
        G = torch.randn(8, 8)
        M = _compute_M_inc(G, dyadic_order=0)
        G_pad = torch.nn.functional.pad(G, (1, 0, 1, 0))
        M_expected = (G_pad[1:, 1:] + G_pad[:-1, :-1]
                      - G_pad[1:, :-1] - G_pad[:-1, 1:])
        assert torch.allclose(M, M_expected)


# ═══════════════════════════════════════════════════════════════════════════
# Signature kernel (full API)
# ═══════════════════════════════════════════════════════════════════════════

class TestSigKernel:
    """Test signature kernel public API."""

    def test_self_similarity_positive(self):
        """k(x, x) should be positive."""
        kernel = SigKernel(RBFKernel(sigma=1.0))
        X = torch.randn(5, 10, 3)
        k = kernel.compute_kernel(X, X)
        assert (k > 0).all()

    def test_symmetry(self):
        """k(x, y) = k(y, x)."""
        kernel = SigKernel(RBFKernel(sigma=1.0))
        X = torch.randn(5, 10, 3)
        Y = torch.randn(5, 10, 3)
        K_XY = kernel.compute_Gram(X, Y)
        K_YX = kernel.compute_Gram(Y, X)
        assert torch.allclose(K_XY, K_YX.T, atol=1e-4)

    def test_gradient_flow(self):
        """Gradients should flow through kernel computation."""
        kernel = SigKernel(RBFKernel(sigma=1.0))
        X = torch.randn(3, 10, 2, requires_grad=True)
        Y = torch.randn(3, 10, 2)
        K = kernel.compute_Gram(X, Y)
        K.sum().backward()
        assert X.grad is not None
        assert not torch.isnan(X.grad).any()

    def test_4d_batching(self):
        kernel = SigKernel(RBFKernel(sigma=1.0))
        X = torch.randn(3, 5, 10, 2)
        Y = torch.randn(3, 4, 10, 2)
        K = kernel.compute_Gram(X, Y)
        assert K.shape == (3, 5, 3, 4)

    def test_3d_compatibility(self):
        kernel = SigKernel(RBFKernel(sigma=1.0))
        X = torch.randn(5, 10, 3)
        Y = torch.randn(5, 10, 3)
        K = kernel.compute_Gram(X, Y)
        assert K.shape == (5, 5)

    def test_constant_path(self):
        """Signature of constant path should be well-defined (K=1)."""
        kernel = SigKernel(RBFKernel(sigma=1.0))
        X = torch.ones(2, 10, 3)
        k = kernel.compute_kernel(X, X)
        assert not torch.isnan(k).any()
        assert (k > 0).all()

    def test_backward_compat_alias(self):
        """LogSigKernel should still work as an alias."""
        kernel = LogSigKernel(RBFKernel(sigma=1.0))
        X = torch.randn(3, 10, 2)
        Y = torch.randn(3, 10, 2)
        K = kernel.compute_Gram(X, Y)
        assert K.shape == (3, 3)

    def test_dyadic_order_changes_value(self):
        """Higher dyadic order should give different (more accurate) values."""
        X = torch.randn(3, 10, 2)
        Y = torch.randn(3, 10, 2)
        K0 = SigKernel(RBFKernel(sigma=1.0), dyadic_order=0).compute_Gram(X, Y)
        K1 = SigKernel(RBFKernel(sigma=1.0), dyadic_order=1).compute_Gram(X, Y)
        # Values should differ (finer discretisation)
        assert not torch.allclose(K0, K1, atol=1e-3)
        # But both should be positive and finite
        assert (K0 > 0).all() and (K1 > 0).all()
        assert not torch.isnan(K0).any() and not torch.isnan(K1).any()


class TestSigKernelDiag:
    """Test block-diagonal signature kernel."""

    def test_output_shape(self):
        kernel = SigKernel(RBFKernel(sigma=1.0))
        X = torch.randn(3, 4, 10, 2)
        Y = torch.randn(3, 5, 10, 2)
        K = kernel.compute_diag_Gram(X, Y)
        assert K.shape == (3, 4, 5)

    def test_3d_squeeze(self):
        kernel = SigKernel(RBFKernel(sigma=1.0))
        X = torch.randn(4, 10, 2)
        Y = torch.randn(4, 10, 2)
        K = kernel.compute_diag_Gram(X, Y)
        assert K.shape == (4,)

    def test_consistency_with_compute_gram(self):
        """Diagonal blocks of compute_Gram should equal compute_diag_Gram."""
        kernel = SigKernel(RBFKernel(sigma=1.0))
        B, N, L, D = 3, 4, 8, 2
        X = torch.randn(B, N, L, D)
        Y = torch.randn(B, N, L, D)
        K_full = kernel.compute_Gram(X, Y)
        K_diag = kernel.compute_diag_Gram(X, Y)
        for b in range(B):
            assert torch.allclose(K_full[b, :, b, :], K_diag[b], atol=1e-4)

    def test_no_cross_batch_contamination(self):
        kernel = SigKernel(RBFKernel(sigma=1.0))
        B, N, L, D = 4, 3, 6, 2
        X = torch.randn(B, N, L, D)
        Y = torch.randn(B, N, L, D)
        K_ref = kernel.compute_diag_Gram(X, Y).detach().clone()
        X2 = X.clone()
        X2[0] = X2[0] + 100.0
        K_pert = kernel.compute_diag_Gram(X2, Y).detach()
        assert not torch.allclose(K_ref[0], K_pert[0])
        for b in range(1, B):
            assert torch.allclose(K_ref[b], K_pert[b], atol=1e-5)

    def test_gradient_flow(self):
        kernel = SigKernel(RBFKernel(sigma=1.0))
        X = torch.randn(3, 4, 8, 2, requires_grad=True)
        Y = torch.randn(3, 4, 8, 2)
        K = kernel.compute_diag_Gram(X, Y)
        K.sum().backward()
        assert X.grad is not None
        assert not torch.isnan(X.grad).any()

    def test_finite_values(self):
        """Kernel values should be finite (Padé scheme may produce negative values)."""
        kernel = SigKernel(RBFKernel(sigma=1.0))
        X = torch.randn(3, 4, 10, 2)
        Y = torch.randn(3, 4, 10, 2)
        K = kernel.compute_diag_Gram(X, Y)
        assert not torch.isnan(K).any()
        assert not torch.isinf(K).any()


# ═══════════════════════════════════════════════════════════════════════════
# Numerical stability
# ═══════════════════════════════════════════════════════════════════════════

class TestNumericalStability:

    def test_moderate_length_sequences(self):
        """Should handle moderate-length sequences without overflow."""
        kernel = SigKernel(RBFKernel(sigma=1.0))
        X = torch.randn(2, 50, 3)
        Y = torch.randn(2, 50, 3)
        K = kernel.compute_Gram(X, Y)
        assert not torch.isnan(K).any()
        assert not torch.isinf(K).any()

    def test_high_dimensional(self):
        kernel = SigKernel(RBFKernel(sigma=10.0))
        X = torch.randn(3, 20, 50)
        Y = torch.randn(3, 20, 50)
        K = kernel.compute_Gram(X, Y)
        assert not torch.isnan(K).any()

    def test_small_bandwidth(self):
        kernel = SigKernel(RBFKernel(sigma=0.01))
        X = torch.randn(2, 10, 3)
        Y = torch.randn(2, 10, 3)
        K = kernel.compute_Gram(X, Y)
        assert not torch.isnan(K).any()

    def test_large_bandwidth(self):
        kernel = SigKernel(RBFKernel(sigma=100.0))
        X = torch.randn(2, 10, 3)
        Y = torch.randn(2, 10, 3)
        K = kernel.compute_Gram(X, Y)
        assert not torch.isnan(K).any()


# ═══════════════════════════════════════════════════════════════════════════
# GPU compatibility
# ═══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TestGPUCompatibility:

    def test_rbf_kernel_gpu(self, device):
        if device.type == 'cpu':
            pytest.skip("GPU not available")
        kernel = RBFKernel(sigma=1.0)
        X = torch.randn(3, 2, 10, 4, device=device)
        Y = torch.randn(3, 2, 10, 4, device=device)
        K = kernel.batch_kernel(X, Y)
        assert K.device.type == 'cuda'
        assert not torch.isnan(K).any()

    def test_sig_kernel_gpu(self, device):
        if device.type == 'cpu':
            pytest.skip("GPU not available")
        kernel = SigKernel(RBFKernel(sigma=1.0))
        X = torch.randn(3, 10, 2, device=device)
        Y = torch.randn(3, 10, 2, device=device)
        K = kernel.compute_Gram(X, Y)
        assert K.device.type == 'cuda'
        assert not torch.isnan(K).any()

    def test_triton_matches_pytorch(self, device):
        """Triton and PyTorch PDE solvers should give the same result."""
        if device.type == 'cpu':
            pytest.skip("GPU not available")
        M = torch.randn(4, 10, 10, device=device)
        K_pt = _solve_pde(M)
        K_dispatch = solve_pde(M)  # will use Triton if available
        assert torch.allclose(K_pt, K_dispatch, atol=1e-4), \
            f"Max diff: {(K_pt - K_dispatch).abs().max():.6f}"

    def test_diag_gram_gpu(self, device):
        """Block-diagonal Gram should work end-to-end on GPU."""
        if device.type == 'cpu':
            pytest.skip("GPU not available")
        kernel = SigKernel(RBFKernel(sigma=1.0), dyadic_order=1)
        X = torch.randn(4, 3, 15, 2, device=device)
        Y = torch.randn(4, 2, 15, 2, device=device)
        K = kernel.compute_diag_Gram(X, Y)
        assert K.shape == (4, 3, 2)
        assert K.device.type == 'cuda'
        # Padé PDE can produce slightly negative values — just check finite
        assert not torch.isnan(K).any()
        assert not torch.isinf(K).any()
