from scratchml.activations import softplus
from numpy.testing import assert_equal, assert_almost_equal
from ..utils import repeat
import unittest
import torch
import numpy as np


class Test_SoftPlus(unittest.TestCase):
    """
    Unittest class created to test the SoftPlus activation function.
    """

    @repeat(10)
    def test1(self):
        """
        Test the SoftPlus function on random values and then compares it
        with the PyTorch implementation.
        """
        X = np.random.rand(10000, 2000)

        s = softplus(X)
        s_pytorch = torch.nn.functional.softplus(torch.from_numpy(X)).numpy()

        assert_almost_equal(s_pytorch, s)
        assert_equal(type(s_pytorch), type(s))
        assert_equal(s_pytorch.shape, s.shape)

    @repeat(10)
    def test2(self):
        """
        Test the SoftPlus derivative on random values and then compares it
        with the PyTorch implementation.
        """
        X = torch.randn(1, requires_grad=True)

        s = softplus(X.detach().numpy(), derivative=True)
        torch.nn.functional.softplus(X).backward()

        assert_almost_equal(X.grad, s)
        assert_equal(X.grad.shape, s.shape)

    def test3(self):
        """
        Test the SoftPlus derivative with a zero value and then compares it
        with the PyTorch implementation.
        """
        X = torch.tensor(0.0, requires_grad=True)

        s = softplus(X.detach().numpy(), derivative=True)
        torch.nn.functional.softplus(X).backward()

        assert_almost_equal(X.grad, s)
        assert_equal(X.grad.shape, s.shape)

    @repeat(5)
    def test_negative_values(self):
        """
        Test the SoftPlus function with negative input values.
        """
        X = -np.random.rand(100, 100)  # Generate negative random values
        
        s = softplus(X)
        s_pytorch = torch.nn.functional.softplus(torch.from_numpy(X)).numpy()
        
        assert_almost_equal(s_pytorch, s)
        assert_equal(type(s_pytorch), type(s))
        assert_equal(s_pytorch.shape, s.shape)
        # Verify outputs are always positive
        assert np.all(s >= 0)

    @repeat(5)
    def test_large_values(self):
        """
        Test the SoftPlus function with large input values for numerical stability.
        """
        X = np.array([100., -100.])  # Large positive and negative values
        
        s = softplus(X)
        s_pytorch = torch.nn.functional.softplus(torch.from_numpy(X)).numpy()
        
        assert_almost_equal(s_pytorch, s)
        # For large positive x, softplus(x) ≈ x
        assert_almost_equal(s[0], X[0], decimal=4)
        # For large negative x, softplus(x) ≈ 0
        assert s[1] < 1.0

    @repeat(5)
    def test_different_shapes(self):
        """
        Test the SoftPlus function with different input shapes.
        """
        # Test 1D array
        X1 = np.random.randn(100)
        s1 = softplus(X1)
        s1_pytorch = torch.nn.functional.softplus(torch.from_numpy(X1)).numpy()
        assert_almost_equal(s1_pytorch, s1)
        
        # Test 3D array
        X2 = np.random.randn(10, 10, 10)
        s2 = softplus(X2)
        s2_pytorch = torch.nn.functional.softplus(torch.from_numpy(X2)).numpy()
        assert_almost_equal(s2_pytorch, s2)
        
        # Test scalar input
        X3 = np.array(0.5)
        s3 = softplus(X3)
        s3_pytorch = torch.nn.functional.softplus(torch.tensor(X3)).numpy()
        assert_almost_equal(s3_pytorch, s3)

    @repeat(5)
    def test_softplus_properties(self):
        """
        Test key mathematical properties of SoftPlus function.
        """
        X = np.linspace(-5, 5, 100)
        s = softplus(X)
        
        # Test positivity
        assert np.all(s > 0)
        
        # Test monotonicity (should be strictly increasing)
        assert np.all(np.diff(s) > 0)
        
        # Test asymptotic behavior
        # For x << 0, softplus(x) ≈ 0
        assert_almost_equal(softplus(np.array([-100])), np.array([0]), decimal=4)
        # For x >> 0, softplus(x) ≈ x
        large_x = 100
        assert_almost_equal(softplus(np.array([large_x])), np.array([large_x]), decimal=4)

if __name__ == "__main__":
    unittest.main(verbosity=2)
