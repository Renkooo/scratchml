from scratchml.activations import tanh
from numpy.testing import assert_equal, assert_almost_equal
from ..utils import repeat
import unittest
import torch
import numpy as np


class Test_TanH(unittest.TestCase):
    """
    Unittest class created to test the TanH activation function.
    """

    @repeat(10)
    def test1(self):
        """
        Test the TanH function on random values and then compares it
        with the PyTorch implementation.
        """
        X = np.random.rand(10000, 2000)

        s = tanh(X)
        s_pytorch = torch.tanh(torch.from_numpy(X)).numpy()

        assert_almost_equal(s_pytorch, s)
        assert_equal(type(s_pytorch), type(s))
        assert_equal(s_pytorch.shape, s.shape)

    @repeat(10)
    def test2(self):
        """
        Test the TanH derivative on random values and then compares it
        with the PyTorch implementation.
        """
        X = torch.randn(1, requires_grad=True)

        s = tanh(X.detach().numpy(), derivative=True)
        torch.tanh(X).backward()

        assert_almost_equal(X.grad, s)
        assert_equal(X.grad.shape, s.shape)

    def test3(self):
        """
        Test the TanH derivative with a zero value and then compares it
        with the PyTorch implementation.
        """
        X = torch.tensor(0.0, requires_grad=True)

        s = tanh(X.detach().numpy(), derivative=True)
        torch.tanh(X).backward()

        assert_almost_equal(X.grad, s)
        assert_equal(X.grad.shape, s.shape)

    @repeat(5)
    def test_negative_values(self):
        """
        Test the TanH function with negative input values.
        """
        X = -np.random.rand(100, 100)  # Generate negative random values
        
        s = tanh(X)
        s_pytorch = torch.tanh(torch.from_numpy(X)).numpy()
        
        assert_almost_equal(s_pytorch, s)
        assert_equal(type(s_pytorch), type(s))
        assert_equal(s_pytorch.shape, s.shape)
        # Test derivative with negative values
        X_tensor = torch.tensor(-1.0, requires_grad=True)
        s_derivative = tanh(X_tensor.detach().numpy(), derivative=True)
        torch.tanh(X_tensor).backward()
        assert_almost_equal(X_tensor.grad, s_derivative)

    @repeat(5)
    def test_large_values(self):
        """
        Test the TanH function with large input values for numerical stability.
        """
        X = np.array([100., -100.])  # Large positive and negative values
        
        s = tanh(X)
        s_pytorch = torch.tanh(torch.from_numpy(X)).numpy()
        
        assert_almost_equal(s_pytorch, s)
        # Verify outputs are bounded between -1 and 1
        assert np.all(s >= -1.0) and np.all(s <= 1.0)
        # Large positive values should approach 1
        assert_almost_equal(s[0], 1.0, decimal=5)
        # Large negative values should approach -1
        assert_almost_equal(s[1], -1.0, decimal=5)

    @repeat(5)
    def test_different_shapes(self):
        """
        Test the TanH function with different input shapes.
        """
        # Test 1D array
        X1 = np.random.randn(100)
        s1 = tanh(X1)
        s1_pytorch = torch.tanh(torch.from_numpy(X1)).numpy()
        assert_almost_equal(s1_pytorch, s1)
        
        # Test 3D array
        X2 = np.random.randn(10, 10, 10)
        s2 = tanh(X2)
        s2_pytorch = torch.tanh(torch.from_numpy(X2)).numpy()
        assert_almost_equal(s2_pytorch, s2)
        
        # Test scalar input
        X3 = np.array(0.5)
        s3 = tanh(X3)
        s3_pytorch = torch.tanh(torch.tensor(X3)).numpy()
        assert_almost_equal(s3_pytorch, s3)

    @repeat(5)
    def test_tanh_properties(self):
        """
        Test mathematical properties of TanH function.
        """
        X = np.linspace(-5, 5, 100)
        s = tanh(X)
        
        # Test boundedness
        assert np.all(s >= -1.0) and np.all(s <= 1.0)
        
        # Test antisymmetric property: tanh(-x) = -tanh(x)
        assert_almost_equal(tanh(-X), -tanh(X))
        
        # Test derivative property: tanh'(x) = 1 - tanh²(x)
        s_prime = tanh(X, derivative=True)
        assert_almost_equal(s_prime, 1 - s**2)
        
        # Test tanh(0) = 0
        assert_almost_equal(tanh(np.array([0])), np.array([0]))

if __name__ == "__main__":
    unittest.main(verbosity=2)
