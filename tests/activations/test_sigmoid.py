from scratchml.activations import sigmoid
from numpy.testing import assert_equal, assert_almost_equal
from ..utils import repeat
import unittest
import torch
import numpy as np


class Test_Sigmoid(unittest.TestCase):
    """
    Unittest class created to test the Sigmoid activation function.
    """

    @repeat(10)
    def test1(self):
        """
        Test the Sigmoid function on random values and then compares it
        with the PyTorch implementation.
        """
        X = np.random.rand(10000, 2000)

        s = sigmoid(X)
        s_pytorch = torch.sigmoid(torch.from_numpy(X)).numpy()

        assert_almost_equal(s_pytorch, s)
        assert_equal(type(s_pytorch), type(s))
        assert_equal(s_pytorch.shape, s.shape)

    @repeat(10)
    def test2(self):
        """
        Test the Sigmoid derivative on random values and then compares it
        with the PyTorch implementation.
        """
        X = torch.randn(1, requires_grad=True)

        s = sigmoid(X.detach().numpy(), derivative=True)
        torch.sigmoid(X).backward()

        assert_almost_equal(X.grad, s)
        assert_equal(X.grad.shape, s.shape)
    @repeat(5)
    def test3(self):
        """
        Test the Sigmoid derivative with a zero value and then compares it
        with the PyTorch implementation.
        """
        X = torch.tensor(0.0, requires_grad=True)

        s = sigmoid(X.detach().numpy(), derivative=True)
        torch.sigmoid(X).backward()

        assert_almost_equal(X.grad, s)
        assert_equal(X.grad.shape, s.shape)

    @repeat(5)
    def test_negative_values(self):
        """
        Test the Sigmoid function with negative input values.
        """
        X = -np.random.rand(100, 100)  # Generate negative random values
        
        s = sigmoid(X)
        s_pytorch = torch.sigmoid(torch.from_numpy(X)).numpy()
        
        assert_almost_equal(s_pytorch, s)
        assert_equal(type(s_pytorch), type(s))
        assert_equal(s_pytorch.shape, s.shape)
        # Check if outputs are between 0 and 1
        assert np.all((s >= 0) & (s <= 1))
    @repeat(5)
    def test_large_values(self):
        """
        Test the Sigmoid function with large input values for numerical stability.
        """
        X = np.array([1000., -1000.])  # Large positive and negative values
        
        s = sigmoid(X)
        s_pytorch = torch.sigmoid(torch.from_numpy(X)).numpy()
        
        assert_almost_equal(s_pytorch, s)
        # Check if large positive values approach 1
        assert_almost_equal(s[0], 1.0, decimal=6)
        # Check if large negative values approach 0
        assert_almost_equal(s[1], 0.0, decimal=6)

    @repeat(5)
    def test_different_shapes(self):
        """
        Test the Sigmoid function with different input shapes.
        """
        # Test with 1D array
        X1 = np.random.rand(100)
        s1 = sigmoid(X1)
        s1_pytorch = torch.sigmoid(torch.from_numpy(X1)).numpy()
        assert_almost_equal(s1_pytorch, s1)
        
        # Test with 3D array
        X2 = np.random.rand(10, 10, 10)
        s2 = sigmoid(X2)
        s2_pytorch = torch.sigmoid(torch.from_numpy(X2)).numpy()
        assert_almost_equal(s2_pytorch, s2)
        
        # Test with scalar input
        X3 = np.array(0.5)
        s3 = sigmoid(X3)
        s3_pytorch = torch.sigmoid(torch.tensor(X3)).numpy()
        assert_almost_equal(s3_pytorch, s3)
    
    @repeat(5)
    def test_numerical_stability(self):
        """
        Test the Sigmoid function for numerical stability with extreme values.
        """
        # Test with very large values
        X_large = np.array([1e10, -1e10])
        s_large = sigmoid(X_large)
        assert_almost_equal(s_large[0], 1.0, decimal=6)
        assert_almost_equal(s_large[1], 0.0, decimal=6)
        
        # Test with very small values
        X_small = np.array([1e-10, -1e-10])
        s_small = sigmoid(X_small)
        s_small_pytorch = torch.sigmoid(torch.tensor(X_small)).numpy()
        assert_almost_equal(s_small_pytorch, s_small)


if __name__ == "__main__":
    unittest.main(verbosity=2)
