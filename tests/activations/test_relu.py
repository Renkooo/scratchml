from scratchml.activations import relu
from numpy.testing import assert_equal, assert_almost_equal
from ..utils import repeat
import unittest
import torch
import numpy as np


class Test_RELU(unittest.TestCase):
    """
    Unittest class created to test the RELU activation function.
    """

    @repeat(10)
    def test1(self):
        """
        Test the RELU function on random values and then compares it
        with the PyTorch implementation.
        """
        X = np.random.rand(10000, 2000)

        s = relu(X)
        s_pytorch = torch.relu(torch.from_numpy(X)).numpy()

        assert_almost_equal(s_pytorch, s)
        assert_equal(type(s_pytorch), type(s))
        assert_equal(s_pytorch.shape, s.shape)

    @repeat(10)
    def test2(self):
        """
        Test the RELU derivative on random values and then compares it
        with the PyTorch implementation.
        """
        X = torch.randn(1, requires_grad=True)

        s = relu(X.detach().numpy(), derivative=True)
        torch.relu(X).backward()

        assert_almost_equal(X.grad, s)
        assert_equal(X.grad.shape, s.shape)

    def test3(self):
        """
        Test the RELU derivative with a zero value and then compares it
        with the PyTorch implementation.
        """
        X = torch.tensor(0.0, requires_grad=True)

        s = relu(X.detach().numpy(), derivative=True)
        torch.relu(X).backward()

        assert_almost_equal(X.grad, s)
        assert_equal(X.grad.shape, s.shape)

    @repeat(5)
    def test_negative_values(self):
        """
        Test the RELU function with negative input values.
        """
        X = -np.random.rand(100, 100)  # Generate negative random values
        
        s = relu(X)
        s_pytorch = torch.relu(torch.from_numpy(X)).numpy()
        
        assert_almost_equal(s_pytorch, s)
        assert_equal(type(s_pytorch), type(s))
        assert_equal(s_pytorch.shape, s.shape)
        # Verify all outputs are >= 0 (ReLU property)
        assert np.all(s >= 0)

    @repeat(5)
    def test_large_values(self):
        """
        Test the RELU function with large input values for numerical stability.
        """
        X = np.array([1e6, -1e6])  # Large positive and negative values
        
        s = relu(X)
        s_pytorch = torch.relu(torch.from_numpy(X)).numpy()
        
        assert_almost_equal(s_pytorch, s)
        assert_equal(type(s_pytorch), type(s))
        assert_equal(s_pytorch.shape, s.shape)
        # Verify negative values are zeroed out
        assert s[1] == 0
        # Verify large positive values pass through unchanged
        assert s[0] == 1e6

    @repeat(5)
    def test_different_shapes(self):
        """
        Test the RELU function with different input shapes.
        """
        # Test with 1D array
        X1 = np.random.randn(100)
        s1 = relu(X1)
        s1_pytorch = torch.relu(torch.from_numpy(X1)).numpy()
        assert_almost_equal(s1_pytorch, s1)
        
        # Test with 3D array
        X2 = np.random.randn(10, 10, 10)
        s2 = relu(X2)
        s2_pytorch = torch.relu(torch.from_numpy(X2)).numpy()
        assert_almost_equal(s2_pytorch, s2)
        
        # Test with scalar input
        X3 = np.array(-0.5)
        s3 = relu(X3)
        s3_pytorch = torch.relu(torch.tensor(X3)).numpy()
        assert_almost_equal(s3_pytorch, s3)
        assert s3 == 0  # Verify negative scalar is zeroed out

if __name__ == "__main__":
    unittest.main(verbosity=2)
