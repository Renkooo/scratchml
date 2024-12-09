from scratchml.activations import leaky_relu
from numpy.testing import assert_equal, assert_almost_equal
from ..utils import repeat
import unittest
import torch
import numpy as np


class Test_Leaky_RELU(unittest.TestCase):
    """
    Unittest class created to test the Leaky RELU activation function.
    """

    @repeat(10)
    def test1(self):
        """
        Test the Leaky RELU function on random values and then compares it
        with the PyTorch implementation.
        """
        X = np.random.rand(10000, 2000)

        s = leaky_relu(X)
        s_pytorch = torch.nn.functional.leaky_relu(
            torch.from_numpy(X),
            negative_slope=0.001,
        ).numpy()

        assert_almost_equal(s_pytorch, s)
        assert_equal(type(s_pytorch), type(s))
        assert_equal(s_pytorch.shape, s.shape)

    @repeat(10)
    def test2(self):
        """
        Test the Leaky RELU derivative on random values and then compares it
        with the PyTorch implementation.
        """
        X = torch.randn(1, requires_grad=True)

        s = leaky_relu(X.detach().numpy(), derivative=True)
        torch.nn.functional.leaky_relu(X, negative_slope=0.001).backward()

        assert_almost_equal(X.grad, s)
        assert_equal(X.grad.shape, s.shape)

    def test3(self):
        """
        Test the Leaky RELU derivative with a zero value and then compares it
        with the PyTorch implementation.
        """
        X = torch.tensor(0.0, requires_grad=True)

        s = leaky_relu(X.detach().numpy(), derivative=True)
        torch.nn.functional.leaky_relu(X, negative_slope=0.001).backward()

        assert_almost_equal(X.grad, s)
        assert_equal(X.grad.shape, s.shape)

    @repeat(5)
    def test_negative_values(self):
        """
        Test the Leaky RELU function with negative input values.
        """
        X = -np.random.rand(100, 100)  # Generate negative random values
        
        s = leaky_relu(X)
        s_pytorch = torch.nn.functional.leaky_relu(
            torch.from_numpy(X),
            negative_slope=0.001
        ).numpy()
        
        assert_almost_equal(s_pytorch, s)
        assert_equal(type(s_pytorch), type(s))
        assert_equal(s_pytorch.shape, s.shape)
        # Verify outputs are scaled by 0.001 for negative values
        assert np.all(s[X < 0] == 0.001 * X[X < 0])

    @repeat(5)
    def test_large_values(self):
        """
        Test the Leaky RELU function with large input values for numerical stability.
        """
        X = np.array([1e6, -1e6])  # Large positive and negative values
        
        s = leaky_relu(X)
        s_pytorch = torch.nn.functional.leaky_relu(
            torch.from_numpy(X),
            negative_slope=0.001
        ).numpy()
        
        assert_almost_equal(s_pytorch, s)
        # Verify large positive values pass through unchanged
        assert s[0] == 1e6
        # Verify large negative values are scaled correctly
        assert s[1] == -1e3

    @repeat(5)
    def test_different_shapes(self):
        """
        Test the Leaky RELU function with different input shapes.
        """
        # Test with 1D array 
        X1 = np.random.randn(100)
        s1 = leaky_relu(X1)
        s1_pytorch = torch.nn.functional.leaky_relu(
            torch.from_numpy(X1),
            negative_slope=0.001
        ).numpy()
        assert_almost_equal(s1_pytorch, s1)
        
        # Test with 3D array
        X2 = np.random.randn(10, 10, 10)
        s2 = leaky_relu(X2)
        s2_pytorch = torch.nn.functional.leaky_relu(
            torch.from_numpy(X2),
            negative_slope=0.001
        ).numpy()
        assert_almost_equal(s2_pytorch, s2)
        
        # Test with scalar input
        X3 = np.array(-0.5)
        s3 = leaky_relu(X3)
        s3_pytorch = torch.nn.functional.leaky_relu(
            torch.tensor(X3),
            negative_slope=0.001
        ).numpy()
        assert_almost_equal(s3_pytorch, s3)
        assert s3 == -0.0005  # Verify negative scalar is scaled correctly

if __name__ == "__main__":
    unittest.main(verbosity=2)
