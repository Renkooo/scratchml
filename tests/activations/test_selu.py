from scratchml.activations import selu
from numpy.testing import assert_equal, assert_almost_equal
from ..utils import repeat
import unittest
import torch
import numpy as np


class Test_SELU(unittest.TestCase):
    """
    Unittest class created to test the SELU activation function.
    """

    @repeat(10)
    def test1(self):
        """
        Test the SELU function on random values and then compares it
        with the PyTorch implementation.
        """
        X = np.random.rand(10000, 2000)

        s = selu(X)
        s_pytorch = torch.nn.functional.selu(torch.from_numpy(X)).numpy()

        assert_almost_equal(s_pytorch, s)
        assert_equal(type(s_pytorch), type(s))
        assert_equal(s_pytorch.shape, s.shape)

    @repeat(10)
    def test2(self):
        """
        Test the SELU derivative on random values and then compares it
        with the PyTorch implementation.
        """
        X = torch.randn(1, requires_grad=True)

        s = selu(X.detach().numpy(), derivative=True)
        torch.nn.functional.selu(X).backward()

        assert_almost_equal(X.grad, s, decimal=5)
        assert_equal(X.grad.shape, s.shape)

    @repeat(5)
    def test3(self):
        """
        Test the SELU derivative with a zero value and then compares it
        with the PyTorch implementation.
        """
        X = torch.tensor(0.0, requires_grad=True)

        s = selu(X.detach().numpy(), derivative=True)
        torch.nn.functional.selu(X).backward()

        assert_almost_equal(X.grad, s)
        assert_equal(X.grad.shape, s.shape)

    @repeat(5)
    def test_negative_values(self):
        """
        Test the SELU function with negative input values.
        """
        X = -np.random.rand(100, 100)  # Generate negative random values
        
        s = selu(X)
        s_pytorch = torch.nn.functional.selu(torch.from_numpy(X)).numpy()
        
        assert_almost_equal(s_pytorch, s)
        assert_equal(type(s_pytorch), type(s))
        assert_equal(s_pytorch.shape, s.shape)

        # Test derivative
        X_tensor = torch.tensor(-1.0, requires_grad=True)
        s_derivative = selu(X_tensor.detach().numpy(), derivative=True)
        torch.nn.functional.selu(X_tensor).backward()
        
        assert_almost_equal(X_tensor.grad, s_derivative)
        assert_equal(X_tensor.grad.shape, s_derivative.shape)

    @repeat(5)
    def test_large_values(self):
        """
        Test the SELU function with large input values for numerical stability.
        """
        X = np.array([1000., -1000.])  # Large positive and negative values
        
        s = selu(X)
        s_pytorch = torch.nn.functional.selu(torch.from_numpy(X)).numpy()
        
        assert_almost_equal(s_pytorch, s)
        assert_equal(type(s_pytorch), type(s))
        assert_equal(s_pytorch.shape, s.shape)

        # Test derivative
        X_tensor = torch.tensor([1000.], requires_grad=True)
        s_derivative = selu(X_tensor.detach().numpy(), derivative=True)
        torch.nn.functional.selu(X_tensor).backward()
        
        assert_almost_equal(X_tensor.grad, s_derivative)
        assert_equal(X_tensor.grad.shape, s_derivative.shape)

    @repeat(5) 
    def test_different_shapes(self):
        """
        Test the SELU function with different input shapes.
        """
        # Test 1D array
        X1 = np.random.rand(100)
        s1 = selu(X1)
        s1_pytorch = torch.nn.functional.selu(torch.from_numpy(X1)).numpy()
        assert_almost_equal(s1_pytorch, s1)
        
        # Test 3D array
        X2 = np.random.rand(10, 10, 10)
        s2 = selu(X2)
        s2_pytorch = torch.nn.functional.selu(torch.from_numpy(X2)).numpy()
        assert_almost_equal(s2_pytorch, s2)
        
        # Test scalar input
        X3 = np.array(0.5)
        s3 = selu(X3)
        s3_pytorch = torch.nn.functional.selu(torch.tensor(X3)).numpy()
        assert_almost_equal(s3_pytorch, s3)

if __name__ == "__main__":
    unittest.main(verbosity=2)
