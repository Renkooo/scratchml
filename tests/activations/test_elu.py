from scratchml.activations import elu
from numpy.testing import assert_equal, assert_almost_equal
from ..utils import repeat
import unittest
import torch
import numpy as np


class Test_ELU(unittest.TestCase):
    """
    Unittest class created to test the ELU activation function.
    """

    @repeat(10)
    def test1(self):
        """
        Test the ELU function on random values and then compares it
        with the PyTorch implementation.
        """
        X = np.random.rand(10000, 2000)

        s = elu(X)
        s_pytorch = torch.nn.functional.elu(torch.from_numpy(X)).numpy()

        assert_almost_equal(s_pytorch, s)
        assert_equal(type(s_pytorch), type(s))
        assert_equal(s_pytorch.shape, s.shape)

    @repeat(10)
    def test2(self):
        """
        Test the ELU derivative on random values and then compares it
        with the PyTorch implementation.
        """
        X = torch.randn(1, requires_grad=True)

        s = elu(X.detach().numpy(), derivative=True)
        torch.nn.functional.elu(X).backward()

        assert_almost_equal(X.grad, s)
        assert_equal(X.grad.shape, s.shape)

    def test3(self):
        """
        Test the ELU derivative with a zero value and then compares it
        with the PyTorch implementation.
        """
        X = torch.tensor(0.0, requires_grad=True)

        s = elu(X.detach().numpy(), derivative=True)
        torch.nn.functional.elu(X).backward()

        assert_almost_equal(X.grad, s)
        assert_equal(X.grad.shape, s.shape)

    @repeat(5)
    def test_negative_values(self):
        """
        Test the ELU function with negative input values.
        """
        X = -np.random.rand(100, 100)  # Generate negative random values
        
        s = elu(X)
        s_pytorch = torch.nn.functional.elu(torch.from_numpy(X)).numpy()
        
        assert_almost_equal(s_pytorch, s)
        assert_equal(type(s_pytorch), type(s))
        assert_equal(s_pytorch.shape, s.shape)
        
        # Test derivative with negative values
        X_tensor = torch.tensor(-1.0, requires_grad=True)
        s_derivative = elu(X_tensor.detach().numpy(), derivative=True)
        torch.nn.functional.elu(X_tensor).backward()
        assert_almost_equal(X_tensor.grad, s_derivative)

    @repeat(5)
    def test_large_values(self):
        """
        Test the ELU function with large input values for numerical stability.
        """
        # Use more reasonable large values that won't cause overflow
        X = np.array([100., -50.])
        
        s = elu(X)
        s_pytorch = torch.nn.functional.elu(torch.from_numpy(X)).numpy()
        
        assert_almost_equal(s_pytorch, s)
        # Verify large positive values pass through unchanged
        assert_almost_equal(s[0], 100.)
        # Verify large negative values are bounded
        assert s[1] >= -1.0  # ELU approaches -1 asymptotically for negative values
        assert s[1] <= 0.0   # ELU should always be negative for negative inputs

        # Test very large values
        X_extreme = np.array([1000., -1000.])
        s_extreme = elu(X_extreme)
        # Positive values should pass through
        assert_almost_equal(s_extreme[0], 1000.)
        # Negative values should be bounded
        assert -1.0 <= s_extreme[1] <= 0.0

    @repeat(5)
    def test_different_shapes(self):
        """
        Test the ELU function with different input shapes.
        """
        # Test 1D array
        X1 = np.random.randn(100)
        s1 = elu(X1)
        s1_pytorch = torch.nn.functional.elu(torch.from_numpy(X1)).numpy()
        assert_almost_equal(s1_pytorch, s1)
        
        # Test 3D array
        X2 = np.random.randn(10, 10, 10)
        s2 = elu(X2)
        s2_pytorch = torch.nn.functional.elu(torch.from_numpy(X2)).numpy()
        assert_almost_equal(s2_pytorch, s2)
        
        # Test scalar input
        X3 = np.array(-0.5)
        s3 = elu(X3)
        s3_pytorch = torch.nn.functional.elu(torch.tensor(X3)).numpy()
        assert_almost_equal(s3_pytorch, s3)

    @repeat(5)
    def test_epsilon_parameter(self):
        """
        Test the ELU function with different epsilon values.
        """
        X = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        
        # Test with default epsilon
        s1 = elu(X)
        s1_pytorch = torch.nn.functional.elu(torch.from_numpy(X)).numpy()
        assert_almost_equal(s1_pytorch, s1)
        
        # Test with custom epsilon
        s2 = elu(X, epsilon=1e-5)
        assert_almost_equal(s1_pytorch, s2, decimal=4)  # Should be similar despite different epsilon
        
        # Test derivative with custom epsilon
        s3 = elu(X, derivative=True, epsilon=1e-5)
        assert np.all(s3[X > 0] == 1.0)  # Derivative should be 1 for positive values

if __name__ == "__main__":
    unittest.main(verbosity=2)
