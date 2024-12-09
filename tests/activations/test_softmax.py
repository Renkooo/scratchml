from scratchml.activations import softmax
from numpy.testing import assert_equal, assert_almost_equal
from ..utils import repeat
import unittest
import torch
import numpy as np


class Test_Softmax(unittest.TestCase):
    """
    Unittest class created to test the Softmax activation function.
    """

    @repeat(10)
    def test1(self):
        """
        Test the Softmax function on random values and then compares it
        with the PyTorch implementation.
        """
        X = np.random.rand(10000, 2000)

        s = softmax(X)
        s_pytorch = torch.nn.functional.softmax(torch.from_numpy(X), dim=-1).numpy()

        assert_almost_equal(s_pytorch, s)
        assert_equal(type(s_pytorch), type(s))
        assert_equal(s_pytorch.shape, s.shape)

    @repeat(10)
    def test2(self):
        """
        Test the Softmax derivative on random values and then compares it
        with the PyTorch implementation.
        """
        X = torch.randn(1, requires_grad=True)

        s = softmax(X.detach().numpy(), derivative=True)
        torch.nn.functional.softmax(X, dim=-1).backward()

        assert_almost_equal(X.grad, s)
        assert_equal(X.grad.shape, s.shape)

    @repeat(5)
    def test3(self):
        """
        Test the Softmax derivative with a zero value and then compares it
        with the PyTorch implementation.
        """
        X = torch.tensor(0.0, requires_grad=True)

        s = softmax(X.detach().numpy(), derivative=True)
        torch.nn.functional.softmax(X, dim=-1).backward()

        assert_almost_equal(X.grad, s)
        assert_equal(X.grad.shape, s.shape)

    @repeat(5)
    def test_probability_properties(self):
        """
        Test that softmax output satisfies probability distribution properties.
        """
        X = np.random.randn(100, 10)
        
        s = softmax(X)
        s_pytorch = torch.nn.functional.softmax(torch.from_numpy(X), dim=-1).numpy()
        
        # Test sum to 1
        assert_almost_equal(np.sum(s, axis=1), np.ones(100))
        # Test values between 0 and 1
        assert np.all((s >= 0) & (s <= 1))
        # Compare with pytorch
        assert_almost_equal(s_pytorch, s)

    @repeat(5)
    def test_large_values(self):
        """
        Test numerical stability with large input values.
        """
        # Test with large positive and negative values
        X = np.array([[1000., -1000.], [10000., 10000.]])
        
        s = softmax(X)
        s_pytorch = torch.nn.functional.softmax(torch.from_numpy(X), dim=-1).numpy()
        
        assert_almost_equal(s_pytorch, s)
        # Check probability properties are maintained
        assert_almost_equal(np.sum(s, axis=1), np.ones(2))
        assert np.all((s >= 0) & (s <= 1))
        
        # For equal large values, outputs should be equal
        assert_almost_equal(s[1, 0], s[1, 1])

    @repeat(5)
    def test_different_dimensions(self):
        """
        Test softmax with different input dimensions.
        """
        # Test 1D input
        X1 = np.random.randn(5)
        s1 = softmax(X1)
        s1_pytorch = torch.nn.functional.softmax(torch.from_numpy(X1), dim=-1).numpy()
        assert_almost_equal(s1_pytorch, s1)
        assert_almost_equal(np.sum(s1), 1.0)
        
        # Test 3D input
        X2 = np.random.randn(3, 4, 5)
        s2 = softmax(X2)
        s2_pytorch = torch.nn.functional.softmax(torch.from_numpy(X2), dim=-1).numpy()
        assert_almost_equal(s2_pytorch, s2)
        assert_almost_equal(np.sum(s2, axis=-1), np.ones((3, 4)))

    @repeat(5)
    def test_edge_cases(self):
        """
        Test softmax with edge cases.
        """
        # Test all zeros
        X1 = np.zeros((3, 4))
        s1 = softmax(X1)
        s1_pytorch = torch.nn.functional.softmax(torch.from_numpy(X1), dim=-1).numpy()
        assert_almost_equal(s1_pytorch, s1)
        assert_almost_equal(s1, np.ones_like(X1) / X1.shape[1])
        
        # Test all same values
        X2 = np.full((2, 5), 10.0)
        s2 = softmax(X2)
        s2_pytorch = torch.nn.functional.softmax(torch.from_numpy(X2), dim=-1).numpy()
        assert_almost_equal(s2_pytorch, s2)
        assert_almost_equal(s2, np.ones_like(X2) / X2.shape[1])
if __name__ == "__main__":
    unittest.main(verbosity=2)
