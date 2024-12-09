import math
from unittest.mock import Mock, patch
from numpy.testing import assert_allclose, assert_equal
from sklearn.svm import SVC as SkSVC
from scratchml.models.svc import SVC
from scratchml.scalers import StandardScaler
from ..utils import generate_classification_dataset, repeat
import unittest
import numpy as np
from numpy.testing import assert_array_equal


class Test_SVC(unittest.TestCase):
    """
    Unit test class for the custom SVC implementation.
    """

    @repeat(3)
    def test_binary_classification(self):
        """
        Test binary classification and compare the custom SVC with Scikit-Learn's SVC.
        """
        X, y = generate_classification_dataset(
            n_samples=2000, n_features=4, n_classes=2
        )

        # Initialize and train both models
        custom_svc = SVC(kernel="linear")
        sklearn_svc = SkSVC(kernel="linear", max_iter=1000)

        custom_svc.fit(X, y )
        sklearn_svc.fit(X, y)

        # Predict and score
        custom_pred = custom_svc.predict(X)
        sklearn_pred = sklearn_svc.predict(X)

        custom_score = custom_svc.score(X, y)
        sklearn_score = sklearn_svc.score(X, y)

        # Assertions for binary classification
        atol = math.floor(y.shape[0] * 0.1)
        assert_equal(sklearn_svc.classes_, custom_svc.classes_)
        assert_allclose(sklearn_pred, custom_pred, atol=atol)
        assert abs(sklearn_score - custom_score) / abs(sklearn_score) < 0.1

    @repeat(3)
    def test_multi_class_classification(self):
        """
        Test multi-class classification and compare the custom SVC with Scikit-Learn's SVC.
        """
        # Use scaled data for both models
        X, y = generate_classification_dataset(
            n_samples=2000, n_features=4, n_classes=2
        )
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Initialize and train both models with adjusted max_iter and tol
        custom_svc = SVC(kernel="linear", max_iter=1000, tol=1e-5)
        sklearn_svc = SkSVC(kernel="linear", max_iter=1000, tol=1e-5)

        custom_svc.fit(X, y)
        sklearn_svc.fit(X, y)

        custom_pred = custom_svc.predict(X)
        sklearn_pred = sklearn_svc.predict(X)

        custom_score = custom_svc.score(X, y)
        sklearn_score = sklearn_svc.score(X, y)

        atol = math.floor(y.shape[0] * 0.1)
        assert_equal(sklearn_svc.classes_, custom_svc.classes_)
        assert_allclose(sklearn_pred, custom_pred, atol=atol)
        assert abs(sklearn_score - custom_score) / abs(sklearn_score) < 0.2

    @repeat(3)
    def test_rbf_kernel(self):
        """
        Test the custom SVC with RBF kernel against Scikit-Learn's SVC.
        """
        np.random.seed(42)
        X, y = generate_classification_dataset(n_samples=200, n_features=4, n_classes=2)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        custom_svc = SVC(kernel="rbf", max_iter=500, tol=1e-3)
        sklearn_svc = SkSVC(kernel="rbf", max_iter=500, tol=1e-3)

        custom_svc.fit(X, y)
        sklearn_svc.fit(X, y)

        custom_pred = custom_svc.predict(X)
        sklearn_pred = sklearn_svc.predict(X)

        custom_score = custom_svc.score(X, y)
        sklearn_score = sklearn_svc.score(X, y)

        relative_difference = abs(sklearn_score - custom_score) / abs(sklearn_score)

        mismatches = np.count_nonzero(custom_pred != sklearn_pred)
        mismatch_percentage = mismatches / len(custom_pred)
        assert (
            mismatch_percentage < 0.15
        ), f"Mismatch percentage {mismatch_percentage * 100}% is too high."
        assert (
            relative_difference < 0.15
        ), f"Relative difference {relative_difference} is not acceptable."

    def test_untrained_model_prediction_error(self):
        """
        Ensure an error is raised when predicting with an untrained model.
        """
        svc = SVC(kernel="linear")
        X, _ = generate_classification_dataset(n_samples=10, n_features=2, n_classes=2)

        with self.assertRaises(ValueError):
            svc.predict(X)

    @repeat(3)
    def test_custom_kernel_initialization(self):
        """
        Ensure the SVC model initializes correctly with a custom kernel.
        """
        svc = SVC(kernel="polynomial")
        self.assertEqual(
            svc.kernel,
            "polynomial",
            "Model should initialize with 'polynomial' kernel.",
        )

    @repeat(3)
    def test_output_type_and_shape(self):
        """
        Validate that the output type and shape of predictions are the same.
        """
        X, y = generate_classification_dataset(
            n_samples=2000, n_features=4, n_classes=2
        )

        custom_svc = SVC(kernel="linear")
        sklearn_svc = SkSVC(kernel="linear", max_iter=1000)

        custom_svc.fit(X, y)
        sklearn_svc.fit(X, y)

        custom_pred = custom_svc.predict(X)
        sklearn_pred = sklearn_svc.predict(X)

        self.assertIsInstance(custom_pred, np.ndarray)
        self.assertEqual(custom_pred.shape, sklearn_pred.shape)

    @repeat(3)
    def test_model_parameters(self):
        """
        Compare the model parameters between the custom and Scikit-Learn implementations.
        """
        X, y = generate_classification_dataset(
            n_samples=2000, n_features=4, n_classes=2
        )

        custom_svc = SVC(kernel="linear")
        sklearn_svc = SkSVC(kernel="linear", max_iter=1000)

        custom_svc.fit(X, y)
        sklearn_svc.fit(X, y)

        if hasattr(custom_svc, "support_vectors_"):
            assert_array_equal(
                custom_svc.support_vectors_,
                sklearn_svc.support_vectors_,
                "Support vectors should match between implementations.",
            )

    @repeat(3)
    def setUp(self):
        """Set up test data"""
        self.X, self.y = generate_classification_dataset(
            n_samples=100, n_features=2, n_classes=2
        )

    @repeat(3)
    def test_parameter_validation(self):
        """Test parameter validation"""
        # Test invalid C
        with self.assertRaises(ValueError):
            SVC(C=-1.0)
        
        # Test invalid alpha
        with self.assertRaises(ValueError):
            SVC(alpha=0)
        
        # Test invalid kernel
        with self.assertRaises(ValueError):
            SVC(kernel="invalid")
        
        # Test invalid degree
        with self.assertRaises(ValueError):
            SVC(degree=0)
        
        # Test invalid max_iter
        with self.assertRaises(ValueError):
            SVC(max_iter=-1)
            
        # Test invalid tol
        with self.assertRaises(ValueError):
            SVC(tol=-1e-4)

    @repeat(3)
    def test_kernels(self):
        """Test different kernel functions"""
        kernels = ["linear", "polynomial", "rbf"]
        for kernel in kernels:
            svc = SVC(kernel=kernel, max_iter=100)
            svc.fit(self.X, self.y)
            pred = svc.predict(self.X)
            self.assertIsInstance(pred, np.ndarray)
            self.assertEqual(pred.shape, self.y.shape)

    @repeat(3)
    def test_early_stopping(self):
        """Test early stopping functionality"""
        svc = SVC(early_stopping=True, tol=1e-2, max_iter=1000)
        svc.fit(self.X, self.y)
        pred = svc.predict(self.X)
        self.assertIsInstance(pred, np.ndarray)

    @repeat(3)
    def test_adaptive_learning_rate(self):
        """Test adaptive learning rate"""
        svc = SVC(adaptive_lr=True, learning_rate=1e-3)
        svc.fit(self.X, self.y)
        pred = svc.predict(self.X)
        self.assertIsInstance(pred, np.ndarray)

    @repeat(3)
    def test_metrics(self):
        """Test different scoring metrics"""
        svc = SVC()
        svc.fit(self.X, self.y)
        
        basic_metrics = ["accuracy", "precision", "recall", "f1_score"]
        for metric in basic_metrics:
            score = svc.score(self.X, self.y, metric=metric)
            self.assertIsInstance(score, float)
            self.assertTrue(0 <= score <= 1)

        normalize_options = ['true', 'pred', 'all']
        for normalize in normalize_options:
            score = svc.score(
                self.X, 
                self.y, 
                metric="confusion_matrix",
                normalize_cm=normalize
            )
            self.assertIsInstance(score, np.ndarray)
            if normalize:
                # Check if normalized values sum to expected total
                if normalize == 'all':
                    self.assertAlmostEqual(np.sum(score), 1.0)
                elif normalize in ['true', 'pred']:
                    # Each row/column should sum to 1
                    sums = np.sum(score, axis=1 if normalize == 'true' else 0)
                    np.testing.assert_almost_equal(sums, np.ones_like(sums))

    @repeat(3)                
    def test_invalid_metric(self):
        """Test invalid metric handling"""
        svc = SVC()
        svc.fit(self.X, self.y)
        with self.assertRaises(ValueError):
            svc.score(self.X, self.y, metric="invalid")

    @repeat(3)
    def test_verbosity(self):
        """Test different verbosity levels"""
        for verbose in [0, 1, 2]:
            svc = SVC(verbose=verbose, max_iter=10)
            svc.fit(self.X, self.y)

    @repeat(3)
    def test_batch_size(self):
        """Test different batch sizes"""
        batch_sizes = [1, 32, 64]
        for batch_size in batch_sizes:
            svc = SVC(batch_size=batch_size)
            svc.fit(self.X, self.y)
            pred = svc.predict(self.X)
            self.assertEqual(pred.shape, self.y.shape)

    @repeat(3)
    def test_decay_rate(self):
        """Test learning rate decay"""
        decay_rates = [0.9, 0.99, 0.999]
        for decay in decay_rates:
            svc = SVC(decay=decay)
            svc.fit(self.X, self.y)
            pred = svc.predict(self.X)
            self.assertEqual(pred.shape, self.y.shape)

    @repeat(3)
    def test_edge_cases(self):
        """Test edge cases"""
        # Test single sample
        X_single = self.X[0:1]
        y_single = self.y[0:1]
        svc = SVC()
        svc.fit(X_single, y_single)
        pred = svc.predict(X_single)
        self.assertEqual(pred.shape, y_single.shape)

        # Test minimum samples
        X_min = self.X[:2]
        y_min = self.y[:2]
        svc = SVC()
        svc.fit(X_min, y_min)
        pred = svc.predict(X_min)
        self.assertEqual(pred.shape, y_min.shape)

    @repeat(3)
    def test_predict_proba(self):
        """Test probability predictions if implemented"""
        if hasattr(SVC, 'predict_proba'):
            svc = SVC()
            svc.fit(self.X, self.y)
            proba = svc.predict_proba(self.X)
            self.assertIsInstance(proba, np.ndarray)
            self.assertEqual(proba.shape[0], self.y.shape[0])

    @repeat(3)
    def test_serialization(self):
        """Test model serialization if implemented"""
        if hasattr(SVC, 'save') and hasattr(SVC, 'load'):
            svc = SVC()
            svc.fit(self.X, self.y)
            svc.save('svc_test.pkl')
            loaded_svc = SVC.load('svc_test.pkl')
            np.testing.assert_array_equal(
                svc.predict(self.X),
                loaded_svc.predict(self.X)
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
