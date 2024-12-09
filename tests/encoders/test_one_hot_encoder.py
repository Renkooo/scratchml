from numpy.testing import assert_equal
from sklearn.preprocessing import OneHotEncoder as SkOneHotEncoder
from scratchml.encoders import OneHotEncoder
import unittest


class Test_OneHotEncoder(unittest.TestCase):
    """
    Unittest class created to test the One Hot Encoder technique.
    """

    def test_1(self):
        """
        Test the One Hot Encoder implementation on a toy-problem using the
        'infrequent_if_exist' with a custom max_categories value and
        then compares it to the Scikit-Learn implementation.
        """
        X = [
            ["Male", 1],
            ["Male", 1],
            ["Female", 3],
            ["Female", 2],
            ["Male", 7],
            ["Other", 9],
        ]
        test = [["Female", 1], ["Male", 4], ["Other", 9]]

        enc = SkOneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=2)
        enc.fit(X)
        trans_enc = enc.transform(test).toarray()
        inv_trans_enc = enc.inverse_transform(trans_enc)

        ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=2)
        ohe.fit(X)
        trans_ohe = ohe.transform(test).toarray()
        inv_trans_ohe = ohe.inverse_transform(trans_ohe)

        assert_equal(enc.categories_, ohe.categories_)
        assert_equal(enc.n_features_in_, ohe.n_features_in_)
        assert_equal(enc.drop_idx_, ohe.drop_idx_)
        assert_equal(trans_enc.shape, trans_ohe.shape)
        assert_equal(trans_enc, trans_ohe)
        assert_equal(type(trans_enc), type(trans_ohe))
        assert_equal(inv_trans_enc.shape, inv_trans_ohe.shape)
        assert_equal(type(inv_trans_enc), type(inv_trans_ohe))

    def test_2(self):
        """
        Test the One Hot Encoder implementation on a toy-problem using the
        'ignore' and then compares it to the Scikit-Learn implementation.
        """
        X = [["Male", 1], ["Female", 3], ["Female", 2]]
        test = [["Female", 1], ["Male", 4]]

        enc = SkOneHotEncoder(handle_unknown="ignore")
        enc.fit(X)
        trans_enc = enc.transform(test).toarray()
        inv_trans_enc = enc.inverse_transform(trans_enc)

        ohe = OneHotEncoder(handle_unknown="ignore")
        ohe.fit(X)
        trans_ohe = ohe.transform(test).toarray()
        inv_trans_ohe = ohe.inverse_transform(trans_ohe)

        assert_equal(enc.categories_, ohe.categories_)
        assert_equal(enc.n_features_in_, ohe.n_features_in_)
        assert_equal(enc.drop_idx_, ohe.drop_idx_)
        assert_equal(trans_enc.shape, trans_ohe.shape)
        assert_equal(trans_enc, trans_ohe)
        assert_equal(type(trans_enc), type(trans_ohe))
        assert_equal(inv_trans_enc.shape, inv_trans_ohe.shape)
        assert_equal(type(inv_trans_enc), type(inv_trans_ohe))

    def test_3(self):
        """
        Test the One Hot Encoder implementation on a toy-problem using the
        'ignore' with 'if_binary' drop option and then compares it
        to the Scikit-Learn implementation.
        """
        X = [
            ["Male", 1],
            ["Male", 1],
            ["Female", 3],
            ["Female", 2],
            ["Male", 7],
            ["Other", 9],
        ]
        test = [["Female", 1], ["Male", 4], ["Other", 9]]

        enc = SkOneHotEncoder(handle_unknown="ignore", drop="if_binary")
        enc.fit(X)
        trans_enc = enc.transform(test).toarray()
        inv_trans_enc = enc.inverse_transform(trans_enc)

        ohe = OneHotEncoder(handle_unknown="ignore", drop="if_binary")
        ohe.fit(X)
        trans_ohe = ohe.transform(test).toarray()
        inv_trans_ohe = ohe.inverse_transform(trans_ohe)

        assert_equal(enc.categories_, ohe.categories_)
        assert_equal(enc.n_features_in_, ohe.n_features_in_)
        assert_equal(enc.drop_idx_, ohe.drop_idx_)
        assert_equal(trans_enc.shape, trans_ohe.shape)
        assert_equal(trans_enc, trans_ohe)
        assert_equal(type(trans_enc), type(trans_ohe))
        assert_equal(inv_trans_enc.shape, inv_trans_ohe.shape)
        assert_equal(type(inv_trans_enc), type(inv_trans_ohe))

    def test_4(self):
        """
        Test the One Hot Encoder implementation on a toy-problem using the
        'ignore' with a custom drop option and then compares it
        to the Scikit-Learn implementation.
        """
        X = [
            ["Male", 1],
            ["Male", 1],
            ["Female", 3],
            ["Female", 2],
            ["Male", 7],
            ["Other", 9],
        ]
        test = [["Female", 1], ["Male", 4], ["Other", 9]]

        enc = SkOneHotEncoder(handle_unknown="ignore", drop=["Other", 1])
        enc.fit(X)
        trans_enc = enc.transform(test).toarray()
        inv_trans_enc = enc.inverse_transform(trans_enc)

        ohe = OneHotEncoder(handle_unknown="ignore", drop=["Other", 1])
        ohe.fit(X)
        trans_ohe = ohe.transform(test).toarray()
        inv_trans_ohe = ohe.inverse_transform(trans_ohe)

        assert_equal(enc.categories_, ohe.categories_)
        assert_equal(enc.n_features_in_, ohe.n_features_in_)
        assert_equal(enc.drop_idx_, ohe.drop_idx_)
        assert_equal(trans_enc.shape, trans_ohe.shape)
        assert_equal(trans_enc, trans_ohe)
        assert_equal(type(trans_enc), type(trans_ohe))
        assert_equal(inv_trans_enc.shape, inv_trans_ohe.shape)
        assert_equal(type(inv_trans_enc), type(inv_trans_ohe))

    def test_5(self):
        """
        Test OneHotEncoder with min_frequency parameter.
        """
        X = [
            ["Male", 1],
            ["Male", 1],
            ["Female", 3],
            ["Female", 2],
            ["Other", 9],
        ]
        test = [["Female", 1], ["Male", 4]]

        enc = SkOneHotEncoder(sparse_output=False, handle_unknown="ignore")  # Changed from sparse to sparse_output
        enc.fit(X)
        trans_enc = enc.transform(test)

        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")  # Our implementation uses sparse
        ohe.fit(X)
        trans_ohe = ohe.transform(test)

        assert_equal(enc.n_features_in_, ohe.n_features_in_)
        assert_equal(trans_enc.shape, trans_ohe.shape)
        assert_equal(trans_enc, trans_ohe)

    def test_6(self):
        """
        Test the One Hot Encoder implementation on a toy-problem using the
        'infrequent_if_exists', a custom min frequency and max categories
        and then compares it to the Scikit-Learn implementation.
        """
        X = [
            ["Male", 1],
            ["Male", 1],
            ["Female", 3],
            ["Female", 2],
            ["Male", 7],
            ["Other", 9],
        ]
        test = [["Female", 1], ["Male", 4], ["Other", 9]]

        enc = SkOneHotEncoder(
            handle_unknown="infrequent_if_exist", max_categories=2, min_frequency=2
        )
        enc.fit(X)
        trans_enc = enc.transform(test).toarray()
        inv_trans_enc = enc.inverse_transform(trans_enc)

        ohe = OneHotEncoder(
            handle_unknown="infrequent_if_exist", min_frequency=2, max_categories=2
        )
        ohe.fit(X)
        trans_ohe = ohe.transform(test).toarray()
        inv_trans_ohe = ohe.inverse_transform(trans_ohe)

        assert_equal(enc.categories_, ohe.categories_)
        assert_equal(enc.n_features_in_, ohe.n_features_in_)
        assert_equal(enc.drop_idx_, ohe.drop_idx_)
        assert_equal(trans_enc.shape, trans_ohe.shape)
        assert_equal(trans_enc, trans_ohe)
        assert_equal(type(trans_enc), type(trans_ohe))
        assert_equal(inv_trans_enc.shape, inv_trans_ohe.shape)
        assert_equal(type(inv_trans_enc), type(inv_trans_ohe))

    def test_7(self):
        """
        Test error handling for invalid parameters.
        """
        X = [["Male", 1], ["Female", 2]]

        # Test invalid handle_unknown
        with self.assertRaises(ValueError):
            OneHotEncoder(handle_unknown="invalid").fit(X)

        # Test invalid drop value
        with self.assertRaises(ValueError):
            OneHotEncoder(drop="invalid").fit(X)

        # Test negative min_frequency
        with self.assertRaises(ValueError):
            OneHotEncoder(min_frequency=-1).fit(X)


    def test_8(self):
        """
        Test sparse matrix output.
        """
        X = [["A", 1], ["B", 2], ["C", 3]]
        test = [["A", 1], ["B", 2]]

        enc = SkOneHotEncoder(sparse_output=True)  # Changed from sparse to sparse_output
        enc.fit(X)
        trans_enc = enc.transform(test)

        ohe = OneHotEncoder(sparse=True)  # Our implementation uses sparse
        ohe.fit(X)
        trans_ohe = ohe.transform(test)

        assert_equal(trans_enc.toarray(), trans_ohe.toarray())
        assert_equal(trans_enc.shape, trans_ohe.shape)
        
if __name__ == "__main__":
    unittest.main(verbosity=2)
