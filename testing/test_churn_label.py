import gc
import unittest

import pytest

from labeler.churn_label import ChurnLabeler


class TestChurnLabeler(unittest.TestCase):

    def setUp(self):

        "Hook method for setting up the test fixture before exercising it."
        self.churn_label = ChurnLabeler()
        self.data = self.churn_label.data

    def tearDown(self):
        "Hook method for deconstructing the test fixture after testing it."

        del self.churn_label
        del self.data
        gc.collect()

    def test_non_empty_label(self):
        self.assertGreater(self.data.shape[0], 0, 'Empty label')

    def test_equal_features_labels_rows(self):
        features, labels = self.churn_label.get_features_label()
        self.assertEqual(features.shape[0], labels.shape[0], 'Number of rows in features are not equal to labels')


if __name__ == '__main__':
    unittest.main()
