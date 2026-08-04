import unittest

from attentionv3.data import cifar100_train_validation_indices


class CIFAR100SplitTests(unittest.TestCase):
    def test_split_is_deterministic_disjoint_and_complete(self):
        train_a, validation_a = cifar100_train_validation_indices(100, 20, seed=7)
        train_b, validation_b = cifar100_train_validation_indices(100, 20, seed=7)
        self.assertEqual((train_a, validation_a), (train_b, validation_b))
        self.assertEqual(len(train_a), 80)
        self.assertEqual(len(validation_a), 20)
        self.assertFalse(set(train_a).intersection(validation_a))
        self.assertEqual(set(train_a).union(validation_a), set(range(100)))

    def test_invalid_validation_size_is_rejected(self):
        with self.assertRaises(ValueError):
            cifar100_train_validation_indices(100, 0)


if __name__ == "__main__":
    unittest.main()
