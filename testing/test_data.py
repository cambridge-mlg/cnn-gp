import unittest
import torch

from cnn_gp import ProductIterator

class TestProductIterator(unittest.TestCase):
    @staticmethod
    def construct_K_iteratively(batch_size, X, X2, n_workers):
        if X2 is None:
            K = torch.zeros((len(X), len(X)), dtype=torch.int64)
        else:
            K = torch.zeros((len(X), len(X2)), dtype=torch.int64)

        no_duplicated_total = K.numel()
        for worker_rank in range(n_workers):
            it = ProductIterator(batch_size, X, X2, worker_rank, n_workers)
            empirical_iter = 0
            for (same, (i, (x1,)), (j, (x2,))) in it:
                k_ij = x1 + x2.t()

                K[i:i+batch_size, j:j+batch_size] = k_ij
                no_duplicated_total -= k_ij.numel()
                if X2 is None:
                    if i == j:
                        assert same, "improper same"
                    else:
                        K[j:j+batch_size, i:i+batch_size] = k_ij.t()
                        no_duplicated_total -= k_ij.numel()
                        assert not same
                else:
                    assert not same

                empirical_iter += 1
            assert empirical_iter == len(it)

        assert no_duplicated_total == 0, "exactly as many iterations as kernel elements"
        return K

    @staticmethod
    def two_dsets(size1, size2):
        X1 = torch.arange(size1).unsqueeze(-1)
        X2 = torch.arange(size2).unsqueeze(-1).sum(1).mul(size1)
        return tuple(map(torch.utils.data.TensorDataset, (X1, X2)))

    @staticmethod
    def one_dset(size):
        return torch.utils.data.TensorDataset(torch.arange(size).pow(2).unsqueeze(-1))

    def test_equivalentK(self):
        for sz1, sz2, bs, nw in [
                (11, 12, 3, 1),
                (11, 12, 3, 2), # more workers
                (11, 12, 3, 5), # more workers, not divisible
                (2, 5, 4, 1),  # Batch size larger than x1
                (5, 3, 4, 1),  # bs larger than x2
                (5, 3, 6, 1),  # bs larger than both
                (2, 5, 4, 2),  # same 3 as before but with 2 workers
                (5, 3, 4, 2),  #
                (5, 3, 6, 2),  #
                (2, 5, 4, 8),  # same 3 as before but with 8 workers
                (5, 3, 4, 8),  #
                (5, 3, 6, 8),  #
                (3, 5, 4, 4),  # more workers than sz1
                (3, 5, 4, 8),  # more workers than both
        ]:
            X, X2 = self.two_dsets(sz1, sz2)
            K = X.tensors[0] + X2.tensors[0].t()
            assert len(set(k.item() for k in K.view(-1))) == K.numel(), (
                "all K are not different, so test would not be conclusive")
            K_ = self.construct_K_iteratively(bs, X, X2, nw)
            assert torch.equal(K, K_)

    def test_equivalentK_symm(self):
        for sz, bs, nw in [
                (11, 3, 1),
                (11, 3, 2), # more workers
                (11, 3, 5), # more workers, not divisible
                (2, 4, 1),  # Batch size larger than x
                (5, 6, 1),  # bs larger than x
                (2, 4, 2),  # same 2 as before but with 2 workers
                (5, 6, 2),  #
                (2, 4, 8),  # same 2 as before but with 2 workers
                (5, 6, 8),  #
                (3, 4, 4),  # more workers than sz1
                (3, 4, 8),  # more workers than both
        ]:
            X, X2 = self.one_dset(sz), None
            K = X.tensors[0] + X.tensors[0].t()
            assert len(set(k.item() for k in K.view(-1))) >= K.numel()//2, (
                "not enough K are different, so test would not be conclusive")
            K_ = self.construct_K_iteratively(bs, X, X2, nw)
            assert torch.equal(K, K_)

