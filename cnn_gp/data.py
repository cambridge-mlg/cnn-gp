import torchvision
from torch.utils.data import ConcatDataset, DataLoader, Subset
import os
import numpy as np
import math

__all__ = ('DatasetFromConfig', 'ProductDataLoader')


def _this_worker_batch(N_batches, worker_rank, n_workers):
    batches_per_worker = np.zeros([n_workers], dtype=np.int)
    batches_per_worker[:] = N_batches // n_workers
    batches_per_worker[:N_batches % n_workers] += 1

    start_batch = np.sum(batches_per_worker[:worker_rank])
    batches_this_worker = batches_per_worker[worker_rank]

    return start_batch, batches_this_worker


def _product_generator(N_batches_X, N_batches_X2, same):
    for i in range(N_batches_X):
        if same:
            # Yield only upper triangle
            yield (True, i, i)
        for j in range(i+1 if same else 0,
                       N_batches_X2):
            yield (False, i, j)


def _round_up_div(a, b):
    return (a+b-1)//b


class ProductDataLoader(object):
    """
    Returns an iterator for loading data from both X and X2. It divides the
    load equally among `n_workers`, returning only the one that belongs to
    `worker_rank`.

    TODO: test this vs. a single batch of kernel calculation for both same and non-same, and for different small batch sizes.
    """
    def __init__(self, batch_size, X, X2=None, worker_rank=0, n_workers=1):
        N_batches_X = _round_up_div(len(X), batch_size)
        if X2 is None:
            same = True
            X2 = X
            N_batches = N_batches_X * (N_batches_X-1) // 2
        else:
            same = False
            N_batches_X2 = _round_up_div(len(X2), batch_size)
            N_batches = N_batches_X * N_batches_X2

        start_batch, self.batches_this_worker = _this_worker_batch(
            N_batches, worker_rank, n_workers)

        gen = iter(_product_generator(N_batches_X, N_batches_X2, same))
        # Iterate until we get to our start_batch. O(n) but whatever
        for _ in range(start_batch):
            next(gen)
        self.idx_iter = gen

        self.prev_j = -2  # this + 1 = -1, which is not a valid j
        self.X_loader = None
        self.X2_loader = None
        self.X = X
        self.X2 = X2
        self.same = same
        self.batch_size = batch_size

    def __len__(self):
        return self.batches_this_worker

    def __iter__(self):
        return self

    def dataloader_beginning_at(self, i, dataset):
        return iter(DataLoader(
            Subset(dataset, slice(i*self.batch_size, len(dataset))),
            batch_size=self.batch_size))

    def __next__(self):
        same, i, j = next(self.idx_iter)

        if self.X_loader is None:
            self.X_loader = self.dataloader_beginning_at(i, self.X)

        if j != self.prev_j+1:
            self.X2_loader = self.dataloader_beginning_at(j, self.X2)
        self.prev_j = j

        return (same,
                (i*self.batch_size, next(self.X_loader)),
                (j*self.batch_size, next(self.X2_loader)))


class DatasetFromConfig(object):
    """
    A dataset that contains train, validation and test, and is created from a
    config file.
    """
    def __init__(self, datasets_path, config):
        """
        Requires:
        config.dataset_name (e.g. "MNIST")
        config.train_range
        config.test_range
        """
        self.config = config

        trans = torchvision.transforms.ToTensor()
        if len(config.transforms) > 0:
            trans = torchvision.transforms.Compose([trans] + config.transforms)

        # Full datasets
        datasets_path = os.path.join(datasets_path, config.dataset_name)
        train_full = config.dataset(datasets_path, train=True, download=True,
                                    transform=trans)
        test_full = config.dataset(datasets_path, train=False, transform=trans)
        self.data_full = ConcatDataset([train_full, test_full])

        # Our training/test split
        # (could omit some data, or include validation in test)
        self.train = Subset(self.data_full, config.train_range)
        self.validation = Subset(self.data_full, config.validation_range)
        self.test = Subset(self.data_full, config.test_range)
