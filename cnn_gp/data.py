import torchvision
from torch.utils.data import ConcatDataset, DataLoader, Subset
import os
import numpy as np
import itertools
import torch

__all__ = ('DatasetFromConfig', 'ProductIterator', 'DiagIterator',
           'print_timings')


def _this_worker_batch(N_batches, worker_rank, n_workers):
    batches_per_worker = np.zeros([n_workers], dtype=np.int)
    batches_per_worker[:] = N_batches // n_workers
    batches_per_worker[:N_batches % n_workers] += 1

    start_batch = worker_rank
    batches_this_worker = batches_per_worker[worker_rank]

    return int(start_batch), int(batches_this_worker)


def _product_generator(N_batches_X, N_batches_X2, same):
    for i in range(N_batches_X):
        # Yield only lower triangle if same
        for j in range(i if same else N_batches_X2):
            yield (False, i, j)
        if same:
            yield (True, i, i)


def _round_up_div(a, b):
    return (a+b-1)//b


class ProductIterator(object):
    """
    Returns an iterator for loading data from both X and X2. It divides the
    load equally among `n_workers`, returning only the one that belongs to
    `worker_rank`.
    """
    def __init__(self, batch_size, X, X2=None, worker_rank=0, n_workers=1):
        N_batches_X = _round_up_div(len(X), batch_size)
        if X2 is None:
            same = True
            X2 = X
            N_batches_X2 = N_batches_X
            N_batches = max(1, N_batches_X * (N_batches_X+1) // 2)
        else:
            same = False
            N_batches_X2 = _round_up_div(len(X2), batch_size)
            N_batches = N_batches_X * N_batches_X2

        start_batch, self.batches_this_worker = _this_worker_batch(
            N_batches, worker_rank, n_workers)

        self.idx_iter = itertools.islice(
            _product_generator(N_batches_X, N_batches_X2, same),
            start_batch, None, n_workers)

        self.n_workers = n_workers
        self.X = X
        self.X2 = X2
        self.same = same
        self.batch_size = batch_size
        self.X_loader = None
        self.prev_i = None

    def __len__(self):
        return self.batches_this_worker

    def __iter__(self):
        return self

    def dataloader_beginning_at(self, i, dataset, step_mult=None):
        B = self.batch_size
        if step_mult is None:
            idx = range(i*B, len(dataset), 1)
        else:
            idx = torch.cat([
                torch.arange(j, min(j+B, len(dataset)), device='cpu')
                for j in range(i*B, len(dataset), B*step_mult)
            ])
        return iter(DataLoader(Subset(dataset, idx), batch_size=self.batch_size))

    def __next__(self):
        same, i, j = next(self.idx_iter)

        if self.X_loader is None:
            self.X_loader = self.dataloader_beginning_at(i, self.X, None)
            self.prev_i = i-1

        if i != self.prev_i:
            self.X2_loader = self.dataloader_beginning_at(j, self.X2, self.n_workers)

        while i != self.prev_i:
            self.x_batch = next(self.X_loader)
            self.prev_i += 1

        return (same,
                (i*self.batch_size, self.x_batch),
                (j*self.batch_size, next(self.X2_loader)))


class DiagIterator(object):
    def __init__(self, batch_size, X, X2=None):
        self.batch_size = batch_size
        dl = DataLoader(X, batch_size=batch_size)
        if X2 is None:
            self.same = True
            self.it = iter(enumerate(dl))
            self.length = len(dl)
        else:
            dl2 = DataLoader(X2, batch_size=batch_size)
            self.same = False
            self.it = iter(enumerate(zip(dl, dl2)))
            self.length = min(len(dl), len(dl2))

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

    def __next__(self):
        if self.same:
            i, xy = next(self.it)
            xy2 = xy
        else:
            i, xy, xy2 = next(self.it)
        ib = i*self.batch_size
        return (self.same, (ib, xy), (ib, xy2))


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

    @staticmethod
    def load_full(dataset):
        return next(iter(DataLoader(dataset, batch_size=len(dataset))))


def _hhmmss(s):
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h == 0.0:
        return f"{m:02d}:{s:02d}"
    else:
        return f"{h:02d}:{m:02d}:{s:02d}"


def print_timings(iterator, desc="time", print_interval=2.):
    """
    Prints the current total number of iterations, speed of iteration, and
    elapsed time.

    Meant as a rudimentary replacement for `tqdm` that prints a new line at
    each iteration, and thus can be used in multiple parallel processes in the
    same terminal.
    """
    import time
    start_time = time.perf_counter()
    total = len(iterator)
    last_printed = -print_interval
    for i, value in enumerate(iterator):
        yield value
        cur_time = time.perf_counter()
        elapsed = cur_time - start_time
        if elapsed > last_printed + print_interval:
            it_s = (i+1)/elapsed
            total_s = total/it_s
            print(f"{desc}: {i+1}/{total} it, {it_s:.02f} it/s,"
                  f"[{_hhmmss(elapsed)}<{_hhmmss(total_s)}]")
            last_printed = elapsed
