"""
Given a pre-computed kernel and a data set, compute train/validation/test accuracy.
"""
import absl.app
import h5py
import numpy as np
import scipy.linalg
import torch
import sklearn.metrics
import scipy

import importlib
from cnn_gp import DatasetFromConfig
FLAGS = absl.app.flags.FLAGS


def solve_system(Kxx, Y):
    print("Running scipy solve Kxx^-1 Y routine")
    assert Kxx.dtype == torch.float64 and Y.dtype == torch.float64, """
    It is important that `Kxx` and `Y` are `float64`s for the inversion,
    even if they were `float32` when being calculated. This makes the
    inversion much less likely to complain about the matrix being singular.
    """
    A = scipy.linalg.solve(
        Kxx.numpy(), Y.numpy(), overwrite_a=True, overwrite_b=False,
        check_finite=False, assume_a='pos')
    return torch.from_numpy(A)


def diag_add(K, diag):
    if isinstance(K, torch.Tensor):
        K.view(K.numel())[::K.shape[-1]+1] += diag
    elif isinstance(K, np.ndarray):
        K.flat[::K.shape[-1]+1] += diag
    else:
        raise TypeError("What do I do with a `{}`, K={}?".format(type(K), K))


def print_accuracy(A, Kxvx, Y, key):
    Ypred = (Kxvx @ A).argmax(dim=1)
    acc = sklearn.metrics.accuracy_score(Y, Ypred)
    print(f"{key} accuracy: {acc*100}%")


def load_kern(dset, i):
    A = np.empty(dset.shape[1:], dtype=np.float32)
    dset.read_direct(A, source_sel=np.s_[i, :, :])
    return torch.from_numpy(A).to(dtype=torch.float64)


def main(_):
    config = importlib.import_module(f"configs.{FLAGS.config}")
    dataset = DatasetFromConfig(FLAGS.datasets_path, config)

    print("Reading training labels")
    _, Y = dataset.load_full(dataset.train)
    n_classes = Y.max() + 1
    Y_1hot = torch.ones((len(Y), n_classes), dtype=torch.float64).neg_()  # all -1
    Y_1hot[torch.arange(len(Y)), Y] = 1.

    with h5py.File(FLAGS.in_path, "r") as f:
        print("Loading kernel")
        Kxx = load_kern(f["Kxx"], 0)
        diag_add(Kxx, FLAGS.jitter)

        print("Solving Kxx^{-1} Y")
        A = solve_system(Kxx, Y_1hot)

        _, Yv = dataset.load_full(dataset.validation)
        Kxvx = load_kern(f["Kxvx"], 0)
        print_accuracy(A, Kxvx, Yv, "validation")
        del Kxvx
        del Yv

        _, Yt = dataset.load_full(dataset.test)
        Kxtx = load_kern(f["Kxtx"], 0)
        print_accuracy(A, Kxtx, Yt, "test")
        del Kxtx
        del Yt


# @(py36) ag919@ulam:~/Programacio/cnn-gp-pytorch$ python classify_gp.py --in_path=/scratch/ag919/grams_pytorch/mnist_as_tf/00_nwork07.h5 --config=mnist_as_tf
# magma.py has some problem loading. Proceeding anyways using CPU.
# Original error: ignoring magma shit
# Reading training labels
# Loading kernel
# Solving Kxx^{-1} Y
# Running scipy solve Kxx^-1 Y routine
# train accuracy: 10.26%
# validation accuracy: 99.31%
# test accuracy: 99.11999999999999%


if __name__ == '__main__':
    f = absl.app.flags
    f.DEFINE_string("datasets_path", "/scratch/ag919/datasets/",
                    "where to save datasets")
    f.DEFINE_enum("config", "mnist", ["mnist", "mnist_as_tf", "cifar10"], "which config to load from `configs`")
    f.DEFINE_string('in_path', "/scratch/ag919/grams_pytorch/mnist/dest.h5",
                    "path of h5 file to load kernels from")
    f.DEFINE_float("jitter", 0.0, "add to the diagonal")
    absl.app.run(main)
