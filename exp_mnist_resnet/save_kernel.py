"""
Save a kernel matrix to disk
"""
import absl.app
import h5py
import torch
import importlib
import os

from cnn_gp import DatasetFromConfig, save_K
FLAGS = absl.app.flags.FLAGS


def main(_):
    print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    n_workers, worker_rank = FLAGS.n_workers, FLAGS.worker_rank
    config = importlib.import_module(f"configs.{FLAGS.config}")
    dataset = DatasetFromConfig(FLAGS.datasets_path, config)
    model = config.initial_model.cuda()

    def kern(x, x2, same, diag):
        with torch.no_grad():
            return model(x.cuda(), x2.cuda(), same,
                         diag).detach().cpu().numpy()

    with h5py.File(FLAGS.out_path, "w") as f:
        kwargs = dict(worker_rank=worker_rank, n_workers=n_workers,
                      batch_size=FLAGS.batch_size, print_interval=2.)
        save_K(f, kern, name="Kxx",     X=dataset.train,      X2=None,          diag=False, **kwargs)
        save_K(f, kern, name="Kxvx",    X=dataset.validation, X2=dataset.train, diag=False, **kwargs)
        save_K(f, kern, name="Kxtx",    X=dataset.test,       X2=dataset.train, diag=False, **kwargs)

    if worker_rank == 0:
        with h5py.File(FLAGS.out_path, "a") as f:
            save_K(f, kern, name="Kv_diag", X=dataset.validation, X2=None, diag=True, **kwargs)
            save_K(f, kern, name="Kt_diag", X=dataset.test,       X2=None, diag=True, **kwargs)


if __name__ == '__main__':
    f = absl.app.flags
    f.DEFINE_string("datasets_path", "/scratch/ag919/datasets/",
                    "where to save datasets")
    f.DEFINE_integer('batch_size', 200,
                     "max number of examples to simultaneously compute "
                     "the kernel of")
    f.DEFINE_string("config", "mnist", "which config to load from `configs`")
    f.DEFINE_integer("n_workers", 1, "num of workers")
    f.DEFINE_integer("worker_rank", 0, "rank of worker")
    f.DEFINE_string('out_path', None, "path of h5 file to save kernels in")
    absl.app.run(main)
