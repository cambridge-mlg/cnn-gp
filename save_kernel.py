import absl.app
import h5py
import torch
import importlib
from mpi4py import MPI

from cnn_gp import DatasetFromConfig, save_K
FLAGS = absl.app.flags.FLAGS


def main(_):
    n_workers = MPI.COMM_WORLD.size
    worker_rank = MPI.COMM_WORLD.rank

    config = importlib.import_module(f"configs.{FLAGS.config}")
    dataset = DatasetFromConfig(FLAGS.datasets_path, config)
    model = config.initial_model.cuda()

    def kern(x, x2, same, diag):
        with torch.no_grad():
            if x2 is not None:
                x2 = x2.cuda()
            return model(x.cuda(), x2, same, diag).detach().cpu().numpy()

    if n_workers == 1:
        kwargs = {}
    else:
        kwargs = dict(driver="mpio", comm=MPI.COMM_WORLD)

    with h5py.File(FLAGS.out_path, "w", **kwargs) as f:
        kwargs = dict(n_workers=n_workers, worker_rank=worker_rank,
                      batch_size=config.kernel_batch_size)
        save_K(f, kern, name="Kxx",     X=dataset.train,      X2=None,          diag=False, **kwargs)
        save_K(f, kern, name="Kxvx",    X=dataset.validation, X2=dataset.train, diag=False, **kwargs)
        save_K(f, kern, name="Kxtx",    X=dataset.test,       X2=dataset.train, diag=False, **kwargs)
        save_K(f, kern, name="Kv_diag", X=dataset.validation, X2=None,          diag=True, **kwargs)
        save_K(f, kern, name="Kt_diag", X=dataset.test,       X2=None,          diag=True, **kwargs)


if __name__ == '__main__':
    f = absl.app.flags
    f.DEFINE_string("datasets_path", "/scratch/ag919/datasets/",
                    "where to save datasets")
    f.DEFINE_enum("config", "mnist", ["mnist", "cifar10"], "which config to load from `configs`")
    f.DEFINE_string('out_path', "/scratch/ag919/grams/mnist.h5",
                    "path of h5 file to save kernels in")
    absl.app.run(main)
