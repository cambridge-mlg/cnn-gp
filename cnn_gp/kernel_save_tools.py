import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

__all__ = ('create_h5py_dataset', 'compute_Kdiag', 'compute_K', 'save_K')


def create_h5py_dataset(f, batch_size, name, diag, N, N2):
    """
    Creates a dataset named `name` on `f`, with chunks of size `chunk_size_MB`.
    The chunks have leading dimension 1, so as to accommodate future resizing
    of the leading dimension of the dataset (which starts at 1).
    """
    if diag:
        chunk_shape = (1, min(batch_size, N))
        shape = (1, N)
        maxshape = (None, N)
    else:
        chunk_shape = (1, min(batch_size, N), min(batch_size, N2))
        shape = (1, N, N2)
        maxshape = (None, N, N2)
    return f.create_dataset(name, shape=shape, dtype=np.float32,
                            fillvalue=np.nan, chunks=chunk_shape,
                            maxshape=maxshape)


def compute_Kdiag(out, kern, X_loader, X2_loader, same, N_batches, batch_size,
                  layer=0):
    with tqdm(total=N_batches) as pbar:
        for i, ((x, _y), (x2, _y2)) in enumerate(zip(iter(X_loader),
                                                     iter(X2_loader))):
            idx = (layer, slice(i*batch_size, i*batch_size + len(x)))
            # out[idx] = kern(x, x2, same=same, diag=True)
            pbar.update(1)


def compute_K(out, kern, X_loader, X2_loader, same, N_batches, batch_size,
              layer=0):
    if same:
        N_square_batches = N_batches * (N_batches-1)//2
    else:
        N_square_batches = N_batches * N_batches
    with tqdm(total=N_square_batches) as pbar:
        for i, (x, _y) in enumerate(iter(X_loader)):
            for j, (x2, _y2) in enumerate(iter(X2_loader)):
                if j > i and same:
                    break

                idx = (layer,
                       slice(i*batch_size, i*batch_size + len(x)),
                       slice(j*batch_size, j*batch_size + len(x2)))

                # if i == j and same:
                #     out[idx] = kern(x, x2, same=True, diag=False)
                # else:
                #     out[idx] = kern(x, x2, same=False, diag=False)
                pbar.update(1)


def save_K(f, kern, name, X, X2, diag, n_workers=1, worker_rank=0, n_max=400):
    N = len(X)
    if X2 is None:
        N2 = N
    else:
        N2 = len(X2)

    if name in f.keys():
        print("Skipping {} (group exists)".format(name))
        return

    batch_size = n_max
    same = X2 is None
    out = create_h5py_dataset(f, batch_size, name, diag, N, N2)
    X_loader = DataLoader(X, batch_size=batch_size)
    X2_loader = DataLoader(X if same else X2, batch_size=batch_size)

    print("Computing {}".format(name))
    N_batches = (N + batch_size - 1) // batch_size
    if diag:
        # compute_Kdiag is usually very cheap, don't bother splitting it
        if worker_rank == 0:
            compute_Kdiag(out, kern, X_loader, X2_loader, same, N_batches,
                          batch_size)
    else:
        compute_K(out, kern, X_loader, X2_loader, same, N_batches, batch_size,
                  worker_rank)
