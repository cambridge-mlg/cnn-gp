"""
Relatively inefficient way to merge the results of the workers
"""
import h5py
import sys
from tqdm import tqdm
import numpy as np

if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} dest_file [source_file1 source_file2 ...]")
    sys.exit(1)

_, dest_file, *src_files = sys.argv

with h5py.File(dest_file, "a") as dest_f:
    for path in tqdm(src_files):
        with h5py.File(path, "r") as src_f:
            valid_keys = [k
                          for k in dest_f.keys()
                          if k in src_f.keys()]
            for k in tqdm(valid_keys):
                dest_data = dest_f[k]
                src_data = src_f[k]
                for i in tqdm(range(len(dest_data))):
                    src = src_data[i, ...]
                    dest = dest_data[i, ...]
                    dest_is_nan = np.isnan(dest)
                    dest[dest_is_nan] = src[dest_is_nan]

                    dest_data[i, ...] = dest
