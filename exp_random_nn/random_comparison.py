import os
import torch
from tqdm import tqdm
import importlib
import numpy as np
import pandas as pd
import absl.app

from cnn_gp import DatasetFromConfig
FLAGS = absl.app.flags.FLAGS


def main(_):
    torch.manual_seed(FLAGS.seed)

    config = importlib.import_module(f"configs.{FLAGS.config}")
    dataset = DatasetFromConfig(FLAGS.datasets_path, config)
    dl = torch.utils.data.DataLoader(dataset.train, batch_size=100)
    inputs, _ = next(iter(dl))
    inputs = inputs.cuda()
    model = config.initial_model.cuda()

    results = []
    r0 = []

    with torch.no_grad():
        true_cov = model(inputs).cpu().numpy()

    with torch.no_grad():
        for _ in tqdm(range(FLAGS.n_samples)):
            nn = model.nn(FLAGS.channels, in_channels=config.in_channels,
                          out_channels=1).cuda()
            results.append(nn(inputs)[:, 0, 0, 0].cpu().numpy())
            r0.append(results[-1][0])
            del nn

    samples_output_filename = os.path.join(
        FLAGS.out_path,
        f"{FLAGS.channels:04d}_{FLAGS.seed:04d}_samples.csv")
    pd.DataFrame({
        'r0': np.array(r0) / np.sqrt(true_cov[0, 0]),
    }).to_csv(samples_output_filename, index=False)

    Ni = inputs.shape[0]
    i = np.arange(Ni) * np.ones([Ni, 1])
    j = i.T
    R = np.vstack(results)
    est_cov = R.T @ R / FLAGS.n_samples

    cov_output_filename = os.path.join(
        FLAGS.out_path,
        f"{FLAGS.channels:04d}_{FLAGS.seed:04d}_cov.csv")
    pd.DataFrame({
        'i': i.ravel(),
        'j': j.ravel(),
        'est': est_cov.ravel(),
        'true': true_cov.ravel()
    }).to_csv(cov_output_filename, index=False)

if __name__ == '__main__':
    f = absl.app.flags
    f.DEFINE_string("datasets_path", "/scratch/ag919/datasets/",
                    "where to save datasets")
    f.DEFINE_string("out_path", None,
                    "where to save the drawn outputs of the NN and kernel")
    f.DEFINE_enum("config", "cifar10", ["mnist", "cifar10", "mnist_as_tf"], "which config to load from `configs`")
    f.DEFINE_integer("seed", 1, "the random seed")
    f.DEFINE_integer("channels", 30, "the number of channels of the random finite NNs")
    f.DEFINE_integer("n_samples", 10000, "Number of samples to draw from the NN")
    absl.app.run(main)
