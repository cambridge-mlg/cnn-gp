# cnn_gp for Pytorch
## Installing the package
It is possible to install the `cnn_gp` package without any of the dependencies
that are needed just for the experiments. Just run
```sh
pip install -e .
```
from the root directory of this same repository

## Running the experiments

First install the `requirements.txt` file. If you are going to use several GPUs, please run:
```sh
HDF5_MPI="ON" pip install -r requirements.txt
```

### Results of `./exp_mnist_resnet/run.bash`
Using `config=mnist_as_tf`, the results are 99.31% validation accuracy and 99.12% test accuracy.
