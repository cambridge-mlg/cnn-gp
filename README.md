# "Deep CNNs as shallow GPs" in Pytorch
Code for "Deep Convolutional Networks as shallow Gaussian Processes"
([arXiv](https://arxiv.org/abs/1808.05587),
[other material](https://agarri.ga/publication/convnets-as-gps/)), by
[Adrià Garriga-Alonso](https://agarri.ga/),
[Laurence Aitchison](http://www.gatsby.ucl.ac.uk/~laurence/) and
[Carl Edward Rasmussen](http://mlg.eng.cam.ac.uk/carl/).

The most extensively used libraries
are [PyTorch](https://pytorch.org/), [NumPy](https://www.numpy.org/) and
[H5Py](http://www.h5py.org/), check `requirements.txt` for the rest.

# The `cnn_gp` package
This library allows you to very easily write down neural network architectures,
and get the kernels corresponding to their equivalent GPs. We can easily build
`Sequential` architectures. For example, a 3-layer convolutional network with a
dense layer at the end is:

```python
from cnn_gp import Sequential, Conv2d, ReLU

model = Sequential(
    Conv2d(kernel_size=3),
    ReLU(),
    Conv2d(kernel_size=3, stride=2),
    ReLU(),
    Conv2d(kernel_size=14, padding=0),  # equivalent to a dense layer
)
```
Optionally call `model = model.cuda()` to use the GPU.

Then, we can compute the kernel between batches of input images:
```python
import torch
# X and Z have shape [N_images, N_channels, img_width, img_height]
X = torch.randn(2, 3, 28, 28)
Z = torch.randn(2, 3, 28, 28)

Kxx = model(X)
Kxx = model(X, X, same=True)

Kxz = model(X, Z)

# diagonal of Kxx matrix above
Kxx_diag = model(X, diag=True)
```

We can also instantiate randomly initialized neural networks that have the
architecture corresponding to the kernel.
```python
network = model.nn(channels=16, in_channels=3, out_channels=10)
isinstance(network, torch.nn.Module)  # evaluates to True

f_X = network(X)  # evaluates network at X
```
Calling `model.nn` will give us an instance of the network above that can do 10-class
classification. It accepts inputs that are RGB images (3 channels) of size
28x28. We can then train this neural network as we would any normal Pytorch
model.

## Installation
It is possible to install the `cnn_gp` package without any of the dependencies
that are needed for the experiments. Just run
```sh
pip install -e .
```
from the root directory of this same repository.

## Current limitations
Dense layers are not implemented. The way to simulate them is to have a
convolutional layer with `padding=0`, and with `kernel_size` as large as the
activations in the previous layer.

# Replicating the experiments

First install the packages in `requirements.txt`. To run each of the
experiments, first take a look at the files `exp_mnist_resnet/run.bash` or
`exp_random_nn/run.bash`. Edit the configuration variables near the top
appropriately. Then, run one of the files from the root of the directory, for
example:

```bash
bash ./exp_mnist_resnet/run.bash
```

## Experiment 1: classify MNIST

Here are the test errors for the best GPs corresponding to the NN architectures
reported in the paper.

 Name in paper | Config file | Validation error | Test error
 --------------|-------------|------------------|----------
ConvNet GP | `mnist_paper_convnet_gp` | 0.71% | 1.03%
Residual CNN GP | `mnist_paper_residual_cnn_gp` | 0.72% | 0.96%
ResNet GP | `mnist_as_tf` | 0.68% | 0.84%

<details>
  <summary>(click to expand) Architecture for ConvNet GP</summary>

  ```python
  var_bias = 7.86
  var_weight = 2.79

  initial_model = Sequential(
      Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
      ReLU(),
      Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
      ReLU(),
      Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
      ReLU(),
      Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
      ReLU(),
      Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
      ReLU(),
      Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
      ReLU(),
      Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2, var_bias=var_bias),
      ReLU(),  # Total 7 layers before dense

      Conv2d(kernel_size=28, padding=0, var_weight=var_weight, var_bias=var_bias),
  ```
</details>
<details>
  <summary>(click to expand) Architecture for Residual CNN GP</summary>

  ```python
  var_bias = 4.69
  var_weight = 7.27
  initial_model = Sequential(
      *(Sum([
          Sequential(),
          Sequential(
              Conv2d(kernel_size=4, padding="same", var_weight=var_weight * 4**2,
                  var_bias=var_bias),
              ReLU(),
          )]) for _ in range(8)),
      Conv2d(kernel_size=4, padding="same", var_weight=var_weight * 4**2,
          var_bias=var_bias),
      ReLU(),
      Conv2d(kernel_size=28, padding=0, var_weight=var_weight,
          var_bias=var_bias),
  )
  ```
</details>

<details>
  <summary>(click to expand) Architecture for ResNet GP</summary>

  ```python
  initial_model = Sequential(
      Conv2d(kernel_size=3),

      # Big resnet block #1
      resnet_block(stride=1, projection_shortcut=True,  multiplier=1),
      resnet_block(stride=1, projection_shortcut=False, multiplier=1),
      resnet_block(stride=1, projection_shortcut=False, multiplier=1),
      resnet_block(stride=1, projection_shortcut=False, multiplier=1),
      resnet_block(stride=1, projection_shortcut=False, multiplier=1),

      # Big resnet block #2
      resnet_block(stride=2, projection_shortcut=True,  multiplier=2),
      resnet_block(stride=1, projection_shortcut=False, multiplier=2),
      resnet_block(stride=1, projection_shortcut=False, multiplier=2),
      resnet_block(stride=1, projection_shortcut=False, multiplier=2),
      resnet_block(stride=1, projection_shortcut=False, multiplier=2),

      # Big resnet block #3
      resnet_block(stride=2, projection_shortcut=True,  multiplier=4),
      resnet_block(stride=1, projection_shortcut=False, multiplier=4),
      resnet_block(stride=1, projection_shortcut=False, multiplier=4),
      resnet_block(stride=1, projection_shortcut=False, multiplier=4),
      resnet_block(stride=1, projection_shortcut=False, multiplier=4),

      # No nonlinearity here, the next Conv2d substitutes the average pooling
      Conv2d(kernel_size=7, padding=0, in_channel_multiplier=4,
             out_channel_multiplier=4),
      ReLU(),
      Conv2d(kernel_size=1, padding=0, in_channel_multiplier=4),
  )
  ```
</details>

## Experiment 2: Check that networks converge to a Gaussian process
In the paper, only ResNet-32 GP is presented. This is why an issue when
constructing the Residual CNN GP was originally not caught. More details in the
relevant subsection.
### ResNet-32 GP
![Resnet-32 GP](/exp_random_nn/mnist_as_tf/figure.png)
### ConvNet GP
![Resnet-32 GP](/exp_random_nn/mnist_paper_convnet_gp/figure.png)
### Residual CNN GP
The best randomly-searched ResNet reported in the paper.

In the original paper there is slight issue with how the kernels relate to the
underlying networks.  The network sums together layers after the ReLU nonlinearity,
which are not Gaussian, and also do not have mean 0. However, the kernel is valid
and does correspond to a neural network.  In particular, if we take an infinite
1x1 convolution, after each relu layer, 
we convert the output of the ReLU's into a zero-mean Gaussian,
with the same kernel, which can be summed.
In the interest of making the results replicable, we have replicated this issue
as well.

The correct way to use ResNets is to sum things after a Conv2d layer, see for
example the `resnet_block` in [`cnn_gp/kernels.py`](/cnn_gp/kernels.py).

![Resnet-32 GP](/exp_random_nn/mnist_paper_residual_cnn_gp/figure.png)


# BibTex citation record
Note: the version in arXiv is slightly newer and contains information about
which hyperparameters turned out to be the most effective for each architecture.

```bibtex
@inproceedings{aga2018cnngp,
  author    = {{Garriga-Alonso}, Adri{\`a} and Aitchison, Laurence and Rasmussen, Carl Edward},
  title     = {Deep Convolutional Networks as shallow {G}aussian Processes},
  booktitle = {International Conference on Learning Representations},
  year      = {2019},
  url       = {https://openreview.net/forum?id=Bklfsi0cKm}}
```
