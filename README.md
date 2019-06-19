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
    Conv2d(kernel_size=3),   # 26x26
	ReLU(),
    Conv2d(kernel_size=3, stride=2), # 12x12
	ReLU(),
	Conv2d(kernel_size=12),
)
```

## Setup
### Installing the package
It is possible to install the `cnn_gp` package without any of the dependencies
that are needed for the experiments. This allows you to build your own Neural netowrkJust run
```sh
pip install -e .
```
from the root directory of this same repository

## Running the experiments

First install the packages in `requirements.txt`.

### Accuracy of the best performing networks in the paper
<details>
  <summary>(click for details) Using `config=mnist_paper_convnet_gp`, this is the
  "ConvNet GP" network in the paper. 0.71% validation error, 1.03% test error.
  </summary>
  ```python
  var_bias = 7.86
  var_weight = 2.79

  initial_model = Sequential(
	  Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2,
			  var_bias=var_bias),
	  ClampingReLU(),
	  Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2,
			  var_bias=var_bias),
	  ClampingReLU(),
	  Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2,
			  var_bias=var_bias),
	  ClampingReLU(),
	  Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2,
			  var_bias=var_bias),
	  ClampingReLU(),
	  Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2,
			  var_bias=var_bias),
	  ClampingReLU(),
	  Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2,
			  var_bias=var_bias),
	  ClampingReLU(),
	  Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2,
			  var_bias=var_bias),
	  ClampingReLU(),  # Total 7 layers

	  Conv2d(kernel_size=28, padding=0, var_weight=var_weight,
			var_bias=var_bias),
  ```
</details>
<details>
  <summary>(click for details) Using `config=mnist_as_tf`, this is the
  "ResNet GP" network in the paper corresponding to a 32-layer ResNet. 0.69% validation error, 0.88% test error.
  </summary>
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

## BibTex citation record
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
