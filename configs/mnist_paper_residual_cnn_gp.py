"""
The best randomly-searched ResNet reported in the paper.

In the original paper there is a bug. This network sums together layers after
the ReLU nonlinearity, which are not Gaussian, and also do not have mean 0. As
a result, the overall network does not converge to a Gaussian process. The
defined kernel is still valid, even if it doesn't correspond to a NN.

In the interest of making the results replicable, we have replicated this bug
as well.

The correct way to use ResNets is to sum things after a Conv2d layer, see for
example the `resnet_block` in `cnn_gp/kernels.py`.
"""
import torchvision
from cnn_gp import Conv2d, ReLU, Sequential, Sum

train_range = range(5000, 55000)
validation_range = list(range(55000, 60000)) + list(range(0, 5000))
test_range = range(60000, 70000)

dataset_name = "MNIST"
model_name = "ResNet"
dataset = torchvision.datasets.MNIST
transforms = []
epochs = 0
in_channels = 1
out_channels = 10

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
