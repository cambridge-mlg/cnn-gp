import torchvision
from cnn_gp import Conv2d, ReLU, Sequential

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

var_bias = 7.86
var_weight = 2.79

layers = []
for _ in range(7):  # n_layers
    layers += [
        Conv2d(kernel_size=7, padding="same", var_weight=var_weight * 7**2,
               var_bias=var_bias),
        ReLU(),
    ]
initial_model = Sequential(
    *layers,
    Conv2d(kernel_size=28, padding=0, var_weight=var_weight,
           var_bias=var_bias),
)
