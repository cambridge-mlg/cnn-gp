import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .kernel_patch import ConvKP, NonlinKP
import math


__all__ = ("NNGPKernel", "Conv2d", "ReLU", "Sequential", "Mixture",
           "MixtureModule", "Sum", "SumModule", "resnet_block")


class NNGPKernel(nn.Module):
    """
    Transforms one kernel matrix into another.
    [N1, N2, W, H] -> [N1, N2, W, H]
    """
    def forward(self, x, y=None, same=None, diag=False):
        """
        Either takes one minibatch (x), or takes two minibatches (x and y), and
        a boolean indicating whether they're the same.
        """
        if y is None:
            assert same is None
            y = x
            same = True

        assert not diag or len(x) == len(y), (
            "diagonal kernels must operate with data of equal length")

        assert 4==len(x.size())
        assert 4==len(y.size())
        assert x.size(1) == y.size(1)
        assert x.size(2) == y.size(2)
        assert x.size(3) == y.size(3)

        N1 = x.size(0)
        N2 = y.size(0)
        C = x.size(1)
        W = x.size(2)
        H = x.size(3)

        # [N1, C, W, H], [N2, C, W, H] -> [N1 N2, 1, W, H]
        if diag:
            xy = (x*y).mean(1, keepdim=True)
        else:
            xy = (x.unsqueeze(1)*y).mean(2).view(N1*N2, 1, W, H)
        xx = (x**2).mean(1, keepdim=True)
        yy = (y**2).mean(1, keepdim=True)

        initial_kp = ConvKP(same, diag, xy, xx, yy)
        final_kp = self.propagate(initial_kp)
        r = NonlinKP(final_kp).xy.squeeze(-1).squeeze(-1)
        if diag:
            return r.squeeze(-1)
        else:
            return r


class Conv2d(NNGPKernel):
    def __init__(self, kernel_size, stride=1, padding="same", dilation=1, bias=None, in_channel_multiplier=1, out_channel_multiplier=1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        if padding == "same":
            assert 1 == kernel_size % 2
            self.padding = dilation*((kernel_size-1)//2)
        else:
            self.padding = padding
        self.dilation = dilation
        self.register_buffer('kernel', t.ones(1, 1, self.kernel_size, self.kernel_size) / self.kernel_size**2)
        self.in_channel_multiplier, self.out_channel_multiplier = in_channel_multiplier, out_channel_multiplier

    def propagate(self, kp):
        #Avoid decreasing the variance where there is padding.
        #square = t.ones(1, 1, kp.W, kp.H).to(self.kernel.device)
        #conv_square = F.conv2d(square, self.kernel, stride=self.stride, padding=self.padding, dilation=self.dilation)#.mean()
        kp = ConvKP(kp)
        xy = F.conv2d(kp.xy, self.kernel, stride=self.stride, padding=self.padding, dilation=self.dilation) #/ conv_square
        xx = F.conv2d(kp.xx, self.kernel, stride=self.stride, padding=self.padding, dilation=self.dilation) #/ conv_square
        yy = F.conv2d(kp.yy, self.kernel, stride=self.stride, padding=self.padding, dilation=self.dilation) #/ conv_square
        return ConvKP(kp.same, kp.diag, xy, xx, yy)

    def nn(self, channels, in_channels=None, out_channels=None):

        if in_channels is None:
            in_channels = channels
        if out_channels is None:
            out_channels = channels
        #return Conv2dModule(self, channels, in_channels=in_channels, out_channels=out_channels)
        conv2d = nn.Conv2d(
            in_channels = in_channels * self.in_channel_multiplier,
            out_channels = out_channels * self.out_channel_multiplier,
            kernel_size = self.kernel_size,
            stride = self.stride,
            padding=self.padding,
            dilation = self.dilation,
            bias=False
        )
        conv2d.weight.data.normal_(0, 1/self.kernel_size/math.sqrt(conv2d.in_channels))
        return conv2d

    def layers(self):
        return 1


class ReLU(NNGPKernel):
    """
    Strategy for improving numerical stability:
      Add IID noise
      Raises the problem of what to do with the diagonal of the covariance for identical elements.
      Strategy: add noise to kp.xx and kp.yy, but not kp.xy
      Run through usual computation: everything stays finite
      Replace diagonal elements of kp.xy with correct values.
      To represent identical elements, use flag in kp
    """
    iid_noise_var = 1e-5
    def propagate(self, kp):
        kp = NonlinKP(kp)

        xx = kp.xx + self.iid_noise_var
        yy = kp.yy + self.iid_noise_var
        xx_yy = xx*yy

        """
        We need to calculate (xy, xx, yy == c, v₁, v₂):
                      ⏤⏤⏤⏤⏤⏤⏤⏤⏤⏤⏤⏤⏤⏤⏤⏤⏤⏤
        √(v₁v₂) / 2π ⎷1 - c²/v₁v₂ + (π - θ)c / √(v₁v₂)

        which is equivalent to:
        1/2π ( √(v₁v₂ - c²) + (π - θ)c )

        # NOTE we divide by 2 to avoid multiplying the ReLU by sqrt(2)
        """
        cos_theta = kp.xy * xx_yy.rsqrt()
        sin_theta = t.sqrt(xx_yy - kp.xy**2)
        theta = t.acos(cos_theta)
        xy = (sin_theta + (math.pi - theta)*kp.xy) / (2*math.pi)
        xx = xx/2.
        if kp.same:
            yy = xx
            if kp.diag:
                xy = xx
            else:
                eye = t.eye(xy.size()[0]).unsqueeze(-1).unsqueeze(-1).to(kp.xy.device)
                xy = (1-eye)*xy + eye*xx
        else:
            yy = yy/2.
        return NonlinKP(kp.same, kp.diag, xy, xx, yy)

    def nn(self, channels, in_channels=None, out_channels=None):
        assert in_channels is None
        assert out_channels is None
        return nn.ReLU()
        # return ReLUModule(self.iid_noise_var)
    def layers(self):
        return 0

class ReLUModule(nn.Module):
    def __init__(self, iid_noise_var):
        super().__init__()
        self.iid_noise_var = iid_noise_var
    def forward(self, input):
        # do not do the noise
        # +math.sqrt(self.iid_noise_var)*t.rand(input.size(), device=input.device))
        return F.relu(input)


#### Combination classes

class Sequential(NNGPKernel):
    def __init__(self, *mods):
        super().__init__()
        self.mods = mods
        for idx, mod in enumerate(mods):
            self.add_module(str(idx), mod)
    def propagate(self, kp):
        for mod in self.mods:
            kp = mod.propagate(kp)
        return kp
    def nn(self, channels, in_channels=None, out_channels=None):
        if len(self.mods) == 0:
            return nn.Sequential()
        elif len(self.mods) == 1:
            return self.mods[0].nn(channels, in_channels=in_channels, out_channels=out_channels)
        else:
            return nn.Sequential(
                self.mods[0].nn(channels, in_channels=in_channels),
                *[mod.nn(channels) for mod in self.mods[1:-1]],
                self.mods[-1].nn(channels, out_channels=out_channels)
            )
    def layers(self):
        return sum(mod.layers() for mod in self.mods)


class Mixture(NNGPKernel):
    """
    Applys multiple modules to the input, and sums the result
    (e.g. for the implementation of a ResNet).

    Parameterised by proportion of each module (proportions add
    up to one, such that, if each model has average variance 1,
    then the output will also have average variance 1.
    """
    def __init__(self, mods, logit_proportions=None):
        super().__init__()
        self.mods = mods
        for idx, mod in enumerate(mods):
            self.add_module(str(idx), mod)
        if logit_proportions is None:
            logit_proportions = t.zeros(len(mods))
        self.logit = nn.Parameter(logit_proportions)
    def propagate(self, kp):
        proportions = F.softmax(self.logit, dim=0)
        total = self.mods[0].propagate(kp) * proportions[0]
        for i in range(1, len(self.mods)):
            total = total + (self.mods[i].propagate(kp) * proportions[i])
        return total
    def nn(self, channels, in_channels=None, out_channels=None):
        return MixtureModule([mod.nn(channels, in_channels=in_channels, out_channels=out_channels) for mod in self.mods], self.logit)
    def layers(self):
        return max(mod.layers() for mod in self.mods)

class MixtureModule(nn.Module):
    def __init__(self, mods, logit_parameter):
        super().__init__()
        self.mods = mods
        self.logit = t.tensor(logit_parameter)
        for idx, mod in enumerate(mods):
            self.add_module(str(idx), mod)
    def forward(self, input):
        sqrt_proportions = F.softmax(self.logit, dim=0).sqrt()
        total = self.mods[0](input)*sqrt_proportions[0]
        for i in range(1, len(self.mods)):
            total = total + self.mods[i](input) # *sqrt_proportions[i]
        return total


class Sum(NNGPKernel):
    def __init__(self, mods):
        super().__init__()
        self.mods = mods
        for idx, mod in enumerate(mods):
            self.add_module(str(idx), mod)
    def propagate(self, kp):
        # This adds 0 to the first kp, hopefully that's a noop
        return sum(m.propagate(kp) for m in self.mods)
    def nn(self, channels, in_channels=None, out_channels=None):
        return SumModule([
            mod.nn(channels, in_channels=in_channels, out_channels=out_channels)
            for mod in self.mods])
    def layers(self):
        return max(mod.layers() for mod in self.mods)


class SumModule(nn.Module):
    def __init__(self, mods):
        super().__init__()
        self.mods = mods
        for idx, mod in enumerate(mods):
            self.add_module(str(idx), mod)
    def forward(self, input):
        # This adds 0 to the first value, hopefully that's a noop
        return sum(m(input) for m in self.mods)


def resnet_block(stride=1, projection_shortcut=False, multiplier=1):
    if stride == 1 and not projection_shortcut:
        return Sum([
            Sequential(),
            Sequential(
                ReLU(),
                Conv2d(3, stride=stride, in_channel_multiplier=multiplier, out_channel_multiplier=multiplier),
                ReLU(),
                Conv2d(3, in_channel_multiplier=multiplier, out_channel_multiplier=multiplier),
            )
        ])
    else:
        return Sequential(
            ReLU(),
            Sum([
                Conv2d(1, stride=stride, in_channel_multiplier=multiplier//stride, out_channel_multiplier=multiplier),
                Sequential(
                    Conv2d(3, stride=stride, in_channel_multiplier=multiplier//stride, out_channel_multiplier=multiplier),
                    ReLU(),
                    Conv2d(3, in_channel_multiplier=multiplier, out_channel_multiplier=multiplier),
                )
            ]),
        )
