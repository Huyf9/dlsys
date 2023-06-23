"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features))
        self.bias = Parameter(ops.transpose(init.kaiming_uniform(fan_in=out_features, fan_out=1))) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = ops.matmul(X, self.weight)
        if self.bias:
            out += self.bias
        return out
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        batch = X.shape[0]
        return X.reshape((batch, -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        num_batch, num_samples = logits.shape
        y_one_hot = init.one_hot(num_samples, y)
        logits_mask = ops.summation(logits * y_one_hot, axes=1)
        out = ops.log(ops.summation(ops.exp(logits), axes=1)) - logits_mask
        return ops.summation(out) / num_batch
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim, requires_grad=True))
        self.bias = Parameter(init.zeros(self.dim, requires_grad=True))
        self.running_mean = init.zeros(self.dim)
        self.running_var = init.ones(self.dim)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size, num_feature = x.shape

        if not self.training:
            running_mean = self.running_mean.reshape((1, num_feature)).broadcast_to(x.shape)
            running_var = self.running_var.reshape((1, num_feature)).broadcast_to(x.shape)
            out = (x - running_mean) / (running_var + self.eps) ** 0.5
            
        else:
            mean_x = x.sum(0) / batch_size
            var_x = ((x - mean_x.reshape((1, num_feature)).broadcast_to(x.shape)) ** 2).sum(0) / batch_size

            self.running_mean = ((1 - self.momentum) * self.running_mean + self.momentum * mean_x)
            self.running_var = ((1 - self.momentum) * self.running_var + self.momentum * var_x)

            out = self.weight * (x - mean_x.reshape((1, num_feature)).broadcast_to(x.shape)) / (var_x.reshape((1, num_feature)).broadcast_to(x.shape) + self.eps) ** 0.5 + self.bias

        return out
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        assert self.dim == x.shape[-1], f"Error dimension: input dim is {x.shape[-1]} but got {self.dim}"
        batch_size, num_feature = x.shape
        mean_x = (x.sum(axes=-1).reshape((batch_size, 1)) / num_feature).broadcast_to(x.shape)
        var_x = (((x - mean_x) ** 2).sum(axes=-1).reshape((batch_size, 1)) / num_feature).broadcast_to(x.shape)
        out = self.weight * (x - mean_x) / ((var_x + self.eps) ** 0.5) + self.bias
        return out
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            drop_mask = init.randb(*x.shape, p=1 - self.p)
            x = drop_mask * x / (1 - self.p)
        return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION



