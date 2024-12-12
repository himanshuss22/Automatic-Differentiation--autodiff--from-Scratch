import numpy as np
from collections import deque
try:
    import cupy as xp
    from cupyx.scipy import ndimage
    is_gpu = True
except ImportError:
    import numpy as xp
    from scipy import ndimage
    is_gpu = False

class Variable:
    def __init__(self, value, parents=None, op=None):
        self.value = np.array(value, dtype=float)
        self.grad = None
        self.parents = parents if parents else []
        self.op = op
        self._grad_fn = None  # Function to compute gradient w.r.t this variable
        self._visited = False  # For topological sort

    def __repr__(self):
        return f"Variable(value={self.value}, grad={self.grad})"

    # Efficient backward pass
    def backward(self, grad=None):
        if grad is None:
            if self.value.shape == ():
                grad = np.array(1.0)
            else:
                grad = np.ones_like(self.value)
        self.grad = grad

        # Topological sort
        topo_order = []
        visited = set()

        def build_topo(variable):
            if variable not in visited:
                visited.add(variable)
                for parent in variable.parents:
                    build_topo(parent)
                topo_order.append(variable)

        build_topo(self)

        # Reverse the topological order for backward pass
        for var in reversed(topo_order):
            if var._grad_fn is not None:
                grads = var._grad_fn(var.grad)
                for parent, parent_grad in zip(var.parents, grads):
                    if parent.grad is None:
                        parent.grad = parent_grad
                    else:
                        parent.grad += parent_grad

    # Overloaded operations
    def __add__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.value + other.value, parents=[self, other], op='+')

        def grad_fn(grad):
            return [grad, grad]

        out._grad_fn = grad_fn
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.value - other.value, parents=[self, other], op='-')

        def grad_fn(grad):
            return [grad, -grad]

        out._grad_fn = grad_fn
        return out

    def __rsub__(self, other):
        return Variable(other) - self

    def __mul__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.value * other.value, parents=[self, other], op='*')

        def grad_fn(grad):
            return [grad * other.value, grad * self.value]

        out._grad_fn = grad_fn
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Variable) else Variable(other)
        out = Variable(self.value / other.value, parents=[self, other], op='/')

        def grad_fn(grad):
            return [grad / other.value, -grad * self.value / (other.value ** 2)]

        out._grad_fn = grad_fn
        return out

    def __rtruediv__(self, other):
        return Variable(other) / self

    def __pow__(self, exponent):
        exponent = exponent if isinstance(exponent, Variable) else Variable(exponent)
        out = Variable(self.value ** exponent.value, parents=[self, exponent], op='**')

        def grad_fn(grad):
            base_grad = grad * exponent.value * (self.value ** (exponent.value - 1))
            exp_grad = grad * self.value ** exponent.value * np.log(self.value)
            return [base_grad, exp_grad]

        out._grad_fn = grad_fn
        return out

    def __neg__(self):
        return self * -1

    # Mathematical functions
    @staticmethod
    def matmul(a, b):
        a = a if isinstance(a, Variable) else Variable(a)
        b = b if isinstance(b, Variable) else Variable(b)
        value = np.dot(a.value, b.value)
        out = Variable(value, parents=[a, b], op='matmul')

        def grad_fn(grad):
            grad_a = np.dot(grad, b.value.T)
            grad_b = np.dot(a.value.T, grad)
            return [grad_a, grad_b]

        out._grad_fn = grad_fn
        return out

    @staticmethod
    def sin(x):
        x = x if isinstance(x, Variable) else Variable(x)
        value = np.sin(x.value)
        out = Variable(value, parents=[x], op='sin')

        def grad_fn(grad):
            grad_x = grad * np.cos(x.value)
            return [grad_x]

        out._grad_fn = grad_fn
        return out

    @staticmethod
    def cos(x):
        x = x if isinstance(x, Variable) else Variable(x)
        value = np.cos(x.value)
        out = Variable(value, parents=[x], op='cos')

        def grad_fn(grad):
            grad_x = -grad * np.sin(x.value)
            return [grad_x]

        out._grad_fn = grad_fn
        return out

    @staticmethod
    def tanh(x):
        x = x if isinstance(x, Variable) else Variable(x)
        value = np.tanh(x.value)
        out = Variable(value, parents=[x], op='tanh')

        def grad_fn(grad):
            grad_x = grad * (1 - value ** 2)
            return [grad_x]

        out._grad_fn = grad_fn
        return out

    @staticmethod
    def exp(x):
        x = x if isinstance(x, Variable) else Variable(x)
        exp_value = np.exp(x.value)
        if np.any(np.isinf(exp_value)):
            raise OverflowError("Overflow encountered in Variable.exp")
        out = Variable(exp_value, parents=[x], op='exp')

        def grad_fn(grad):
            grad_x = grad * exp_value
            return [grad_x]

        out._grad_fn = grad_fn
        return out

    @staticmethod
    def log(x):
        x = x if isinstance(x, Variable) else Variable(x)
        if np.any(x.value <= 0):
            raise ValueError("Logarithm undefined for non-positive values in Variable.log")
        value = np.log(x.value)
        out = Variable(value, parents=[x], op='log')

        def grad_fn(grad):
            grad_x = grad / x.value
            return [grad_x]

        out._grad_fn = grad_fn
        return out

    @staticmethod
    def sigmoid(x):
        x = x if isinstance(x, Variable) else Variable(x)
        sig_value = 1 / (1 + np.exp(-x.value))
        out = Variable(sig_value, parents=[x], op='sigmoid')

        def grad_fn(grad):
            grad_x = grad * sig_value * (1 - sig_value)
            return [grad_x]

        out._grad_fn = grad_fn
        return out

    @staticmethod
    def relu(x):
        x = x if isinstance(x, Variable) else Variable(x)
        relu_value = np.maximum(0, x.value)
        out = Variable(relu_value, parents=[x], op='relu')

        def grad_fn(grad):
            grad_x = grad * (x.value > 0).astype(float)
            return [grad_x]

        out._grad_fn = grad_fn
        return out

    @staticmethod
    def sinh(x):
        x = x if isinstance(x, Variable) else Variable(x)
        value = np.sinh(x.value)
        out = Variable(value, parents=[x], op='sinh')

        def grad_fn(grad):
            grad_x = grad * np.cosh(x.value)
            return [grad_x]

        out._grad_fn = grad_fn
        return out

    @staticmethod
    def cosh(x):
        x = x if isinstance(x, Variable) else Variable(x)
        value = np.cosh(x.value)
        out = Variable(value, parents=[x], op='cosh')

        def grad_fn(grad):
            grad_x = grad * np.sinh(x.value)
            return [grad_x]

        out._grad_fn = grad_fn
        return out

    @staticmethod
    def where(condition, x, y):
        x = x if isinstance(x, Variable) else Variable(x)
        y = y if isinstance(y, Variable) else Variable(y)
        condition = np.array(condition, dtype=bool)
        # Check for shape compatibility
        if condition.shape != x.value.shape or condition.shape != y.value.shape:
            raise ValueError("Shapes of condition, x, and y must be the same in Variable.where")
        value = np.where(condition, x.value, y.value)
        out = Variable(value, parents=[x, y], op='where')

        def grad_fn(grad):
            grad_x = grad * condition.astype(float)
            grad_y = grad * (~condition).astype(float)
            return [grad_x, grad_y]

        out._grad_fn = grad_fn
        return out

    @staticmethod
    def swish(x):
        x = x if isinstance(x, Variable) else Variable(x)
        sigmoid = 1 / (1 + np.exp(-x.value))
        value = x.value * sigmoid
        out = Variable(value, parents=[x], op='swish')

        def grad_fn(grad):
            grad_x = grad * (sigmoid + x.value * sigmoid * (1 - sigmoid))
            return [grad_x]

        out._grad_fn = grad_fn
        return out

    @staticmethod
    def mse_loss(y_pred, y_true):
        y_pred = y_pred if isinstance(y_pred, Variable) else Variable(y_pred)
        y_true = y_true if isinstance(y_true, Variable) else Variable(y_true)
        diff = y_pred - y_true
        value = np.mean(diff.value ** 2)
        out = Variable(value, parents=[y_pred], op='mse_loss')

        def grad_fn(grad):
            grad_input = (2 / y_pred.value.size) * diff.value * grad
            return [grad_input]

        out._grad_fn = grad_fn
        return out

    @staticmethod
    def cross_entropy_loss(y_pred, y_true):
        y_pred = y_pred if isinstance(y_pred, Variable) else Variable(y_pred)
        y_true = y_true if isinstance(y_true, Variable) else Variable(y_true)
        epsilon = 1e-12
        y_pred_clipped = np.clip(y_pred.value, epsilon, 1. - epsilon)
        value = -np.sum(y_true.value * np.log(y_pred_clipped)) / y_pred.value.shape[0]
        out = Variable(value, parents=[y_pred], op='cross_entropy_loss')

        def grad_fn(grad):
            grad_input = grad * (y_pred.value - y_true.value) / (y_pred.value * (1 - y_pred.value) * y_pred.value.shape[0])
            return [grad_input]

        out._grad_fn = grad_fn
        return out

    @staticmethod
    def elu(x, alpha=1.0):
        x = x if isinstance(x, Variable) else Variable(x)
        value = np.where(x.value > 0, x.value, alpha * (np.exp(x.value) - 1))
        out = Variable(value, parents=[x], op='elu')

        def grad_fn(grad):
            grad_input = grad * np.where(x.value > 0, 1, alpha * np.exp(x.value))
            return [grad_input]

        out._grad_fn = grad_fn
        return out

    @staticmethod
    def selu(x):
        x = x if isinstance(x, Variable) else Variable(x)
        scale = 1.0507
        alpha = 1.67326
        value = scale * np.where(x.value > 0, x.value, alpha * (np.exp(x.value) - 1))
        out = Variable(value, parents=[x], op='selu')

        def grad_fn(grad):
            grad_input = grad * scale * np.where(x.value > 0, 1, alpha * np.exp(x.value))
            return [grad_input]

        out._grad_fn = grad_fn
        return out

    @staticmethod
    def binary_cross_entropy_loss(y_pred, y_true):
        y_pred = y_pred if isinstance(y_pred, Variable) else Variable(y_pred)
        y_true = y_true if isinstance(y_true, Variable) else Variable(y_true)
        epsilon = 1e-12
        y_pred_clipped = np.clip(y_pred.value, epsilon, 1 - epsilon)
        value = -np.mean(y_true.value * np.log(y_pred_clipped) + (1 - y_true.value) * np.log(1 - y_pred_clipped))
        out = Variable(value, parents=[y_pred], op='binary_cross_entropy_loss')

        def grad_fn(grad):
            grad_input = grad * (y_pred.value - y_true.value) / (y_pred.value * (1 - y_pred.value) * y_pred.value.shape[0])
            return [grad_input]

        out._grad_fn = grad_fn
        return out

    @staticmethod
    def hinge_loss(y_pred, y_true):
        y_pred = y_pred if isinstance(y_pred, Variable) else Variable(y_pred)
        y_true = y_true if isinstance(y_true, Variable) else Variable(y_true)
        value = np.mean(np.maximum(0, 1 - y_true.value * y_pred.value))
        out = Variable(value, parents=[y_pred], op='hinge_loss')

        def grad_fn(grad):
            mask = (1 - y_true.value * y_pred.value) > 0
            grad_input = -grad * y_true.value * mask / y_pred.value.shape[0]
            return [grad_input]

        out._grad_fn = grad_fn
        return out

    def sum(self, axis=None, keepdims=False):
        value = np.sum(self.value, axis=axis, keepdims=keepdims)
        out = Variable(value, parents=[self], op='sum')

        def grad_fn(grad):
            grad_input = np.ones_like(self.value) * grad
            return [grad_input]

        out._grad_fn = grad_fn
        return out

    def mean(self, axis=None, keepdims=False):
        value = np.mean(self.value, axis=axis, keepdims=keepdims)
        out = Variable(value, parents=[self], op='mean')

        def grad_fn(grad):
            grad_input = np.ones_like(self.value) * grad / self.value.size
            return [grad_input]

        out._grad_fn = grad_fn
        return out

    def max(self, axis=None, keepdims=False):
        out_value = np.max(self.value, axis=axis, keepdims=keepdims)
        out = Variable(out_value, parents=[self], op='max')

        def grad_fn(grad):
            grad_input = np.zeros_like(self.value)
            max_mask = (self.value == out_value)
            grad_input[max_mask] = grad
            return [grad_input]

        out._grad_fn = grad_fn
        return out

    def min(self, axis=None, keepdims=False):
        out_value = np.min(self.value, axis=axis, keepdims=keepdims)
        out = Variable(out_value, parents=[self], op='min')

        def grad_fn(grad):
            grad_input = np.zeros_like(self.value)
            min_mask = (self.value == out_value)
            grad_input[min_mask] = grad
            return [grad_input]

        out._grad_fn = grad_fn
        return out

    # Allow functions to be called as methods
    def sin(self):
        return Variable.sin(self)

    def cos(self):
        return Variable.cos(self)

    def tanh(self):
        return Variable.tanh(self)

    def exp(self):
        return Variable.exp(self)

    def log(self):
        return Variable.log(self)

    def sigmoid(self):
        return Variable.sigmoid(self)

    def relu(self):
        return Variable.relu(self)

    def sinh(self):
        return Variable.sinh(self)

    def cosh(self):
        return Variable.cosh(self)

    def elu(self, alpha=1.0):
        return Variable.elu(self, alpha)

    def selu(self):
        return Variable.selu(self)

class Optimizer:
    def __init__(self, parameters):
        self.parameters = parameters

    def step(self):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, parameters, lr=0.01):
        super().__init__(parameters)
        self.lr = lr

    def step(self):
        for param in self.parameters:
            if param.grad is not None:
                param.value -= self.lr * param.grad

class Adam(Optimizer):
    def __init__(self, parameters, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.m = [np.zeros_like(p.value) for p in parameters]
        self.v = [np.zeros_like(p.value) for p in parameters]
        self.t = 0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                # Update biased first moment estimate
                self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * param.grad
                # Update biased second raw moment estimate
                self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * (param.grad ** 2)
                # Compute bias-corrected first moment estimate
                m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
                # Compute bias-corrected second raw moment estimate
                v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
                # Update parameters
                param.value -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

class RMSprop(Optimizer):
    def __init__(self, parameters, lr=0.001, beta=0.9, eps=1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.s = [np.zeros_like(p.value) for p in parameters]

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                # Update running average of squared gradients
                self.s[i] = self.beta * self.s[i] + (1 - self.beta) * (param.grad ** 2)
                # Update parameters
                param.value -= self.lr * param.grad / (np.sqrt(self.s[i]) + self.eps)

class Adagrad(Optimizer):
    def __init__(self, parameters, lr=0.01, eps=1e-8):
        super().__init__(parameters)
        self.lr = lr
        self.eps = eps
        self.s = [np.zeros_like(p.value) for p in parameters]

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is not None:
                # Accumulate squared gradients
                self.s[i] += param.grad ** 2
                # Update parameters
                param.value -= self.lr * param.grad / (np.sqrt(self.s[i]) + self.eps)
