"""
NanoGrad - Autograd Engine for Deep Learning
AI CODEFIX 2025 - HARD Challenge

Fixed version of the autograd engine.
"""

import numpy as np
from typing import Set, List, Callable, Tuple


class Value:
    """
    Stores a single scalar value and its gradient.
    """

    def __init__(self, data: float, _children: Tuple['Value', ...] = (), _op: str = ''):
        self.data = float(data)
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other: 'Value') -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: 'Value') -> 'Value':
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # d/dx(x*y) = y  ;  d/dy(x*y) = x
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data

        out._backward = _backward
        return out

    def __pow__(self, other: float) -> 'Value':
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            # d/dx(x^n) = n * x^(n-1)
            self.grad += out.grad * (other * (self.data ** (other - 1)))

        out._backward = _backward
        return out

    def relu(self) -> 'Value':
        out = Value(max(0, self.data), (self,), 'ReLU')

        def _backward():
            # Use >= convention (gradient 1 for x>0; at x==0 we treat as 1 here)
            self.grad += out.grad * (1.0 if self.data >= 0 else 0.0)

        out._backward = _backward
        return out

    def __neg__(self) -> 'Value':
        return self * -1

    def __sub__(self, other: 'Value') -> 'Value':
        return self + (-other)

    def __truediv__(self, other: 'Value') -> 'Value':
        return self * (other ** -1)

    def __radd__(self, other: float) -> 'Value':
        return self + other

    def __rmul__(self, other: float) -> 'Value':
        return self * other

    def __rsub__(self, other: float) -> 'Value':
        return Value(other) - self

    def __rtruediv__(self, other: float) -> 'Value':
        return Value(other) / self

    def backward(self) -> None:
        topo: List[Value] = []
        visited: Set[Value] = set()

        def build_topo(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def zero_grad(self) -> None:
        self.grad = 0.0


def topological_sort(root: Value) -> List[Value]:
    topo: List[Value] = []
    visited: Set[Value] = set()

    def dfs(v: Value) -> None:
        if v in visited:
            return
        visited.add(v)
        for child in v._prev:
            dfs(child)
        topo.append(v)

    dfs(root)
    return topo


def cached_backward(values: List[Value]) -> None:
    for v in values:
        v._backward()


class Neuron:
    """A single neuron with weighted inputs and bias."""

    def __init__(self, nin: int):
        self.w = [Value(np.random.randn()) for _ in range(nin)]
        self.b = Value(0.01)     # keeps neuron active at initialization


    def __call__(self, x: List[Value]) -> Value:
        """Forward pass through neuron."""
        # w · x + b (explicit accumulation to ensure correct graph links)
        act = self.b
        for wi, xi in zip(self.w, x):
            act = act + (wi * xi)
        return act.relu()

    def parameters(self) -> List[Value]:
        return self.w + [self.b]


class Layer:
    """A layer of neurons."""

    def __init__(self, nin: int, nout: int):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x: List[Value]) -> List[Value]:
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self) -> List[Value]:
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    """Multi-Layer Perceptron (simple neural network)."""

    def __init__(self, nin: int, nouts: List[int]):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x: List[Value]) -> Value:
        """Forward pass through network."""
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self) -> List[Value]:
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = 0.0


def train_step(model: MLP, xs: List[List[Value]], ys: List[Value], lr: float = 0.01) -> float:
    # Forward pass
    ypred = [model(x) for x in xs]
    # Compute MSE loss
    loss = sum((yp - yt)**2 for yp, yt in zip(ypred, ys))
    # Zero gradients before backward to avoid accumulation
    model.zero_grad()
    # Backward pass
    loss.backward()
    # Update parameters
    for p in model.parameters():
        p.data -= lr * p.grad
    return loss.data


def numerical_gradient(f: Callable[[float], float], x: float, h: float = 1e-5) -> float:
    return (f(x + h) - f(x - h)) / (2 * h)


def validate_graph(root: Value) -> bool:
    visited = set()
    rec_stack = set()

    def has_cycle(v: Value) -> bool:
        visited.add(v)
        rec_stack.add(v)
        for child in v._prev:
            if child not in visited:
                if has_cycle(child):
                    return True
            elif child in rec_stack:
                return True
        rec_stack.remove(v)
        return False

    return not has_cycle(root)


def safe_div(a: Value, b: Value, epsilon: float = 1e-10) -> Value:
    return a / Value(b.data + epsilon)


if __name__ == "__main__":
    print("=" * 60)
    print("NanoGrad - Autograd Engine Test")
    print("=" * 60)

    # Simple test
    print("\n--- Test 1: Basic Operations ---")
    x = Value(2.0)
    y = Value(3.0)
    z = x * y + x**2
    z.backward()

    print(f"x = {x.data}, y = {y.data}")
    print(f"z = x*y + x^2 = {z.data}")
    print(f"dz/dx = {x.grad} (expected: y + 2*x = 3 + 2*2 = 7)")
    print(f"dz/dy = {y.grad} (expected: x = 2)")

    # Test neural network
    print("\n--- Test 2: Small Neural Network ---")
    model = MLP(2, [4, 1])

    # Simple training data: XOR problem
    xs = [
        [Value(0.0), Value(0.0)],
        [Value(0.0), Value(1.0)],
        [Value(1.0), Value(0.0)],
        [Value(1.0), Value(1.0)],
    ]
    ys = [Value(0.0), Value(1.0), Value(1.0), Value(0.0)]

    print("Training for 10 steps...")
    for i in range(10):
        loss = train_step(model, xs, ys, lr=0.01)
        if i % 5 == 0:
            print(f"Step {i}: loss = {loss:.4f}")

    print("\n" + "=" * 60)
    print("This version has fixes applied so gradients and training behave correctly.")
    print("=" * 60)
