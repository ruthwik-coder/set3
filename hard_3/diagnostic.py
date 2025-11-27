# diagnostic.py
from nanograd import MLP, Value
import numpy as np

np.random.seed(42)
model = MLP(2, [4, 1])

# Simple data: XOR
xs = [
    [Value(0.0), Value(0.0)],
    [Value(0.0), Value(1.0)],
    [Value(1.0), Value(0.0)],
    [Value(1.0), Value(1.0)],
]
ys = [Value(0.0), Value(1.0), Value(1.0), Value(0.0)]

def print_param_info(step):
    ps = model.parameters()
    print(f"\n--- Step {step} param info (showing first 6 params) ---")
    for i, p in enumerate(ps[:6]):
        print(f"param[{i}] data={p.data:.6f} grad={p.grad:.6f}")

# Run two iterations and inspect
for i in range(2):
    ypred = [model(x) for x in xs]
    loss = sum((yp - yt)**2 for yp, yt in zip(ypred, ys))
    print(f"iter {i}: loss (before backward) = {loss.data:.6f}")
    model.zero_grad()
    loss.backward()
    print_param_info(i)
    # update
    for p in model.parameters():
        p.data -= 0.01 * p.grad
