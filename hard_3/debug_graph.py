# debug_graph.py
import numpy as np
from nanograd import MLP, Value, topological_sort

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

# Build prediction & loss exactly like validator
ypred = [model(x) for x in xs]
loss = sum((yp - yt)**2 for yp, yt in zip(ypred, ys))

print("Forward loss.data =", loss.data)

# Build topo (use the helper if available, otherwise use Value.backward's builder)
try:
    topo = topological_sort(loss)
    print("Topological sort length:", len(topo))
except Exception as e:
    # fallback to building using loss.backward's internal builder (but without running backprop)
    print("topological_sort unavailable / failed:", e)
    # build using loss.backward's builder (replicate it)
    def build(v, visited, out):
        if v not in visited:
            visited.add(v)
            for c in v._prev:
                build(c, visited, out)
            out.append(v)
    topo = []
    build(loss, set(), topo)
    print("Built topo length (fallback):", len(topo))

# Gather param list
params = model.parameters()
print("Num params:", len(params))

# Check which params are present in topo
topo_set = set(topo)
for i, p in enumerate(params[:12]):  # show first 12
    print(f"param[{i}] present_in_graph={p in topo_set} data={p.data:.6f} grad(before)={p.grad:.6f}")

# Zero grads then backward (like validator)
model.zero_grad()
loss.backward()

print("\nAfter backward (showing first 12 params):")
for i, p in enumerate(params[:12]):
    print(f"param[{i}] grad(after)={p.grad:.6f} present_in_graph={p in topo_set}")
