import numpy as np
import matplotlib.pyplot as plt
from NNw2 import X, Y, params, loss, forward_prop   # importing NNw2 runs your training first

# ---- 1. What X and Y actually are: look at 3 points outside and 3 inside ----
print("\n   x1      x2    distance from centre   Y (inside circle?)")
some = list(np.where(Y[0] == 0)[0][:3]) + list(np.where(Y[0] == 1)[0][:3])
for i in some:
    x1, x2 = X[0, i], X[1, i]
    dist = np.sqrt(x1**2 + x2**2)
    print(f"{x1:6.2f}  {x2:6.2f}        {dist:5.2f}              {Y[0, i]}")

# ---- 2. Ask the trained network about every point on a grid ----
# make a 200x200 grid of (x1, x2) points covering the plot
g1, g2 = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))
grid = np.vstack([g1.ravel(), g2.ravel()])          # shape (2, 40000): same layout as X
probs, _ = forward_prop(grid, params)               # (1, 40000) probabilities
region = probs.reshape(g1.shape)                    # back to 200x200 for drawing

# ---- 3. Draw it ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# left: the network's opinion (background) + the real data (dots)
ax1.contourf(g1, g2, region, levels=[0, 0.5, 1], colors=["#f4c7c3", "#c6dbef"], alpha=0.8)
ax1.scatter(X[0], X[1], c=Y[0], cmap="bwr_r", edgecolors="k", s=20)
ax1.add_patch(plt.Circle((0, 0), 1, fill=False, ls="--", color="k"))   # the true circle
ax1.set_title("Blue area = network says 1 (inside)\nDashed = the real circle")
ax1.set_xlabel("x1"); ax1.set_ylabel("x2"); ax1.set_aspect("equal")

# right: the cost going down
ax2.plot(np.arange(len(loss)) * 100, loss)
ax2.axhline(0.693, ls="--", color="gray", label="0.693 = coin flip")
ax2.set_title("Cost during training"); ax2.set_xlabel("iteration"); ax2.set_ylabel("cost J")
ax2.legend()

plt.tight_layout()
plt.show()
