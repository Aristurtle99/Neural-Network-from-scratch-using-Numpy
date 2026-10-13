# Neural network from scratch (NumPy)

A 2-layer neural network (one hidden layer) written with nothing but NumPy: forward pass, cross-entropy cost, hand-derived backpropagation, and gradient descent. No autograd and no ML frameworks.

It learns to classify whether a 2D point lies inside the unit circle. A single straight line can't separate those two classes, so logistic regression fails on this task. A hidden layer of 4 tanh units is enough.

**Result:** 98.25% training accuracy after 10,000 iterations (hidden size 4, learning rate 0.5). The cost goes from 0.693 to about 0.056.

## Run it

```bash
pip install -r requirements.txt
python NNw2.py        # trains, prints the cost every 1000 iterations and the accuracy, plots the cost
python plot_circle.py # shows sample points, the learned decision boundary, and the cost curve
```

`plot_circle.py` imports from `NNw2.py`, so it runs the training first. Close the first cost plot window and the decision-boundary figure will appear.

## Files

| File | What it does |
|---|---|
| `NNw2.py` | dataset, `initialize_parameters`, `forward_prop`, `compute_cost`, `backward_prop`, `update_params`, `train`, `predict` |
| `plot_circle.py` | shows the network's prediction over a grid next to the true circle, plus the cost curve |

## The data

400 points are drawn from a standard 2D normal distribution. Each label is

```math
y = \begin{cases} 1 & \text{if } x_1^2 + x_2^2 < 1 \\ 0 & \text{otherwise} \end{cases}
```

About 40% of the points are inside the circle.

## The model

Notation: $m = 400$ examples, each with $n_x = 2$ features, stacked as columns. The hidden layer has $n_h = 4$ units and there is one output unit ($n_y = 1$).

| Quantity | Shape |
|---|---|
| $X$ | $(n_x, m)$ |
| $W^{[1]}, b^{[1]}$ | $(n_h, n_x)$, $(n_h, 1)$ |
| $Z^{[1]}, A^{[1]}$ | $(n_h, m)$ |
| $W^{[2]}, b^{[2]}$ | $(n_y, n_h)$, $(n_y, 1)$ |
| $Z^{[2]}, A^{[2]}, Y$ | $(n_y, m)$ |

### Initialisation

The weights start as small random numbers (`np.random.randn(...) * 0.01`) and the biases start at zero.

- **Why random:** if every hidden unit started with identical weights, they would all receive the same gradient and stay identical forever. Random values break that symmetry.
- **Why small:** it keeps tanh in its steep middle region, where gradients are large.

### Forward pass

```math
Z^{[1]} = W^{[1]} X + b^{[1]}, \qquad A^{[1]} = \tanh(Z^{[1]})
```
```math
Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}, \qquad A^{[2]} = \sigma(Z^{[2]}) = \frac{1}{1 + e^{-Z^{[2]}}}
```

$A^{[2]}$ is the predicted probability that a point is inside the circle. Each bias $b$ is a column vector that NumPy broadcasts across all $m$ columns.

### Cost

Binary cross-entropy, averaged over the examples:

```math
J = -\frac{1}{m} \sum_{i=1}^{m} \Big[ y^{(i)} \log a^{(i)} + (1 - y^{(i)}) \log (1 - a^{(i)}) \Big]
```

where $a^{(i)}$ is the output for example $i$.

### Backpropagation, derived

Backprop is the chain rule applied layer by layer, from the output back toward the input. Below I write $dV$ for $\partial \mathcal{L} / \partial V$, where $\mathcal{L}$ is the loss for a single example. The $1/m$ factor appears when averaging over all examples.

**Output layer.** Differentiating the loss with respect to the output $a$:

```math
\frac{\partial \mathcal{L}}{\partial a} = -\frac{y}{a} + \frac{1 - y}{1 - a}
```

The sigmoid has the derivative $\sigma'(z) = \sigma(z)(1 - \sigma(z)) = a(1 - a)$. By the chain rule:

```math
dz^{[2]} = \frac{\partial \mathcal{L}}{\partial a} \cdot a(1 - a) = -y(1 - a) + (1 - y)a = a - y
```

Everything else cancels and only $a - y$ is left, which is why sigmoid and cross-entropy are usually paired. Over all examples, `dZ2 = A2 - Y`.

Since $Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$, the gradients for the second layer's parameters are:

```math
dW^{[2]} = \frac{1}{m} dZ^{[2]} A^{[1]T}, \qquad db^{[2]} = \frac{1}{m} \sum_{i} dZ^{[2](i)}
```

The bias gradient sums over examples (`axis=1`). `keepdims=True` keeps it at shape $(n_y, 1)$ so it lines up with $b^{[2]}$.

**Hidden layer.** The error flows back through $W^{[2]}$:

```math
dA^{[1]} = W^{[2]T} dZ^{[2]}
```

Then through tanh, using $\tanh'(z) = 1 - \tanh^2(z)$. We already have $A^{[1]} = \tanh(Z^{[1]})$ from the forward pass, so we can reuse it:

```math
dZ^{[1]} = dA^{[1]} \odot (1 - A^{[1]2})
```

Here $\odot$ is element-wise multiplication (`*` in NumPy, not `np.dot`). The first layer's parameter gradients follow the same pattern as the second layer's:

```math
dW^{[1]} = \frac{1}{m} dZ^{[1]} X^T, \qquad db^{[1]} = \frac{1}{m} \sum_{i} dZ^{[1](i)}
```

A quick shape check: $dZ^{[1]}$ is $(n_h, m)$ and $X^T$ is $(m, n_x)$, so $dW^{[1]}$ is $(n_h, n_x)$, the same shape as $W^{[1]}$. Every gradient has the same shape as the parameter it updates.

### Gradient descent

Each iteration runs forward, cost, backward, update, with learning rate $\alpha = 0.5$:

```math
W^{[l]} \leftarrow W^{[l]} - \alpha \, dW^{[l]}, \qquad b^{[l]} \leftarrow b^{[l]} - \alpha \, db^{[l]}
```

### Prediction

A point is classified as inside the circle when $A^{[2]} > 0.5$.

## Notes from training

- **The starting cost is $\ln 2 \approx 0.693$.** With weights near zero, $Z^{[2]} \approx 0$, so the network outputs 0.5 for every point. The cost of that guess is $-\log 0.5 = \ln 2$.
- **There's a short plateau at about 0.673 before the cost drops.** That's exactly the cost of predicting 0.4 (the fraction of points inside the circle) for every point. So first the network learns only the class balance. Then the hidden units separate and start bending the decision boundary into a closed shape around the circle.

## Background

Built while working through Andrew Ng's [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning), Course 1 (shallow neural networks).
