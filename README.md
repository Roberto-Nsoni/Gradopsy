# Gradopsy

A machine-learning engine written from scratch on top of NumPy (tensors with reverse-mode
automatic differentiation, layers, losses, first- and second-order optimizers) and a website
that runs that same engine in the browser, through Pyodide, to show it working.

**Live:** [gradopsy.rupar.workers.dev](https://gradopsy.rupar.workers.dev/)

Nothing of the mathematics is reimplemented in JavaScript: the page loads the Python package and
asks it for every number it draws.

## The engine (`src/gradopsy/`)

- **Autograd.** A `Tensor` records the graph as operations run, and `backward()` walks it in
  reverse topological order, applying each operation's own gradient rule. Broadcasting is handled
  in the backward pass.
- **Operations.** Addition, product, power, matrix product, exp, log, sum, mean, transpose,
  reshape, ReLU, sigmoid, tanh, 2D convolution and max pooling.
- **Layers and losses.** `Linear`, `Conv2D`, `MaxPool2D`, `Flatten`, `Sequential`, `MLP`; MSE,
  cross-entropy from logits (fused with the softmax), binary cross-entropy from logits.
- **Optimizers.** SGD, Momentum, RMSProp and Adam; BFGS and damped Newton with a
  finite-difference Hessian, with a strong-Wolfe line search; a linear learning-rate decay.
- **Training.** A steppable `Trainer`, early stopping, a recorder of parameter trajectories, 2D
  datasets and test surfaces for the optimizers, and an MNIST reader.

Every operation is checked against a numerical gradient before it is used: 130 tests in total.

## The website (`demos/`)

Vite + React, with the engine loaded into the page by Pyodide.

- **playground:** draw a 2D dataset, build a network layer by layer, and watch it train live.
  Every neuron is drawn as its own field over the plane, and the training history can be rewound.
- **recognize:** draw a digit and read it with two networks trained on MNIST by the engine. The
  whole forward pass is laid out stage by stage, and pointing at any value shows how it was
  computed, with the network's own numbers.

## Results

### Linear regression against the closed form

Every optimizer of the engine has to land on the weights of the normal equations,
$w^* = (X^\top X)^{-1} X^\top y$. Steps until $\lVert w - w^* \rVert < 10^{-6}$, on 200 points with
3 features, well conditioned and with the features on scales 1, 10 and 100 (condition number
about 9300):

| optimizer | well conditioned | scales 1, 10, 100 |
|---|---:|---:|
| SGD | 9 | not in 500 |
| Momentum | 270 | not in 500 |
| Adam | 286 | 379 |
| BFGS | 7 | 8 |
| Newton | 2 | 2 |

### MNIST

MNIST test accuracy, mean ± standard deviation over three seeds, measured on the float32 weights
the website loads (`demos/public/recognize/results.json`):

| model | parameters | test accuracy | moved and resized test |
|---|---:|---:|---:|
| MLP 784 → 256 → 128 → 10 | 235,146 | 98.48 % ± 0.09 | 97.38 % |
| CNN (two conv + pool blocks, linear) | 11,274 | 98.65 % ± 0.17 | 97.49 % |

Both are trained with Adam and random scaling and shifting of the training digits, so the website
can read a digit wherever and however large it is drawn.

## The paper (`paper/`)

A LaTeX write-up of the engine: the autodiff, every operation with its gradient, the losses and
optimizers, how it is tested, how it runs in the browser, and the experiments. The MNIST table
comes from `paper/make_tables.py`, and the rest of the experiments (linear regression against the
closed form, learning curves, shifted digits) from the notebook `paper/experiments.ipynb`.

## Running it

Python 3.12 or newer.

```bash
pip install -e ".[dev]"
pytest -q                                  # the 130 tests

python scripts/get_mnist.py                # MNIST into data/raw/mnist
python scripts/train_mnist.py --report     # both models x 3 seeds -> demos/public/recognize/

cd paper
python make_tables.py
jupyter nbconvert --to notebook --execute --inplace experiments.ipynb
```

The website (needs Node and [uv](https://docs.astral.sh/uv/), which builds the engine's wheel):

```bash
cd demos
npm install
npm run dev
```

## License

MIT, see [LICENSE](LICENSE).
