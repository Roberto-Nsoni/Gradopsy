
import numpy as np
from src.ml.tensor import Tensor
from src.ml.nn.module import Linear

def calc_analytical_solution(X_np, y_np):
    W = np.linalg.inv(X_np.T @ X_np) @ X_np.T @ y_np
    return W.reshape(-1, 1)

def train_model(X_np, y_np, P, lr=0.01, steps=10000, bias=False):
    X_t = Tensor(X_np, requires_grad=False)
    y_t = Tensor(y_np, requires_grad=False)

    model = Linear(P, 1, bias=bias)
    print("Training...")
    for i in range(steps):
        pred = model(X_t)
        loss = ((pred - y_t) ** 2).mean()

        model.zero_grad()
        loss.backward()

        for _, p in model.parameters():
            p.data -= lr * p.grad

        if i % (steps//10) == 0:
            print(f"[step {i}] loss = {loss.data}")

    learned_W = None
    for _, p in model.parameters():
        learned_W = p.data

    return learned_W

def run_experiment(X_np, true_W, y_np):
    learned_W = train_model(X_np, y_np, P=X_np.shape[1])
    analytical_W = calc_analytical_solution(X_np, y_np)

    print("True W:\n", true_W)
    print("Learned W:\n", learned_W)
    print("Analytical W:\n", analytical_W)
    print("Difference (analytical - learned):\n", analytical_W - learned_W)

def main():
    N = 200   # number of samples
    P = 3     # number of features

    # ---- Gaussian noise -----
    print("\n" + "-" * 20 + " Gaussian noise " + "-" * 20)
    X = np.random.randn(N, P)
    W = np.random.randn(P, 1)
    y = X @ W + 0.1 * np.random.randn(N, 1)
    run_experiment(X, W, y)

    # ---- Scaled -----
    print("\n" + "-" * 20 + " Scaled " + "-" * 20)
    X = np.random.randn(N, P)
    X = X * np.array([1, 10, 100])
    W = np.random.randn(P, 1)
    y = X @ W + 0.2 * np.random.randn(N, 1)
    run_experiment(X, W, y)

    # ---- Collinear -----
    print("\n" + "-" * 20 + " Collinear " + "-" * 20)
    X_base = np.random.randn(N, 1)
    X = np.hstack([
        X_base,
        X_base + 0.01 * np.random.randn(N, 1),
        np.random.randn(N, 1)
    ])
    W = np.random.randn(P, 1)
    y = X @ W + 0.2 * np.random.randn(N, 1)
    run_experiment(X, W, y)

    # ---- Outliers -----
    print("\n" + "-" * 20 + " Outliers " + "-" * 20)
    X = np.random.randn(N, P)
    W = np.random.randn(P, 1)
    y = X @ W
    outliers = np.random.choice(N, 5, replace=False)
    y[outliers] += 20 * np.random.randn(5, 1)
    run_experiment(X, W, y)

    # ---- Non-linear -----
    print("\n" + "-" * 20 + " Non-linear " + "-" * 20)
    X = np.random.randn(N, P)
    W = np.random.randn(P, 1)
    y = (
        X @ W
        + 0.5 * np.sin(X[:, 0:1])
        + 0.2 * (X[:, 1:2] ** 2)
        + 0.3 * np.random.randn(N, 1)
    )
    run_experiment(X, W, y)


if __name__ == "__main__":
    main()