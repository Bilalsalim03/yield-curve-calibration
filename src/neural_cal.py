"""
Neural network calibration of the Vasicek model (after Hernandez, 2016).

Train an MLP to map a yield curve to model parameters, so that calibration
becomes a single forward pass instead of an iterative optimisation.
"""
import time
import numpy as np
import torch
import torch.nn as nn
from src.models import vasicek_yield

SIGMA_FIXED = 0.01
PARAM_RANGES = {"a": (0.02, 0.5), "b": (0.0, 0.12), "r0": (0.0, 0.10)}


def make_dataset(taus, n=100_000, seed=0, noise_bps=5.0):
    """
    Sample random parameters and compute the yield curve each implies,
    with Gaussian noise added so the network is robust to curves that
    are not exactly Vasicek-shaped.
    """
    rng = np.random.default_rng(seed)
    a = rng.uniform(*PARAM_RANGES["a"], n)
    b = rng.uniform(*PARAM_RANGES["b"], n)
    r0 = rng.uniform(*PARAM_RANGES["r0"], n)
    curves = vasicek_yield(a[:, None], b[:, None], SIGMA_FIXED, r0[:, None], taus)
    curves = curves + rng.normal(0.0, noise_bps * 1e-4, curves.shape)
    params = np.column_stack([a, b, r0])
    return curves.astype(np.float32), params.astype(np.float32)


class Calibrator(nn.Module):
    """Small MLP: yield curve in, (a, b, r0) out."""

    def __init__(self, n_in, n_out=3, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):
        return self.net(x)


def train_calibrator(taus, n=200_000, epochs=80, batch=512, lr=1e-3, hidden=128, seed=0):
    """Train the calibrator on simulated curves. Returns (model, scalers)."""
    X, Y = make_dataset(taus, n, seed)
    x_mean, x_std = X.mean(0), X.std(0)
    y_mean, y_std = Y.mean(0), Y.std(0)
    Xt = torch.tensor((X - x_mean) / x_std)
    Yt = torch.tensor((Y - y_mean) / y_std)

    torch.manual_seed(seed)
    model = Calibrator(len(taus), hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        perm = torch.randperm(len(Xt))
        total = 0.0
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), Yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1:3d}  loss {total / len(Xt):.6f}")

    model.eval()
    scalers = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    return model, scalers


def neural_calibrate(model, scalers, market_yields, n_repeat=1000):
    """Calibrate to a market curve in one forward pass. Time is averaged over n_repeat calls."""
    x = torch.tensor(((market_yields - scalers["x_mean"]) / scalers["x_std"]).astype(np.float32))[None, :]
    with torch.no_grad():
        model(x)                                   # warm-up call, not timed
        t0 = time.perf_counter()
        for _ in range(n_repeat):
            y = model(x)
        elapsed = (time.perf_counter() - t0) / n_repeat
    params = y.numpy()[0] * scalers["y_std"] + scalers["y_mean"]

    a = float(np.clip(params[0], *PARAM_RANGES["a"]))
    b = float(np.clip(params[1], *PARAM_RANGES["b"]))
    r0 = float(np.clip(params[2], *PARAM_RANGES["r0"]))
    return {"a": float(a), "b": float(b), "sigma": SIGMA_FIXED, "r0": float(r0), "time_s": elapsed}