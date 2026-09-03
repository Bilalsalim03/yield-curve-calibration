"""
Load a market yield curve and convert it to zero-coupon bond prices.
"""
import numpy as np
import pandas as pd


def load_yield_curve(path="data/yield_curve.csv"):
    """
    Returns (taus, yields, prices) as numpy arrays.

    The CSV holds spot rates in percent. Yields are returned as decimals
    (continuously compounded), and prices as P(tau) = exp(-y * tau).
    """
    df = pd.read_csv(path).dropna()
    taus = df["maturity"].to_numpy(dtype=float)
    yields = df["rate"].to_numpy(dtype=float) / 100.0
    prices = np.exp(-yields * taus)
    return taus, yields, prices