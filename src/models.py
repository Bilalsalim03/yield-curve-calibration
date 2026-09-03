"""
Short-rate models with closed-form zero-coupon bond prices.
"""
import numpy as np


def vasicek_bond_price(a, b, sigma, r0, tau):
    """
    Zero-coupon bond price under the Vasicek model.

    Parameters
    ----------
    a     : mean reversion speed
    b     : long-run mean of the short rate
    sigma : volatility of the short rate
    r0    : current short rate
    tau   : time to maturity in years (scalar or numpy array)
    """
    tau = np.asarray(tau, dtype=float)
    B = (1 - np.exp(-a * tau)) / a
    A = np.exp((B - tau) * (a**2 * b - 0.5 * sigma**2) / a**2
               - (sigma**2 * B**2) / (4 * a))
    return A * np.exp(-B * r0)


def vasicek_yield(a, b, sigma, r0, tau):
    """Continuously compounded zero yield implied by the Vasicek bond price."""
    tau = np.asarray(tau, dtype=float)
    return -np.log(vasicek_bond_price(a, b, sigma, r0, tau)) / tau