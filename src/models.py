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

def hull_white_bond_price(theta_values, theta_knots, a, sigma, r0, tau, ds=0.01):
    """
    Zero-coupon bond price under Hull-White with piecewise-constant theta(t).

    theta_values : theta on each interval
    theta_knots  : right-hand end of each interval, increasing, last >= max(tau)
    ds           : step for numerical integration of the theta term
    """
    theta_values = np.asarray(theta_values, dtype=float)
    theta_knots = np.asarray(theta_knots, dtype=float)
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    prices = np.empty_like(tau)

    for i, T in enumerate(tau):
        s = np.arange(0.0, T, ds) + ds / 2            # midpoints of integration cells
        idx = np.minimum(np.searchsorted(theta_knots, s), len(theta_values) - 1)
        theta_s = theta_values[idx]                    # theta at each midpoint
        B_Ts = (1 - np.exp(-a * (T - s))) / a
        integral = np.sum(theta_s * B_Ts) * ds

        B_T = (1 - np.exp(-a * T)) / a
        V_T = sigma**2 / (2 * a**2) * (T - 2 * B_T + (1 - np.exp(-2 * a * T)) / (2 * a))
        prices[i] = np.exp(-B_T * r0 - integral + V_T)

    return prices


def hull_white_yield(theta_values, theta_knots, a, sigma, r0, tau, ds=0.01):
    """Continuously compounded zero yield under Hull-White."""
    tau = np.atleast_1d(np.asarray(tau, dtype=float))
    return -np.log(hull_white_bond_price(theta_values, theta_knots, a, sigma, r0, tau, ds)) / tau