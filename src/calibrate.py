"""
Calibrate short-rate models to a market yield curve.
"""
import time
import numpy as np
from scipy.optimize import least_squares
from src.models import vasicek_yield


def vasicek_residuals(params, taus, market_yields):
    """Difference between model and market yields at each maturity."""
    a, b, sigma, r0 = params
    return vasicek_yield(a, b, sigma, r0, taus) - market_yields


def calibrate_vasicek(taus, market_yields, x0=(0.1, 0.06, 0.01, 0.04)):
    """
    Fit Vasicek parameters by Levenberg-Marquardt least squares.

    Returns a dict with the fitted parameters, RMSE in basis points,
    and wall-clock calibration time in seconds.
    """
    t0 = time.perf_counter()
    result = least_squares(
        vasicek_residuals, x0, args=(taus, market_yields), method="lm"
    )
    elapsed = time.perf_counter() - t0

    a, b, sigma, r0 = result.x
    rmse_bps = np.sqrt(np.mean(result.fun**2)) * 1e4

    return {
        "a": a, "b": b, "sigma": sigma, "r0": r0,
        "rmse_bps": rmse_bps,
        "time_s": elapsed,
        "success": result.success,
    }