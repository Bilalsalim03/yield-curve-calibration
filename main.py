"""
Yield curve calibration: classical vs neural.
Run:  python main.py
"""
import numpy as np
import pandas as pd

from src.data import load_yield_curve
from src.calibrate import calibrate_vasicek, calibrate_hull_white
from src.neural_cal import train_calibrator, neural_calibrate
from src.models import vasicek_yield, hull_white_yield
from src.plots import plot_yield_fit


def rmse_bps(model_yields, market_yields):
    return np.sqrt(np.mean((model_yields - market_yields) ** 2)) * 1e4


def main():
    taus, yields, prices = load_yield_curve()
    print(f"Loaded {len(taus)} maturities from BoE spot curve.\n")

    # 1. Vasicek by least squares
    vas = calibrate_vasicek(taus, yields)
    vas_y = vasicek_yield(vas["a"], vas["b"], vas["sigma"], vas["r0"], taus)

    # 2. Hull-White with piecewise-constant theta
    knots = [2, 5, 10, 15, 20, 25]
    hw = calibrate_hull_white(taus, yields, theta_knots=knots)
    hw_y = hull_white_yield(hw["theta_values"], hw["theta_knots"], hw["a"], hw["sigma"], hw["r0"], taus)

    # 3. Neural calibrator
    print("Training neural calibrator...")
    model, scalers = train_calibrator(taus)
    nn_fit = neural_calibrate(model, scalers, yields)
    nn_y = vasicek_yield(nn_fit["a"], nn_fit["b"], nn_fit["sigma"], nn_fit["r0"], taus)

    # 4. Hybrid: neural warm start, then least squares
    hyb = calibrate_vasicek(taus, yields, x0=(nn_fit["a"], nn_fit["b"], nn_fit["sigma"], nn_fit["r0"]))
    hyb_y = vasicek_yield(hyb["a"], hyb["b"], hyb["sigma"], hyb["r0"], taus)

    # Results table
    rows = [
        ("Vasicek (LS)",       vas["a"],    vas["b"],    vas["r0"],    rmse_bps(vas_y, yields), vas["time_s"]),
        ("Hull-White (LM)",    hw["a"],     np.nan,      hw["r0"],     rmse_bps(hw_y, yields),  hw["time_s"]),
        ("Neural (Vasicek)",   nn_fit["a"], nn_fit["b"], nn_fit["r0"], rmse_bps(nn_y, yields),  nn_fit["time_s"]),
        ("Hybrid (NN -> LM)",  hyb["a"],    hyb["b"],    hyb["r0"],    rmse_bps(hyb_y, yields), hyb["time_s"] + nn_fit["time_s"]),
    ]
    table = pd.DataFrame(rows, columns=["method", "a", "b", "r0", "rmse_bps", "time_s"])
    print("\n" + table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    table.to_csv("results/results.csv", index=False)

    # Chart
    plot_yield_fit(taus, yields, {
        "Vasicek (LS)": vas_y,
        "Hull-White (LM)": hw_y,
        "Neural (Vasicek)": nn_y,
    })
    print("\nSaved results/results.csv and results/fit.png")


if __name__ == "__main__":
    main()