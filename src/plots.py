"""
Plotting helpers for calibration results.
"""
import matplotlib.pyplot as plt


def plot_yield_fit(taus, market_yields, fitted_curves, path="results/fit.png"):
    """
    Plot market yields against one or more fitted model curves.

    fitted_curves : dict of {label: yields_array}
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(taus, market_yields * 100, "ko", ms=4, label="BoE spot curve")
    for label, y in fitted_curves.items():
        ax.plot(taus, y * 100, lw=2, label=label)
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Zero yield (%)")
    ax.set_title("Yield curve calibration")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)