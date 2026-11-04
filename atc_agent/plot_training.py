"""
Learning curves of the submitted lineage (main run -> time-aware -> geodesic)
and of the time-aware control, from the training logs (progress.jsonl).

Metrics are those of the training rollouts' competition-configuration episodes
(SA with 10 intruders, MA original cohort), i.e. of the *stochastic* policy, as
episode-weighted means over a sliding window of training steps. The x axis is
broken at 1.0B steps: the right panels show the fine-tunes at a finer scale.

    python -m atc_agent.plot_training --out docs/solution/report/figures/training.pdf
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MAIN_RUN = r"C:\Users\localadmin\Desktop\ai4realnet-atc-competition\runs\main"
# (run dir, first step kept, last step kept); steps are global across resumes
LINEAGE = [(MAIN_RUN, 0, 1.0005e9), ("runs/ft_ta", 1.0e9, 1.0705e9), ("runs/ft_geo", 1.07e9, 1.135e9),
           ("runs/ft_geo_r", 1.135e9, 1.1675e9), ("runs/ft_geo_r2", 1.1673e9, 1.19e9)]
CONTROL = [("runs/ft_ta", 1.0703e9, 1.31e9)]
SPLIT, GEO_START = 1.0e9, 1.0704e9
SERIES = {"sa10": ("single-agent", "#2a78d6"), "ma_orig": ("multi-agent", "#eb6834")}
LAMBDAS = [("lam_intr", "intrusion", "-"), ("lam_ra", "restricted area", "--"), ("lam_sec", "sector exit", ":")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(segments):
    rows = []
    for run, lo, hi in segments:
        with open(f"{run}/progress.jsonl") as f:
            rows += [r for r in map(json.loads, f) if lo <= r["t"] <= hi]
    return sorted(rows, key=lambda r: r["t"])


def windowed(rows, key, window):
    """Episode-weighted sliding means of timeouts, violations and flight time."""
    t = np.array([r["t"] for r in rows], float)
    n = np.array([r[key]["n"] for r in rows], float)
    m = {
        "timeout": 1.0 - np.array([r[key]["waypoint_reached"] for r in rows]),
        "violations": np.array([r[key]["intrusion_events"] + r[key]["restricted_area_events"]
                                + r[key]["sector_exit_events"] for r in rows]),
        "flight_time": np.array([r[key]["flight_time"] for r in rows]),
    }
    c_n = np.concatenate([[0], np.cumsum(n)])
    lo = np.searchsorted(t, t - window, side="left")
    hi = np.arange(1, len(t) + 1)
    out = {k: (np.concatenate([[0], np.cumsum(n * v)])[hi] - np.concatenate([[0], np.cumsum(n * v)])[lo])
           / (c_n[hi] - c_n[lo]) for k, v in m.items()}
    keep = t - window >= t[0]          # full windows only
    return t[keep], {k: v[keep] for k, v in out.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"],
                         "mathtext.fontset": "stix", "font.size": 7, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5})
    lin, ctl = load(LINEAGE), load(CONTROL)
    fig, axes = plt.subplots(4, 2, figsize=(3.4, 3.9), sharey="row",
                             gridspec_kw=dict(width_ratios=[1.0, 1.0], wspace=0.06, hspace=0.18))
    rows = [("timeout", "timeouts [%]", 100.0, False), ("violations", "violations / flight", 1.0, True),
            ("flight_time", "flight time [s]", 1.0, False)]
    for col, (window, xlim) in enumerate([(10e6, (0, SPLIT)), (5e6, (SPLIT, 1.30e9))]):
        for key, (label, color) in SERIES.items():
            for data, ls, lw in ((lin, "-", 0.9), (ctl, (0, (3, 1.5)), 0.7)):
                if col == 0 and data is ctl:
                    continue
                t, m = windowed(data, key, window)
                for r, (mk, _, scale, _) in enumerate(rows):
                    axes[r, col].plot(t / 1e9, m[mk] * scale, color=color, ls=ls, lw=lw,
                                      label=label if (r == 0 and col == 1 and data is lin) else None)
        for lk, label, ls in LAMBDAS:
            for data, lw in ((lin, 0.9),) + (((ctl, 0.5),) if col == 1 else ()):
                t = np.array([r["t"] for r in data]) / 1e9
                axes[3, col].plot(t, [r["lam"][lk] for r in data], color=INK if data is lin else MUTED,
                                  ls=ls, lw=lw, label=label if (col == 0 and data is lin) else None)
        for r in range(4):
            ax = axes[r, col]
            ax.set_xlim(xlim[0] / 1e9, xlim[1] / 1e9)
            ax.grid(True, color=GRID, lw=0.4)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            if r < 3:
                ax.tick_params(labelbottom=False)
            if col == 1:
                ax.axvline(GEO_START / 1e9, color=MUTED, lw=0.5, ls=":")
                ax.spines["left"].set_visible(False)
                ax.tick_params(left=False)
    for r, (_, ylabel, _, logy) in enumerate(rows):
        axes[r, 0].set_ylabel(ylabel)
        if logy:
            axes[r, 0].set_yscale("log")
    axes[3, 0].set_ylabel(r"multiplier $\lambda$")
    axes[3, 0].set_xlabel("training steps [$10^9$]")
    axes[3, 1].set_xlabel("fine-tunes [$10^9$]")
    axes[3, 0].set_xticks([0, 0.25, 0.5, 0.75])
    axes[3, 1].set_xticks([1.0, 1.1, 1.2, 1.3])
    axes[0, 0].set_ylim(bottom=0)
    axes[0, 0].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    axes[0, 1].text(1.035, 1.02, "time-\naware", transform=axes[0, 1].get_xaxis_transform(), ha="center",
                    va="bottom", color=MUTED, fontsize=6)
    axes[0, 1].text(1.13, 1.02, "geodesic", transform=axes[0, 1].get_xaxis_transform(), ha="center",
                    va="bottom", color=MUTED, fontsize=6)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color=MUTED, ls=(0, (3, 1.5)), lw=0.7))
    labels.append("control: time-aware continued")
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=False,
               handlelength=1.8, columnspacing=1.0, fontsize=6.5)
    axes[3, 0].legend(frameon=False, loc="lower right", handlelength=2.2, fontsize=6)
    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(a.out.rsplit(".", 1)[0] + ".png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
