"""Matplotlib style shared by the report figures: Times-like serif as in the LaTeX source."""

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
REPORT_RC = {"font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"], "mathtext.fontset": "stix",
             "font.size": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
             "ytick.color": MUTED, "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5}
