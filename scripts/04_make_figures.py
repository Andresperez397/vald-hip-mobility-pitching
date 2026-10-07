"""Figures from the saved result tables (no data needed).

python scripts/04_make_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
T = ROOT / "reports" / "tables"
F = ROOT / "reports" / "figures"
BLUE, ORANGE, GREY, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#9a9890", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update(
    {
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.titlelocation": "left",
    }
)  # fmt: skip


def save(fig, name):
    F.mkdir(parents=True, exist_ok=True)
    fig.savefig(F / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_audit(res, raw):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    labels = ["Release speed", "Elbow torque"]
    keys = ["ball_release_speed_mph", "max_elbow_varus_torque_nm"]
    clean = [res["release_speed"]["models"]["body_size_and_handedness"]["cv_r2"],
             res["elbow_torque"]["models"]["body_size_and_handedness"]["cv_r2"]]  # fmt: skip
    dirty = [raw[k]["body_size_and_handedness"]["cv_r2"] for k in keys]
    y = np.arange(2)
    ax.barh(y + 0.19, dirty, height=0.34, color=ORANGE, label="Table as delivered (103 rows)")
    ax.barh(y - 0.19, clean, height=0.34, color=BLUE, label="After the audit (97 / 76 rows)")
    for yy, v in zip(y + 0.19, dirty, strict=True):
        ax.text(v + 0.01, yy, f"{v:.2f}", va="center", fontsize=9, color=INK)
    for yy, v in zip(y - 0.19, clean, strict=True):
        ax.text(v + 0.01, yy, f"{v:.2f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.05)
    ax.invert_yaxis()
    ax.set_xlabel("Cross-validated R², body size + handedness + level only")
    ax.set_title("Body size appears to explain 90% of pitch speed. It doesn't.")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8.5)
    ax.grid(axis="y", visible=False)
    save(fig, "fig1_audit.png")


def fig_q1(res):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    fig.subplots_adjust(wspace=0.08)
    for ax, key, title in (
        (axes[0], "release_speed", "Release speed (n = 97)"),
        (axes[1], "elbow_torque", "Elbow varus torque (n = 76)"),
    ):
        m = res[key]["models"]
        names = [("plus_hip_tests", "+ 14 hip tests"), ("plus_hip_and_shoulder_tests", "+ hip and 6 shoulder tests")]
        for i, (k, _lab) in enumerate(names):
            lo, hi = m[k]["gain_ci_first_repeat"]
            g = m[k]["gain_over_body_size"]
            ax.plot([lo, hi], [i, i], color=BLUE, lw=2, solid_capstyle="round")
            ax.plot(g, i, "o", color=BLUE, ms=8, mec="white", mew=2)
        ax.axvline(0, color=ORANGE, lw=1.5, ls="--")
        ax.set_yticks(range(2), [n[1] for n in names])
        ax.set_ylim(-0.6, 1.6)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("R² gain over body size (95% interval)")
    fig.suptitle(
        "No detectable gain from static tests beyond body size",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="bold",
        y=1.04,
    )
    save(fig, "fig2_static_tests.png")


def fig_q2(res):
    h = res["hip_mocap"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    ax = axes[0]
    ax.hist(h["gain_all_targets"], bins=40, color=GREY)
    ax.axvline(0, color=INK, lw=1)
    ax.set_xlabel("R² gain from hip tests, per motion-capture target")
    ax.set_ylabel("Targets")
    ax.set_title(f"{h['targets']} hip and pelvis targets: centered on zero")
    ax.set_xlim(-0.3, 0.2)
    ax = axes[1]
    ax.hist(h["null_best_gains"], bins=25, color=GREY)
    ax.axvline(h["best_gain"], color=ORANGE, lw=2)
    ax.axvline(h["null_best_gain_p95"], color=INK, lw=1, ls="--")
    ax.text(
        h["best_gain"],
        ax.get_ylim()[1] * 0.92,
        f" best real target\n {h['best_gain']:.2f}",
        color=INK,
        fontsize=8.5,
        va="top",
    )
    ax.text(
        h["null_best_gain_p95"],
        ax.get_ylim()[1] * 0.55,
        "95th pct of\nshuffled \n",
        color=MUTED,
        fontsize=8,
        ha="right",
    )
    ax.set_xlabel("Best R² gain across all targets, with the tests shuffled")
    ax.set_title(f"Beating luck: p = {h['p_value_best_target']:.3f} for the best target")
    save(fig, "fig3_hip_mocap.png")


def fig_power(pw):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for key, lab, col in (
        ("release_speed", "Release speed (n = 97)", BLUE),
        ("elbow_torque", "Elbow torque (n = 76)", ORANGE),
    ):
        d = pw[key]["by_planted_gain"]
        xs = [float(k) for k in d]
        ys = [d[k]["power"] for k in d]
        ax.plot(xs, ys, marker="o", color=col, lw=2, label=lab)
    ax.axhline(0.8, color=GREY, lw=1, ls="--")
    ax.text(0.0, 0.82, "80% power", color=MUTED, fontsize=8.5)
    ax.set_xlabel("R² gain from the static tests that was planted in simulated outcomes")
    ax.set_ylabel("Chance this design detects it")
    ax.set_ylim(0, 1.02)
    ax.set_title("A real effect under ~0.2 R² would usually be missed")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    save(fig, "fig4_power.png")


def main():
    res = json.loads((T / "results.json").read_text())
    raw = json.loads((T / "uncleaned_comparison.json").read_text())
    fig_audit(res, raw)
    fig_q1(res)
    fig_q2(res)
    fig_power(json.loads((T / "power.json").read_text()))


if __name__ == "__main__":
    main()
