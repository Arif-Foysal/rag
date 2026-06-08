#!/usr/bin/env python3
"""Conceptual figure: *what* we conformalize.

Two-lane contrast schematic for the Introduction of the conformal-gating paper.
Top lane  = prior conformal LLM/RAG (guarantee on the generated *output*).
Bottom lane = this work (guarantee on the pre-generation *gate decision*).

No data dependency; pure matplotlib patches (boxes, arrows, and small vector
icons). Set ICONS=False for the plain version. Saves to OUT.
"""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon, Circle, Arc,
)

# ----------------------------------------------------------------------
# Toggle + output name. While previewing the icon version we write to a
# separate file so the live manuscript figure (fig_concept.pdf) is untouched.
# ----------------------------------------------------------------------
ICONS = True
OUT = "fig_concept.pdf"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "text.usetex": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

BLUE = "#2b6ca3"     # ours / retrieve path
ORANGE = "#d2691e"   # prior work / output guarantee
RED = "#c0392b"      # harmful skip
GRAY_FILL = "#f2f2f2"
GRAY_EDGE = "#444444"
INK = "#222222"

FS_TITLE = 11.0
FS_BOX = 10.0
FS_NOTE = 8.5


# ======================================================================
# Small vector icons. Each draws inside a box centred at (x, y) with
# nominal half-size s, stroked in colour c.
# ======================================================================
def ic_doc(ax, x, y, s, c):
    w, h = 1.0 * s, 1.4 * s
    pts = [(x - w / 2, y - h / 2), (x - w / 2, y + h / 2),
           (x + w / 2 - 0.4 * s, y + h / 2), (x + w / 2, y + h / 2 - 0.4 * s),
           (x + w / 2, y - h / 2)]
    ax.add_patch(Polygon(pts, closed=True, fill=False, edgecolor=c, lw=1.0, zorder=6))
    ax.plot([x + w / 2 - 0.4 * s, x + w / 2 - 0.4 * s, x + w / 2],
            [y + h / 2, y + h / 2 - 0.4 * s, y + h / 2 - 0.4 * s],
            color=c, lw=0.9, zorder=6)
    for dy in (0.22, 0.0, -0.22):
        ax.plot([x - w / 2 + 0.2 * s, x + w / 2 - 0.2 * s], [y + dy * s, y + dy * s],
                color=c, lw=0.8, zorder=6)


def ic_chip(ax, x, y, s, c):
    w = h = 1.0 * s
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fill=False,
                           edgecolor=c, lw=1.1, zorder=6))
    ax.add_patch(Rectangle((x - w / 4, y - h / 4), w / 2, h / 2, fill=False,
                           edgecolor=c, lw=0.8, zorder=6))
    for f in (-0.55, 0.0, 0.55):
        ax.plot([x + f * w / 2, x + f * w / 2], [y + h / 2, y + h / 2 + 0.22 * s],
                color=c, lw=0.9, zorder=6)
        ax.plot([x + f * w / 2, x + f * w / 2], [y - h / 2, y - h / 2 - 0.22 * s],
                color=c, lw=0.9, zorder=6)
        ax.plot([x - w / 2, x - w / 2 - 0.22 * s], [y + f * h / 2, y + f * h / 2],
                color=c, lw=0.9, zorder=6)
        ax.plot([x + w / 2, x + w / 2 + 0.22 * s], [y + f * h / 2, y + f * h / 2],
                color=c, lw=0.9, zorder=6)


def ic_list(ax, x, y, s, c):
    for dy in (0.3, 0.0, -0.3):
        ax.add_patch(Circle((x - 0.45 * s, y + dy * s), 0.07 * s, color=c, zorder=6))
        ax.plot([x - 0.25 * s, x + 0.5 * s], [y + dy * s, y + dy * s],
                color=c, lw=0.9, zorder=6)


def ic_gauge(ax, x, y, s, c):
    cy = y - 0.25 * s
    ax.add_patch(Arc((x, cy), 1.5 * s, 1.5 * s, theta1=15, theta2=165,
                     edgecolor=c, lw=1.2, zorder=6))
    ax.plot([x, x + 0.45 * s], [cy, cy + 0.5 * s], color=c, lw=1.1, zorder=6)
    ax.add_patch(Circle((x, cy), 0.07 * s, color=c, zorder=6))


def ic_threshold(ax, x, y, s, c):
    ax.plot([x - 0.6 * s, x + 0.6 * s], [y - 0.4 * s, y - 0.4 * s],
            color=c, lw=1.0, zorder=6)
    ax.plot([x - 0.05 * s, x - 0.05 * s], [y - 0.55 * s, y + 0.5 * s],
            color=c, lw=1.1, ls=(0, (2, 1.5)), zorder=6)
    ax.add_patch(Circle((x - 0.05 * s, y + 0.22 * s), 0.11 * s, color=c, zorder=6))


def ic_book(ax, x, y, s, c):
    w, h = 1.0 * s, 1.35 * s
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fill=False,
                           edgecolor=c, lw=1.1, zorder=6))
    ax.plot([x - w / 2 + 0.18 * s, x - w / 2 + 0.18 * s], [y - h / 2, y + h / 2],
            color=c, lw=0.9, zorder=6)
    ax.plot([x - w / 2 + 0.34 * s, x + w / 2 - 0.12 * s], [y + 0.25 * s, y + 0.25 * s],
            color=c, lw=0.8, zorder=6)


def ic_magnifier(ax, x, y, s, c):
    ax.add_patch(Circle((x - 0.12 * s, y + 0.15 * s), 0.5 * s, fill=False,
                        edgecolor=c, lw=1.2, zorder=6))
    ax.plot([x + 0.24 * s, x + 0.62 * s], [y - 0.22 * s, y - 0.6 * s],
            color=c, lw=1.6, zorder=6)


def ic_warning(ax, x, y, s, c):
    pts = [(x, y + 0.6 * s), (x - 0.62 * s, y - 0.48 * s), (x + 0.62 * s, y - 0.48 * s)]
    ax.add_patch(Polygon(pts, closed=True, fill=False, edgecolor=c, lw=1.2, zorder=6))
    ax.plot([x, x], [y + 0.28 * s, y - 0.12 * s], color=c, lw=1.3, zorder=6)
    ax.add_patch(Circle((x, y - 0.3 * s), 0.06 * s, color=c, zorder=6))


# ======================================================================
def box(ax, x, y, w, h, text, *, fc=GRAY_FILL, ec=GRAY_EDGE, tc=INK, lw=1.1,
        fs=FS_BOX, bold=False, rounded=0.10, icon=None, ic_color=None):
    """Rounded box centred at (x, y); optional left-inset vector icon.

    With an icon, the label is left-aligned in the space to its right so the
    glyph and text never overlap.
    """
    left, right = x - w / 2, x + w / 2
    patch = FancyBboxPatch(
        (left, y - h / 2), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={rounded}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3,
    )
    ax.add_patch(patch)
    if icon is not None and ICONS:
        icon(ax, left + 0.46, y, 0.32, ic_color or ec)
        ax.text(left + 0.92, y, text, ha="left", va="center", fontsize=fs,
                color=tc, zorder=4, fontweight="bold" if bold else "normal")
    else:
        ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc,
                zorder=4, fontweight="bold" if bold else "normal")
    return left, right


def arrow(ax, x0, y0, x1, y1, *, color=GRAY_EDGE, lw=1.3, ls="-", style="-|>"):
    ax.add_patch(
        FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=12,
                        linewidth=lw, color=color, linestyle=ls, zorder=2,
                        shrinkA=0, shrinkB=0)
    )


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, OUT)

    fig, ax = plt.subplots(figsize=(9.9, 3.05))
    ax.set_xlim(0, 23.4)
    ax.set_ylim(-0.55, 6.6)
    ax.axis("off")

    # Lane background bands -------------------------------------------------
    ax.add_patch(Rectangle((0.1, 3.66), 23.1, 2.8, facecolor="#fbf3ec",
                           edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((0.1, 0.34), 23.1, 3.02, facecolor="#eef4f9",
                           edgecolor="none", zorder=0))

    # Lane labels -----------------------------------------------------------
    ax.text(0.32, 6.22, "Prior conformal LLM / RAG: guarantee on the generated "
            "output", ha="left", va="center", fontsize=FS_TITLE, color=ORANGE,
            fontweight="bold")
    ax.text(0.32, 3.12, "This work: guarantee on the pre-generation gate decision",
            ha="left", va="center", fontsize=FS_TITLE, color=BLUE, fontweight="bold")

    # ----------------------------------------------------------------------
    # TOP LANE  (prior work)
    # ----------------------------------------------------------------------
    yt = 4.86
    _, q_r = box(ax, 1.80, yt, 2.9, 1.14, "query $x$", icon=ic_doc)
    l_l, l_r = box(ax, 5.10, yt, 2.6, 1.14, "LLM", icon=ic_chip)
    a_l, a_r = box(ax, 8.95, yt, 3.5, 1.14, "candidate\nanswers", icon=ic_list)
    arrow(ax, q_r, yt, l_l, yt, color=GRAY_EDGE)
    arrow(ax, l_r, yt, a_l, yt, color=GRAY_EDGE)

    set_box = FancyBboxPatch(
        (11.9, yt - 0.82), 8.7, 1.64,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.6, edgecolor=ORANGE, facecolor="white",
        linestyle=(0, (4, 2)), zorder=3,
    )
    ax.add_patch(set_box)
    ax.text(16.25, yt + 0.22, "conformal prediction set", ha="center", va="center",
            fontsize=FS_BOX, color=ORANGE, fontweight="bold")
    ax.text(16.25, yt - 0.32, r"covers correct answer w.p. $1-\alpha$",
            ha="center", va="center", fontsize=FS_NOTE, color=INK)
    arrow(ax, a_r, yt, 11.9, yt, color=ORANGE)

    # ----------------------------------------------------------------------
    # BOTTOM LANE  (ours)
    # ----------------------------------------------------------------------
    yb = 1.72
    _, q2_r = box(ax, 1.80, yb, 2.9, 1.14, "query $x$", icon=ic_doc)
    s_l, s_r = box(ax, 5.55, yb, 3.6, 1.14, "ignorance\nscore $s(x)$",
                   ec=BLUE, lw=1.3, icon=ic_gauge)
    t_l, t_r = box(ax, 10.35, yb, 5.0, 1.14, r"CRC threshold $\hat{\tau}$",
                   ec=BLUE, lw=1.3, icon=ic_threshold)
    arrow(ax, q2_r, yb, s_l, yb, color=BLUE)
    arrow(ax, s_r, yb, t_l, yb, color=BLUE)

    g_x = 14.45
    gate = FancyBboxPatch(
        (g_x - 1.0, yb - 0.64), 2.0, 1.28,
        boxstyle="round,pad=0.02,rounding_size=0.6",
        linewidth=1.4, edgecolor=BLUE, facecolor="white", zorder=3,
    )
    ax.add_patch(gate)
    ax.text(g_x, yb, "gate\n$s(x)\\leq\\hat{\\tau}$?", ha="center", va="center",
            fontsize=FS_BOX, color=BLUE, fontweight="bold")
    arrow(ax, t_r, yb, g_x - 1.0, yb, color=BLUE)

    skip_y = 2.74
    retr_y = 0.74
    box(ax, 19.5, skip_y, 7.2, 0.98, "skip $\\rightarrow$ closed-book answer",
        ec=RED, tc=RED, lw=1.3, icon=ic_book, ic_color=RED)
    box(ax, 19.5, retr_y, 7.2, 0.98, "retrieve $\\rightarrow$ RAG answer",
        ec=BLUE, tc=BLUE, lw=1.3, icon=ic_magnifier, ic_color=BLUE)
    arrow(ax, g_x + 1.0, yb + 0.24, 15.9, skip_y, color=RED)
    arrow(ax, g_x + 1.0, yb - 0.24, 15.9, retr_y, color=BLUE)

    # harmful-skip annotation with a warning glyph, in the inter-lane gap
    hy = 3.52
    if ICONS:
        ic_warning(ax, 16.1, hy, 0.30, RED)
        ax.text(16.5, hy, "harmful skip = hallucination", ha="left",
                va="center", fontsize=FS_NOTE, color=RED)
    else:
        ax.text(19.5, hy, "harmful skip = hallucination", ha="center",
                va="center", fontsize=FS_NOTE, color=RED)

    # asymmetric-cost note, set below the lane with a clear gap above it
    ax.text(11.7, -0.30,
            "asymmetric cost: a harmful skip costs a hallucination; "
            "an unnecessary retrieval costs only latency",
            ha="center", va="center", fontsize=FS_NOTE, color=INK, style="italic")

    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
