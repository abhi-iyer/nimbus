"""perturbation_viz.pdf: what the input-noise x-axis physically is.

a single compressed row (the diagonal of the task x eps grid): one sample image
per task, each shown at a steadily increasing input-noise eps. every cell is a
rounded card with a soft drop shadow; the task name and eps sit beneath it.

perturbation matches experiments/vit/eval.py::_perturb_input: gaussian noise
rescaled so its L2 norm is eps * ||x|| per image (relative-L2).
"""
import argparse

import torch
import matplotlib as mpl
from matplotlib.patches import FancyBboxPatch
from matplotlib.patheffects import withSimplePatchShadow

from experiments.vit import config as cfg, data as data
from experiments.vit.eval import _perturb_input

# eps climbing 0 -> 1 across the tasks (one value per column).
DEFAULT_EPS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]

# --- colors ---
CARD_BORDER = "#b9c4d4"   # soft blue-gray
TEXT_COLOR = "#1a1a1a"    # near-black
EPS_COLOR = "#555555"     # muted gray

# --- layout (inches) ---
CELL_IN = 1.7             # square image card
COL_GAP_IN = 0.16
SIDE_MARGIN_IN = 0.12
CORNER_IN = 0.11          # rounded-corner radius

TOP_MARGIN_IN = 0.05
CAPTION_H_IN = 0.20
CAPTION_GAP_IN = 0.12
CELL_TO_TASK_IN = 0.14
TASK_H_IN = 0.20
TASK_TO_EPS_IN = 0.04
EPS_H_IN = 0.16
BOTTOM_MARGIN_IN = 0.08


def _setup_mpl():
    mpl.use("Agg")
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = [
        "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    import matplotlib.pyplot as plt
    return plt


def _denorm(x):
    mean = torch.tensor(cfg.IMAGE_MEAN).view(3, 1, 1)
    std = torch.tensor(cfg.IMAGE_STD).view(3, 1, 1)
    return (x.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def _sample_image(task):
    """one denormalized test image for task, or None if unavailable locally."""
    try:
        loaders = data.task_loaders(task, batch_size=8, num_workers=0)
        x, _ = next(iter(loaders["test"]))
    except Exception as e:  # dataset not present on this machine
        print(f"skip {task}: {e}")
        return None
    return x[0:1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, nargs="*", default=None,
                    help="one eps per task (defaults to a 0->1 sweep)")
    args = ap.parse_args()
    plt = _setup_mpl()

    tasks = list(cfg.TASK_NAMES)
    epslist = args.eps if args.eps is not None else DEFAULT_EPS
    # pair each task with its own (increasing) eps: the grid diagonal.
    n = min(len(tasks), len(epslist))
    tasks, epslist = tasks[:n], epslist[:n]

    gen = torch.Generator(device="cpu").manual_seed(cfg.GLOBAL_SEED)

    # gather images first so missing datasets drop their column.
    cells = []
    for t, eps in zip(tasks, epslist):
        img = _sample_image(t)
        if img is None:
            continue
        noisy = _perturb_input(img, eps, gen)
        cells.append((t, eps, _denorm(noisy[0])))
    if not cells:
        raise SystemExit("no datasets available locally to render")

    ncols = len(cells)
    fig_w = (2 * SIDE_MARGIN_IN + ncols * CELL_IN + (ncols - 1) * COL_GAP_IN)
    fig_h = (TOP_MARGIN_IN + CAPTION_H_IN + CAPTION_GAP_IN + CELL_IN
             + CELL_TO_TASK_IN + TASK_H_IN + TASK_TO_EPS_IN + EPS_H_IN
             + BOTTOM_MARGIN_IN)

    fig = plt.figure(figsize=(fig_w, fig_h))

    # overlay axes spanning the whole figure: caption, shadows, labels.
    ax_bg = fig.add_axes([0, 0, 1, 1])
    ax_bg.set_xlim(0, 1)
    ax_bg.set_ylim(0, 1)
    ax_bg.axis("off")
    ax_bg.patch.set_alpha(0)

    def yf(inches_from_top):
        return 1.0 - inches_from_top / fig_h

    caption_y = yf(TOP_MARGIN_IN)
    cell_top = TOP_MARGIN_IN + CAPTION_H_IN + CAPTION_GAP_IN
    cell_bot_in = cell_top + CELL_IN
    task_y = yf(cell_bot_in + CELL_TO_TASK_IN + TASK_H_IN / 2)
    eps_y = yf(cell_bot_in + CELL_TO_TASK_IN + TASK_H_IN + TASK_TO_EPS_IN
               + EPS_H_IN / 2)

    ax_bg.text(
        0.5, caption_y,
        r"Input perturbation: Gaussian noise added at relative-L2 magnitude "
        r"$\varepsilon$   (one test image per task, $\varepsilon$ climbing "
        r"left$\rightarrow$right)",
        fontsize=11, transform=ax_bg.transAxes, va="top", ha="center",
        family="sans-serif",
    )

    shadow = withSimplePatchShadow(
        offset=(1.8, -1.8), shadow_rgbFace=(0.18, 0.20, 0.26), alpha=0.16)

    corner_ax = CORNER_IN / CELL_IN  # rounded radius in per-cell axes fraction

    for i, (task, eps, arr) in enumerate(cells):
        left_in = SIDE_MARGIN_IN + i * (CELL_IN + COL_GAP_IN)
        left_f = left_in / fig_w
        w_f = CELL_IN / fig_w
        bot_f = yf(cell_bot_in)
        h_f = CELL_IN / fig_h
        cx = left_f + w_f / 2

        # soft drop shadow drawn on the full-figure axes so it isn't clipped.
        shadow_patch = FancyBboxPatch(
            (left_f, bot_f), w_f, h_f,
            boxstyle=f"round,pad=0,rounding_size={CORNER_IN / fig_w}",
            facecolor="white", edgecolor="none",
            transform=ax_bg.transAxes, zorder=1,
        )
        shadow_patch.set_path_effects([shadow])
        ax_bg.add_patch(shadow_patch)

        # image in its own square axes -> rounded corners for clip + border.
        cell_ax = fig.add_axes([left_f, bot_f, w_f, h_f], zorder=2)
        cell_ax.axis("off")
        im = cell_ax.imshow(arr, extent=(0, 1, 0, 1), aspect="auto",
                            interpolation="lanczos", zorder=2)
        clip = FancyBboxPatch(
            (0, 0), 1, 1,
            boxstyle=f"round,pad=0,rounding_size={corner_ax}",
            transform=cell_ax.transAxes,
        )
        im.set_clip_path(clip)
        border = FancyBboxPatch(
            (0, 0), 1, 1,
            boxstyle=f"round,pad=0,rounding_size={corner_ax}",
            facecolor="none", edgecolor=CARD_BORDER, linewidth=0.8,
            transform=cell_ax.transAxes, zorder=3,
        )
        cell_ax.add_patch(border)

        disp = cfg.TASK_DISPLAY.get(task, task)
        ax_bg.text(cx, task_y, disp, fontsize=11, weight="bold",
                   transform=ax_bg.transAxes, va="center", ha="center",
                   family="sans-serif", color=TEXT_COLOR)
        ax_bg.text(cx, eps_y, rf"$\varepsilon$ = {eps:g}", fontsize=10,
                   color=EPS_COLOR, transform=ax_bg.transAxes,
                   va="center", ha="center", family="sans-serif")

    out = cfg.FIGURES_DIR / "perturbation_viz.pdf"
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()
