"""
visualize input perturbation as scrolls, one per (task, p) pair.

one column per task (N=7), laid out horizontally with p climbing left to right.
each column is a pastel rounded rectangle ("scroll") containing a tokenized
snippet wrapped to ~9 lines. corrupted positions are shown as white text on a
red highlight. task name and p value sit below each scroll.

illustrates corrupt_tokens in experiments/llm/perturbations.py: each position is
independently replaced by a random vocab token id with probability p. replacements
here are drawn only from tokens that decode without a newline, and the snippet is
re-encoded with a greedy BPE rather than the heap encoder in tokenizer.py.
"""
import json
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.patheffects import withSimplePatchShadow

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

FIG_DIR = Path(__file__).resolve().parent
DATA_DIR = FIG_DIR.parent / "data_dir"
TEXTS_DIR = DATA_DIR / "texts"
TOK_PATH = DATA_DIR / "tokenizer.json"
OUT_DIR = FIG_DIR


def load_bpe(path):
    with open(path) as f:
        data = json.load(f)
    vocab = {i: bytes([i]) for i in range(256)}
    merges = {}
    for a, b, v in data["merges"]:
        merges[(a, b)] = v
        vocab[v] = vocab[a] + vocab[b]
    return merges, vocab


def encode_bpe(text, merges):
    ids = list(text.encode("utf-8"))
    while True:
        best_id = None
        best_pair = None
        for i in range(len(ids) - 1):
            m = merges.get((ids[i], ids[i + 1]))
            if m is not None and (best_id is None or m < best_id):
                best_id = m
                best_pair = (ids[i], ids[i + 1])
        if best_id is None:
            break
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best_pair:
                new_ids.append(best_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        ids = new_ids
    return ids


def decode_token(vocab, tid):
    return vocab[tid].decode("utf-8", errors="replace")


MERGES, VOCAB = load_bpe(TOK_PATH)
VOCAB_SIZE = len(VOCAB)
# vocab ids whose decoded bytes contain no newline, used for source
# normalization and for drawing corruption replacements.
VALID_VOCAB_IDS = np.array(
    [tid for tid in range(VOCAB_SIZE)
     if b"\n" not in VOCAB[tid] and b"\r" not in VOCAB[tid]]
)

# one column per task (N=7), with p climbing left to right.
TASKS = ["early_modern_english", "wikitext", "tinystories", "code",
         "arxiv", "legal", "multilingual"]
DISPLAY = ["Early Modern English", "WikiText", "TinyStories", "Code",
           "arXiv", "Legal", "French/German"]
P_VALUES = [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
P_LABELS = ["0.0", "0.05", "0.1", "0.25", "0.5", "0.75", "1.0"]

OFFSETS = {
    "early_modern_english": 200_000,
    "wikitext": 5_000,
    "tinystories": 2_000,
    "code": 30_000,
    "arxiv": 200_000,
    "legal": 200_000,
    "multilingual": 200_000,
}
N_TOKENS = 240


def get_token_ids(task, n=N_TOKENS):
    text = (TEXTS_DIR / f"{task}.txt").read_text(encoding="utf-8", errors="ignore")
    start = OFFSETS[task]
    chunk = text[start:start + 6000]
    # collapse any whitespace run into a single space so no source token
    # decodes to a string containing a newline.
    chunk = re.sub(r"\s+", " ", chunk).strip()
    ws = chunk.find(" ")
    if ws > 0:
        chunk = chunk[ws + 1:]
    ids = encode_bpe(chunk[:3500], MERGES)
    return ids[:n]


RNG = np.random.default_rng(7)


def corrupt(ids, p):
    ids_arr = np.array(ids)
    mask = RNG.random(len(ids)) < p
    # draw corruption replacements only from tokens that contain no newline.
    draw_idx = RNG.integers(0, len(VALID_VOCAB_IDS), size=len(ids))
    rand = VALID_VOCAB_IDS[draw_idx]
    return np.where(mask, rand, ids_arr).tolist(), mask.tolist()


def sanitize(s):
    out = []
    for ch in s:
        cp = ord(ch)
        if ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif cp < 0x20 or cp == 0x7F or cp == 0xFFFD:
            out.append("·")
        else:
            out.append(ch)
    return "".join(out)


# --- colors ----------------------------------------------------------------
SCROLL_BG = "#edf1f7"      # soft pastel blue-gray
SCROLL_BORDER = "#b9c4d4"
HIGHLIGHT_COLOR = "#d9443a"  # muted vermillion
TEXT_COLOR = "#1a1a1a"       # near-black
HIGHLIGHT_TEXT = "#ffffff"   # white on red

# --- layout (specified in inches, converted to axes coords) ----------------
FIG_W = 15.5          # widened for N=7 columns
FS = 9
MAX_LINES = 9
LINE_H_IN = FS * 1.32 / 72.0
INNER_PAD_Y_IN = 0.08
INNER_PAD_X_IN = 0.08

TOP_MARGIN_IN = 0.05
CAPTION_H_IN = 0.18
CAPTION_GAP_IN = 0.08
SCROLL_H_IN = MAX_LINES * LINE_H_IN + 2 * INNER_PAD_Y_IN
SCROLL_TO_TASK_IN = 0.14
TASK_H_IN = 0.18
TASK_TO_P_IN = 0.05
P_H_IN = 0.15
BOTTOM_MARGIN_IN = 0.10

FIG_H = (TOP_MARGIN_IN + CAPTION_H_IN + CAPTION_GAP_IN + SCROLL_H_IN
         + SCROLL_TO_TASK_IN + TASK_H_IN + TASK_TO_P_IN + P_H_IN
         + BOTTOM_MARGIN_IN)

fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
ax.patch.set_alpha(0)
for s in ax.spines.values():
    s.set_visible(False)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
_inv = ax.transAxes.inverted()


def px_to_axesx(px):
    return _inv.transform((px, 0))[0] - _inv.transform((0, 0))[0]


def px_to_axesy(px):
    return _inv.transform((0, px))[1] - _inv.transform((0, 0))[1]


def y_from_top(inches_from_top):
    return 1.0 - inches_from_top / FIG_H


# horizontal: one column per task, spanning the full figure width.
N_COLS = len(TASKS)
COL_GAP = 0.014
COL_W = (1 - COL_GAP * (N_COLS + 1)) / N_COLS
COL_CENTERS = [COL_GAP + COL_W / 2 + i * (COL_W + COL_GAP) for i in range(N_COLS)]

# vertical anchors (axes y = [0,1], 1 = top).
CAPTION_Y = y_from_top(TOP_MARGIN_IN)
SCROLL_TOP = y_from_top(TOP_MARGIN_IN + CAPTION_H_IN + CAPTION_GAP_IN)
SCROLL_BOT = y_from_top(TOP_MARGIN_IN + CAPTION_H_IN + CAPTION_GAP_IN + SCROLL_H_IN)
LABEL_TASK_Y = y_from_top(TOP_MARGIN_IN + CAPTION_H_IN + CAPTION_GAP_IN + SCROLL_H_IN
                          + SCROLL_TO_TASK_IN + TASK_H_IN / 2)
LABEL_P_Y = y_from_top(TOP_MARGIN_IN + CAPTION_H_IN + CAPTION_GAP_IN + SCROLL_H_IN
                       + SCROLL_TO_TASK_IN + TASK_H_IN + TASK_TO_P_IN + P_H_IN / 2)

LINE_H = LINE_H_IN / FIG_H
INNER_PAD_X = INNER_PAD_X_IN / FIG_W
INNER_PAD_Y = INNER_PAD_Y_IN / FIG_H
SCROLL_W = COL_W * 0.97

# top caption, drawn outside the scrolls.
ax.text(
    0.5, CAPTION_Y,
    "Input perturbation: each token is independently replaced by a uniformly "
    "random vocabulary token with probability p    (red highlight = corrupted)",
    fontsize=10, transform=ax.transAxes, va="top", ha="center", family="sans-serif",
)

shadow = withSimplePatchShadow(
    offset=(1.8, -1.8), shadow_rgbFace=(0.18, 0.20, 0.26), alpha=0.14
)

for i, (task, disp, p, p_label) in enumerate(zip(TASKS, DISPLAY, P_VALUES, P_LABELS)):
    cx = COL_CENTERS[i]
    sx = cx - SCROLL_W / 2
    sy = SCROLL_BOT
    sh = SCROLL_TOP - SCROLL_BOT

    box = FancyBboxPatch(
        (sx, sy), SCROLL_W, sh,
        boxstyle="round,pad=0.0,rounding_size=0.018",
        facecolor=SCROLL_BG,
        edgecolor=SCROLL_BORDER,
        linewidth=0.6,
        transform=ax.transAxes,
        zorder=1,
    )
    box.set_path_effects([shadow])
    ax.add_patch(box)

    ids = get_token_ids(task)
    cor, mask = corrupt(ids, p)

    inner_left = sx + INNER_PAD_X
    inner_right = sx + SCROLL_W - INNER_PAD_X
    y_top = SCROLL_TOP - INNER_PAD_Y

    # create token Text artists offscreen.
    text_objs = []
    for cid, is_cor in zip(cor, mask):
        s = sanitize(decode_token(VOCAB, cid))
        s = s.replace("$", r"\$")
        color = HIGHLIGHT_TEXT if is_cor else TEXT_COLOR
        t = ax.text(-1, -1, s, fontsize=FS, color=color,
                    transform=ax.transAxes, va="top", ha="left",
                    family="sans-serif", zorder=3)
        text_objs.append(t)

    # measure widths, then reposition with greedy wrapping.
    fig.canvas.draw()
    widths = [px_to_axesx(t.get_window_extent(renderer=renderer).width)
              for t in text_objs]

    cur_x = inner_left
    cur_line = 0
    for t, w in zip(text_objs, widths):
        if cur_x + w > inner_right and cur_x > inner_left + 1e-9:
            cur_line += 1
            cur_x = inner_left
            if cur_line >= MAX_LINES:
                t.set_visible(False)
                continue
        if cur_line >= MAX_LINES:
            t.set_visible(False)
            continue
        t.set_position((cur_x, y_top - cur_line * LINE_H))
        cur_x += w

    # measure final token bboxes and paint red highlight rectangles under the
    # corrupted tokens, inset horizontally and expanded vertically.
    fig.canvas.draw()
    inset_x_ax = px_to_axesx(1.2)
    pad_y_ax = px_to_axesy(1.2)
    for t, is_cor in zip(text_objs, mask):
        if not is_cor or not t.get_visible():
            continue
        bb = t.get_window_extent(renderer=renderer)
        if bb.width <= 0 or bb.height <= 0:
            continue
        x0, y0 = _inv.transform((bb.xmin, bb.ymin))
        x1, y1 = _inv.transform((bb.xmax, bb.ymax))
        rect_w = (x1 - x0) - 2 * inset_x_ax
        if rect_w <= 0:
            continue
        rect = Rectangle(
            (x0 + inset_x_ax, y0 - pad_y_ax),
            rect_w,
            (y1 - y0) + 2 * pad_y_ax,
            facecolor=HIGHLIGHT_COLOR, edgecolor="none",
            transform=ax.transAxes, zorder=2,
        )
        ax.add_patch(rect)

    # task label + p label (beneath the scroll).
    ax.text(cx, LABEL_TASK_Y, disp, fontsize=10, weight="bold",
            transform=ax.transAxes, va="center", ha="center", family="sans-serif")
    ax.text(cx, LABEL_P_Y, f"p = {p_label}", fontsize=10, color="#555555",
            transform=ax.transAxes, va="center", ha="center", family="sans-serif")

out_path = OUT_DIR / "input_perturbation_viz.pdf"
fig.savefig(out_path, bbox_inches="tight", pad_inches=0.15)
print(f"saved {out_path}")
