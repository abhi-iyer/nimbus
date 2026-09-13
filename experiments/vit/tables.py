"""ViT result tables: the .txt tables, derived from the in-memory res dict that
run.py's headline eval builds.

a headline run writes these tables directly and the figures parse them. this
module is the single source of truth for the row naming, used by both the writer
(run.py) and the parsers (the figures).
"""
from pathlib import Path

from nimbus.metrics import METRICS
from nimbus.tables import (
    write_summary_table, write_robustness_table, write_robustness_stats_table,
)

METRIC_IDX = {m: i for i, m in enumerate(METRICS)}   # acc,top3,top5,nll,ece -> 0..4

# clean-method key -> display, for the ensemble methods (one row per aggregation).
_ENS = {
    "nimbus_face":      "Nimbus",
    "baseline_ens":     "Baseline ens",
    "baseline_ens_dpn": "Baseline+N ens",
    "linear_interp":    "Linear interp",
}
# single-model methods: one bare row from the 'single' mode.
_SINGLE = {
    "task_arithmetic": "Task Arithmetic",
    "ties":            "TIES",
    "fisher":          "Fisher",
}
_MODE_TAG = {"logit_avg": "logit avg", "softmax_avg": "softmax avg",
             "weight_avg": "param avg"}
# robust[] keys use "nimbus" for the nimbus_face pool; everything else matches.
_ROBUST_BASE = {"nimbus_face": "nimbus"}

# display order for the tables (ensembles by aggregation, then merges).
_ENS_ORDER = ["nimbus_face", "baseline_ens", "baseline_ens_dpn", "linear_interp"]
_SINGLE_ORDER = ["task_arithmetic", "ties", "fisher"]
_MODE_ORDER = ["logit_avg", "softmax_avg", "weight_avg"]


def summary_rowname(method, mode):
    """row label for an ensemble (method, mode) or a single-model method."""
    if method in _SINGLE:
        return _SINGLE[method]
    return f"{_ENS[method]}, {_MODE_TAG[mode]}"


# robust[] base key -> display (the nimbus_face pool keys robust under "nimbus").
_ROBUST_DISP = {"nimbus": "Nimbus", "baseline_ens": "Baseline ens",
                "baseline_ens_dpn": "Baseline+N ens", "linear_interp": "Linear interp",
                "task_arithmetic": "Task Arithmetic", "ties": "TIES", "fisher": "Fisher"}
_ROBUST_SINGLE = {"task_arithmetic", "ties", "fisher"}


def robust_cname(base, mode):
    """
    row label used in the robustness table for a robust-base + aggregation mode
    (single-model merges collapse to their bare name).
    """
    if base in _ROBUST_SINGLE:
        return _ROBUST_DISP[base]
    return f"{_ROBUST_DISP[base]}, {_MODE_TAG[mode]}"


def _rows():
    """ordered (method, mode, rowname): ensembles x 3 modes, then single models."""
    for m in _ENS_ORDER:
        for mode in _MODE_ORDER:
            yield m, mode, summary_rowname(m, mode)
    for m in _SINGLE_ORDER:
        yield m, "single", summary_rowname(m, "single")


# --- writing (run.py) -------------------------------------------------------
def write_tables(res, out_dir):
    out_dir = Path(out_dir)
    scopes = ["all"] + list(res["tasks"])
    footer = ["", "Scopes: all + per-task; tasks = " + ", ".join(res["tasks"])]

    blocks = [(name, {s: res["clean"][m][mode][s] for s in scopes})
              for m, mode, name in _rows()]
    write_summary_table(blocks, scopes, out_dir / "summary_table.txt",
                        footer_lines=footer)

    # robustness is all-scope only. fold (method, mode) into the row/pool name;
    # key='v', scope='all' so the nimbus.tables nesting resolves.
    eps = res["robust"]["nimbus_input"]["logit_avg"]["eps"]
    curves = [(name, name, "v") for _, _, name in _rows()]

    def sweeps(noise, stat):
        out = {}
        for m, mode, name in _rows():
            block = res["robust"][f"{_ROBUST_BASE.get(m, m)}_{noise}"][mode]
            out[name] = {"v": {"all": {k: block[f"{k}_{stat}"] for k in METRICS}}}
        return out

    write_robustness_table(
        curves, ["all"],
        {"input": (eps, sweeps("input", "mean")),
         "weight": (eps, sweeps("weight", "mean"))},
        out_dir / "robustness_table.txt", footer_lines=footer)
    write_robustness_stats_table(
        curves, ["all"],
        {"input": (eps, sweeps("input", "mean"), sweeps("input", "std")),
         "weight": (eps, sweeps("weight", "mean"), sweeps("weight", "std"))},
        out_dir / "robustness_stats_table.txt", footer_lines=footer)


# --- parsing (figure scripts) ----------------------------------------------
def parse_summary(path):
    """parse summary_table.txt -> {rowname: {scope: [acc, top3, top5, nll, ece]}}."""
    out, current = {}, None
    for line in Path(path).read_text().splitlines():
        if not line.startswith("│"):
            continue
        cells = [c.strip() for c in line.strip("│").split("│")]
        if len(cells) < 7:
            continue
        model, scope = cells[0], cells[1]
        try:
            vals = [float(c) for c in cells[2:7]]
        except ValueError:
            continue
        if model:
            current = model
        if current is None:
            continue
        out.setdefault(current, {})[scope] = vals
    return out


def parse_robustness(path):
    """parse robustness_table.txt -> {(rowname, noise): {eps: [acc, top3, top5, nll, ece]}}."""
    out = {}
    for line in Path(path).read_text().splitlines():
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 9:
            continue
        method, noise, eps_str, scope = cells[0], cells[1], cells[2], cells[3]
        try:
            eps = float(eps_str)
            vals = [float(c) for c in cells[4:9]]
        except ValueError:
            continue
        out.setdefault((method, noise), {})[eps] = vals
    return out
