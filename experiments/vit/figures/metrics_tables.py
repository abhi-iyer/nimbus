"""dump per-task + robustness metric tables as markdown into
results/metrics_tables.txt for the ViT experiment.

reads the run's summary_table.txt + robustness_table.txt (written by
experiments.vit.run via experiments.vit.tables) and re-emits them as markdown
with one column per method.

    python -m experiments.vit.figures.metrics_tables
"""
import argparse

from experiments.vit import config as cfg
from experiments.vit import tables as RT

# eps rows retained in the robustness tables.
EPS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
MODES = [("logit avg", "Logit"), ("softmax avg", "Softmax"), ("param avg", "Param avg")]
MI = RT.METRIC_IDX


def _fmt(x):
    """round to <=2 decimals, dropping trailing zeros but keeping one (e.g. 1.0, 0.92)."""
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return s if "." in s else s + ".0"


def _md(headers, rows):
    line = lambda cs: "| " + " | ".join(cs) + " |"
    return "\n".join([line(headers), line(["---"] * len(headers))] + [line(r) for r in rows])


# (display, per-task summary row [logit for ensembles], {mode_tag: robustness row}).
def _ens(disp):
    return (disp, f"{disp}, logit avg",
            {tag: f"{disp}, {tag}" for tag, _ in MODES})


def _single(disp):
    return (disp, disp, {tag: disp for tag, _ in MODES})


COLUMNS = [
    _ens("Nimbus"),
    _ens("Baseline ens"),
    _ens("Baseline+N ens"),
    _ens("Linear interp"),
    _single("Task Arithmetic"),
    _single("TIES"),
    _single("Fisher"),
]


def _at(d, e):
    for k, v in d.items():
        if abs(k - e) < 1e-6:
            return v
    raise KeyError(e)


def per_task_table(summary, tasks, metric):
    mi = MI[metric]
    headers = ["task"] + [c[0] for c in COLUMNS]
    rows = [[t] + [_fmt(summary[sname][t][mi]) for _d, sname, _r in COLUMNS]
            for t in tasks]
    return _md(headers, rows)


def noise_table(rob, noise, metric):
    mi = MI[metric]
    headers = ["$\\varepsilon$", "mode"] + [c[0] for c in COLUMNS]
    rows = []
    for e in EPS:
        for j, (tag, disp) in enumerate(MODES):
            row = [_fmt(e) if j == 0 else "", disp]
            for _d, _s, robnames in COLUMNS:
                row.append(_fmt(_at(rob[(robnames[tag], noise)], e)[mi]))
            rows.append(row)
    return _md(headers, rows)


def dump():
    summary = RT.parse_summary(cfg.RESULTS_DIR / "summary_table.txt")
    rob = RT.parse_robustness(cfg.RESULTS_DIR / "robustness_table.txt")
    tasks = [s for s in next(iter(summary.values())) if s != "all"]
    S = [
        "**Uniform sampling top-3 acc**:\n\n" + per_task_table(summary, tasks, "top3"),
        "**Uniform sampling NLL loss**:\n\n" + per_task_table(summary, tasks, "nll"),
        "**Input noise top-3 acc**:\n\n" + noise_table(rob, "input", "top3"),
        "**Input noise NLL loss**:\n\n" + noise_table(rob, "input", "nll"),
        "**Input noise ECE**:\n\n" + noise_table(rob, "input", "ece"),
        "**Weight noise top-3 acc**:\n\n" + noise_table(rob, "weight", "top3"),
        "**Weight noise NLL loss**:\n\n" + noise_table(rob, "weight", "nll"),
        "**Weight noise ECE**:\n\n" + noise_table(rob, "weight", "ece"),
    ]
    out = cfg.RESULTS_DIR / "metrics_tables.txt"
    out.write_text("\n\n\n".join(S) + "\n")
    print("wrote", out)


def main():
    argparse.ArgumentParser().parse_args()
    dump()


if __name__ == "__main__":
    main()
