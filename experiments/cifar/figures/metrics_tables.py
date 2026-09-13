"""
dump per-task and robustness metric tables as markdown into
results/N={N}/metrics_tables.txt for every CIFAR experiment.

reads the run's summary_table.txt (per-task clean metrics) and
robustness_table.txt (input/weight noise sweeps) and re-emits them as markdown
with one column per method.

usage:
    python -m experiments.cifar.figures.metrics_tables            # all N
    python -m experiments.cifar.figures.metrics_tables --N 7
"""
import argparse
from pathlib import Path

from experiments.cifar.config import EXPERIMENT_CONFIGS

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# eps rows retained in the robustness tables.
EPS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
MODES = [("logit", "Logit"), ("softmax", "Softmax"), ("weight", "Param avg")]
IDX = {"acc": 0, "top3": 1, "top5": 2, "nll": 3, "ece": 4}   # [acc, top3, top5, nll, ece]


def _fmt(x):
    """round to <=2 decimals, dropping trailing zeros but keeping one (e.g. 1.0, 0.92)."""
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return s if "." in s else s + ".0"

# (display, per-task summary name, robustness name per mode). single-model
# baselines use the same name for every mode.
def _single(name):
    return {"logit": name, "softmax": name, "weight": name}

COLUMNS = [
    ("Nimbus", "Simplex logit ens uniform degree",
     {"logit": "Simplex / logit ens uniform degree",
      "softmax": "Simplex / softmax ens uniform degree",
      "weight": "Simplex / param avg"}),
    ("Baseline ensemble ($\\lvert\\mathcal{D}\\rvert$)", "Baseline logit ens",
     {"logit": "Baseline logit ens", "softmax": "Baseline softmax ens",
      "weight": "Baseline param avg"}),
    ("Baseline ensemble ($\\lvert\\mathcal{D}\\rvert+N$)", "Baseline+N logit ens",
     {"logit": "Baseline+N logit ens", "softmax": "Baseline+N softmax ens",
      "weight": "Baseline+N param avg"}),
    ("Linear interpolation", "Linear logit ens",
     {"logit": "Linear logit ens", "softmax": "Linear softmax ens",
      "weight": "Linear param avg"}),
    ("Single, 4x compute", "single4x", _single("single4x")),
    ("Fisher", "Fisher merge", _single("Fisher merge")),
    ("TIES", "TIES merge", _single("TIES merge")),
    ("PCGrad", "PCGrad", _single("PCGrad")),
    ("ASAM", "ASAM", _single("ASAM")),
    ("SWA", "SWA", _single("SWA")),
]


def parse_summary(path):
    """parse the box-drawing summary table into {model: {scope: [acc, top3, top5, nll, ece]}}."""
    out, cur = {}, None
    for line in path.read_text().splitlines():
        if not line.startswith("│"):
            continue
        cells = [c.strip() for c in line.strip("│").split("│")]
        if len(cells) < 7 or cells[1] == "Scope":
            continue
        model, scope = cells[0], cells[1]
        if model:
            cur = model
        try:
            vals = [float(c) for c in cells[2:7]]
        except ValueError:
            continue
        out.setdefault(cur, {})[scope] = vals
    return out


def parse_robust(path):
    """parse the robustness table into {(method, noise, eps, scope): [acc, top3, top5, nll, ece]}."""
    out = {}
    for line in path.read_text().splitlines():
        if "|" not in line or line.startswith("Method") or set(line.strip()) <= {"-"}:
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 9:
            continue
        try:
            eps = round(float(cells[2]), 3)
            vals = [float(c) for c in cells[4:9]]
        except ValueError:
            continue
        out[(cells[0], cells[1], eps, cells[3])] = vals
    return out


def md_table(headers, rows):
    """render a markdown table from headers and rows."""
    line = lambda cs: "| " + " | ".join(cs) + " |"
    return "\n".join([line(headers), line(["---"] * len(headers))] + [line(r) for r in rows])


def per_task_table(summary, tasks, metric):
    """markdown table of one metric per task, one column per method."""
    mi = IDX[metric]
    headers = ["task"] + [c[0] for c in COLUMNS]
    rows = []
    for t in tasks:
        row = [t] + [_fmt(summary[sname][t][mi]) for _disp, sname, _rob in COLUMNS]
        rows.append(row)
    return md_table(headers, rows)


def noise_table(rob, noise, metric):
    """markdown table of one metric vs eps and mode, one column per method."""
    mi = IDX[metric]
    headers = ["$\\varepsilon$", "mode"] + [c[0] for c in COLUMNS]
    rows = []
    for e in EPS:
        for j, (mode_key, mode_disp) in enumerate(MODES):
            row = [_fmt(e) if j == 0 else "", mode_disp]
            for _disp, _sname, robnames in COLUMNS:
                row.append(_fmt(rob[(robnames[mode_key], noise, round(e, 3), 'all')][mi]))
            rows.append(row)
    return md_table(headers, rows)


def dump(N):
    """write metrics_tables.txt for one N."""
    res = RESULTS_DIR / f"N={N}"
    summary = parse_summary(res / "summary_table.txt")
    rob = parse_robust(res / "robustness_table.txt")
    tasks = [f"task_{i}" for i in range(N)]
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
    out = res / "metrics_tables.txt"
    out.write_text("\n\n\n".join(S) + "\n")
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=None)
    args = ap.parse_args()
    Ns = [args.N] if args.N else [N for N, _, _ in EXPERIMENT_CONFIGS]
    for N in Ns:
        dump(N)


if __name__ == "__main__":
    main()
