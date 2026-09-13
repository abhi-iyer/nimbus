"""
dump per-task and robustness metric tables as markdown into
results/metrics_tables.txt for the LLM (NanoGPT) leg.

reads the run's summary_table.txt (per-task clean metrics) and
robustness_table.txt (input/weight noise sweeps) and re-emits them as markdown
with one column per method.

usage:
    python -m experiments.llm.figures.metrics_tables
"""
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

# task display names, in TASK_NAMES order (task_0 .. task_6).
TASK_DISPLAY = ["Early Modern English", "WikiText", "TinyStories", "Code",
                "arXiv", "Legal", "French/German"]

# eps rows retained in the robustness tables (the rest are dropped for brevity).
EPS = [0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
MODES = [("logit", "Logit"), ("softmax", "Softmax"), ("weight", "Param avg")]
IDX = {"acc": 0, "top3": 1, "top5": 2, "nll": 3, "ece": 4}   # [acc, top3, top5, nll, ece]


def _fmt(x):
    """round to <=2 decimals, dropping trailing zeros but keeping one (e.g. 1.0, 0.92)."""
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return s if "." in s else s + ".0"


def _single(name):
    return {"logit": name, "softmax": name, "weight": name}


# (display, per-task summary name [logit for ensembles], robustness name per mode).
COLUMNS = [
    ("Nimbus", "Simplex logit ens uniform degree",
     {"logit": "Simplex / logit ens uniform degree",
      "softmax": "Simplex / softmax ens uniform degree",
      "weight": "Simplex / param avg"}),
    ("Baseline ensemble ($\\lvert\\mathcal{D}\\rvert$)", "Baseline logit ens",
     {"logit": "Baseline logit ens", "softmax": "Baseline softmax ens",
      "weight": "Baseline param avg"}),
    ("Baseline ensemble ($\\lvert\\mathcal{D}\\rvert{+}N$)", "Baseline+N logit ens",
     {"logit": "Baseline+N logit ens", "softmax": "Baseline+N softmax ens",
      "weight": "Baseline+N param avg"}),
    ("Linear interpolation", "Linear logit ens",
     {"logit": "Linear logit ens", "softmax": "Linear softmax ens",
      "weight": "Linear param avg"}),
    ("TIES", "TIES merge", _single("TIES merge")),
    ("Fisher", "Fisher merge", _single("Fisher merge")),
]


def parse_summary(path):
    """parse the box-drawing table into {model: {scope: [acc, top3, top5, nll, ece]}}."""
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
    """parse into {(method, noise, eps, scope): [acc, top3, top5, nll, ece]}."""
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
    line = lambda cs: "| " + " | ".join(cs) + " |"
    return "\n".join([line(headers), line(["---"] * len(headers))] + [line(r) for r in rows])


def per_task_table(summary, metric):
    mi = IDX[metric]
    headers = ["task"] + [c[0] for c in COLUMNS]
    rows = []
    for i, disp in enumerate(TASK_DISPLAY):
        rows.append([disp] + [_fmt(summary[sname][f"task_{i}"][mi])
                              for _disp, sname, _rob in COLUMNS])
    return md_table(headers, rows)


def noise_table(rob, noise, metric):
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


def dump():
    summary = parse_summary(RESULTS_DIR / "summary_table.txt")
    rob = parse_robust(RESULTS_DIR / "robustness_table.txt")
    S = [
        "**Uniform sampling top-3 acc**:\n\n" + per_task_table(summary, "top3"),
        "**Uniform sampling NLL loss**:\n\n" + per_task_table(summary, "nll"),
        "**Input noise (token corruption) top-3 acc**:\n\n" + noise_table(rob, "input", "top3"),
        "**Input noise (token corruption) NLL loss**:\n\n" + noise_table(rob, "input", "nll"),
        "**Input noise (token corruption) ECE**:\n\n" + noise_table(rob, "input", "ece"),
        "**Weight noise top-3 acc**:\n\n" + noise_table(rob, "weight", "top3"),
        "**Weight noise NLL loss**:\n\n" + noise_table(rob, "weight", "nll"),
        "**Weight noise ECE**:\n\n" + noise_table(rob, "weight", "ece"),
    ]
    out = RESULTS_DIR / "metrics_tables.txt"
    out.write_text("\n\n\n".join(S) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    dump()
