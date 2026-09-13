"""ascii table writers for summary and robustness results."""

import math
from pathlib import Path

import numpy as np

from .metrics import METRICS


W_MET = 7


def _hline(l, m, r, w_mod, w_scope):
    """horizontal rule for the summary table borders."""
    return (l + '─' * (w_mod + 2) + m + '─' * (w_scope + 2) + m +
            '─' * (W_MET + 2) + m + '─' * (W_MET + 2) + m + '─' * (W_MET + 2) + m +
            '─' * (W_MET + 2) + m + '─' * (W_MET + 2) + r)


def _trow(mod, scope, acc, top3, top5, nll, ece, w_mod, w_scope):
    """one summary table row."""
    return (f"│ {mod:<{w_mod}} │ {scope:<{w_scope}} │ "
            f"{acc:>{W_MET}} │ {top3:>{W_MET}} │ {top5:>{W_MET}} │ "
            f"{nll:>{W_MET}} │ {ece:>{W_MET}} │")


def _fmt(v):
    """format a metric dict as (acc, top3, top5, nll, ece) strings."""
    return (f"{v['acc']:.3f}", f"{v['top3']:.3f}", f"{v['top5']:.3f}",
            f"{v['nll']:.3f}", f"{v['ece']:.3f}")


def write_summary_table(method_blocks, scopes, out_path, footer_lines=()):
    """write a box-drawn summary table of per-scope metrics for each method."""
    blocks = list(method_blocks)
    names = [n for n, _ in blocks]
    w_mod = max(len("Model"), max((len(n) for n in names), default=0))
    w_scope = max(len("Scope"), max((len(s) for s in scopes), default=0))

    lines = [
        _hline('┌', '┬', '┐', w_mod, w_scope),
        _trow('Model', 'Scope', 'Acc', 'Top3', 'Top5', 'NLL', 'ECE', w_mod, w_scope),
        _hline('├', '┼', '┤', w_mod, w_scope),
    ]
    for bi, (name, by_scope) in enumerate(blocks):
        for si, scope in enumerate(scopes):
            display = name if si == 0 else ''
            lines.append(_trow(display, scope, *_fmt(by_scope[scope]), w_mod, w_scope))
        if bi < len(blocks) - 1:
            lines.append(_hline('├', '┼', '┤', w_mod, w_scope))
    lines.append(_hline('└', '┴', '┘', w_mod, w_scope))
    if footer_lines:
        lines.append('')
        lines.extend(footer_lines)
    text = '\n'.join(lines) + '\n'
    print('\n' + text, end='')
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(text)
    print(f"  wrote {out_path}")
    return text


def _rob_line(method, noise, eps, scope, vals):
    """one robustness table row."""
    acc, top3, top5, nll, ece = vals
    return (f"{method:<48} | {noise:<10} | {eps:>6.3f} | "
            f"{scope:<8} | {acc:>7.4f} | {top3:>7.4f} | {top5:>7.4f} | "
            f"{nll:>8.4f} | {ece:>7.4f}")


def _rob_header():
    """robustness table header row."""
    return (f"{'Method':<48} | {'Noise':<10} | {'eps/p':>6} | "
            f"{'Scope':<8} | {'Acc':>7} | {'Top3':>7} | {'Top5':>7} | "
            f"{'NLL':>8} | {'ECE':>7}")


def write_robustness_table(curves, scopes, sweeps_by_kind, out_path, footer_lines=()):
    """write a robustness table: metrics per method, noise, epsilon, and scope."""
    rows = [_rob_header(), '-' * len(_rob_header())]
    for noise_label, (xs, sweeps) in sweeps_by_kind.items():
        rows.append('')
        rows.append(f"=== {noise_label} ===")
        for cname, cpool, ckey in curves:
            if cpool not in sweeps:
                continue
            for ei, x in enumerate(xs):
                for scope in scopes:
                    vals = tuple(float(sweeps[cpool][ckey][scope][m][ei]) for m in METRICS)
                    rows.append(_rob_line(cname, noise_label, float(x), scope, vals))
                rows.append('')
            rows.append('')
    if footer_lines:
        rows.extend(footer_lines)
    text = '\n'.join(rows) + '\n'
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(text)
    print(f"  wrote {out_path}")
    return text


W_STAT = 15


def _rob_stats_line(method, noise, eps, scope, mvs, svs):
    """one robustness stats row, cells formatted as mean +/- std."""
    cells = " | ".join(f"{m:.4f}±{s:.4f}".rjust(W_STAT) for m, s in zip(mvs, svs))
    return (f"{method:<48} | {noise:<10} | {eps:>6.3f} | "
            f"{scope:<8} | {cells}")


def _rob_stats_header():
    """robustness stats table header row."""
    cells = " | ".join(c.rjust(W_STAT) for c in ('Acc', 'Top3', 'Top5', 'NLL', 'ECE'))
    return (f"{'Method':<48} | {'Noise':<10} | {'eps/p':>6} | "
            f"{'Scope':<8} | {cells}")


def write_robustness_stats_table(curves, scopes, sweeps_by_kind, out_path,
                                  footer_lines=()):
    """
    write a robustness table with mean +/- std across trials per (method, scope, eps).

    sweeps_by_kind[label] is a 3-tuple (xs, mean_sweeps, std_sweeps), where each of
    mean_sweeps and std_sweeps maps pool -> ens_key -> scope -> metric -> array of
    per-eps scalars (same shape as the regular robustness table).
    """
    header = _rob_stats_header()
    rows = [header, '-' * len(header)]
    for noise_label, (xs, mean_sweeps, std_sweeps) in sweeps_by_kind.items():
        rows.append('')
        rows.append(f"=== {noise_label} ===")
        for cname, cpool, ckey in curves:
            if cpool not in mean_sweeps:
                continue
            for ei, x in enumerate(xs):
                for scope in scopes:
                    mvs = tuple(float(mean_sweeps[cpool][ckey][scope][m][ei]) for m in METRICS)
                    svs = tuple(float(std_sweeps[cpool][ckey][scope][m][ei]) for m in METRICS)
                    rows.append(_rob_stats_line(cname, noise_label, float(x), scope, mvs, svs))
                rows.append('')
            rows.append('')
    if footer_lines:
        rows.extend(footer_lines)
    text = '\n'.join(rows) + '\n'
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(text)
    print(f"  wrote {out_path}")
    return text


def fmt_metric(metric, v):
    """format one metric value, nan-safe; nll gets 2 decimals, others 3."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return " nan "
    if metric == "nll":
        return f"{v:5.2f}"
    return f"{v:5.3f}"


def with_mean(by_scope, N):
    """append a 'mean' scope averaging each metric over the N task scopes."""
    out = dict(by_scope)
    mean_vals = {}
    for m in METRICS:
        vals = [by_scope[f'task_{i}'][m] for i in range(N)
                if not (isinstance(by_scope[f'task_{i}'][m], float)
                        and math.isnan(by_scope[f'task_{i}'][m]))]
        mean_vals[m] = float(np.mean(vals)) if vals else float('nan')
    out['mean'] = mean_vals
    return out


def print_wide_table(method_rows, N, task_names=None):
    """print a wide per-task table: metric columns grouped by task, plus mean."""
    group_keys = [f'task_{i}' for i in range(N)] + ['mean']
    if task_names is None:
        task_names = [f'Task {i}' for i in range(N)]
    group_labels = list(task_names) + ['Mean']

    method_w = max(len("Method"), max(len(n) for n, _ in method_rows)) + 2
    cell_w = {m: max(len(m.upper()), 6) for m in METRICS}
    for _, per_scope in method_rows:
        for gk in group_keys:
            vals = per_scope[gk]
            for m in METRICS:
                s = fmt_metric(m, vals[m])
                cell_w[m] = max(cell_w[m], len(s))
    group_w = sum(cell_w[m] for m in METRICS) + (len(METRICS) - 1)

    def fmt_group(vals):
        return " ".join(fmt_metric(m, vals[m]).rjust(cell_w[m]) for m in METRICS)

    row1 = [" " * method_w] + [g.center(group_w) for g in group_labels]
    metric_hdr = " ".join(m.upper().rjust(cell_w[m]) for m in METRICS)
    row2 = ["Method".ljust(method_w)] + [metric_hdr for _ in group_keys]
    sep = " | "
    print(sep.join(row1))
    print(sep.join(row2))
    print("-+-".join("-" * len(p) for p in row2))
    for name, per_scope in method_rows:
        parts = [name.ljust(method_w)]
        for gk in group_keys:
            parts.append(fmt_group(per_scope[gk]))
        print(sep.join(parts))
