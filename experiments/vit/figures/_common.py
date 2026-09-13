"""shared figure helpers for the ViT leg."""
from experiments.vit import config as cfg


def task_display(tasks):
    return [cfg.TASK_DISPLAY.get(t, t) for t in tasks]
