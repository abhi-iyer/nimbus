from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
import torchvision as tv
from torch.utils.data import Dataset
from torchvision import transforms

from nimbus.seeds import set_seed

from .config import DATA_DIR


def _batch_augment(x, crop_size=32, padding=4):
    """reflect-pad, per-sample random crop, and per-sample random horizontal flip."""
    x = F.pad(x, (padding, padding, padding, padding), mode="reflect")
    N, C, H, W = x.shape
    top  = torch.randint(0, H - crop_size + 1, (N,))
    left = torch.randint(0, W - crop_size + 1, (N,))
    x = x.unfold(2, crop_size, 1).unfold(3, crop_size, 1)
    x = x[torch.arange(N), :, top, left].contiguous()
    flip_mask = torch.rand(N) < 0.5
    x[flip_mask] = x[flip_mask].flip(-1)
    return x


class CIFAR100_CL(Dataset):
    """
    class-indexed CIFAR-100. item c is all images of class c; length is the
    number of classes. train/val are split per class by a seeded generator.
    """

    def __init__(self, split, seed=42, val_per_class=50):
        super().__init__()
        assert split in ("train", "val", "test")
        set_seed(seed)
        self.split = split
        self.val_per_class = val_per_class

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                 (0.2675, 0.2565, 0.2761)),
        ])

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.dataset = tv.datasets.CIFAR100(
            str(DATA_DIR),
            train=(split != "test"),
            transform=transform,
            download=True,
        )

        targets = torch.tensor(self.dataset.targets)
        self.class_indices = [torch.where(targets == c)[0]
                              for c in range(self.num_classes)]

        split_gen = torch.Generator().manual_seed(seed)
        self.data, self.targets = [], []

        for c in range(self.num_classes):
            idxs = self.class_indices[c]
            if split in ("train", "val"):
                perm = torch.randperm(len(idxs), generator=split_gen)
                idxs = idxs[perm]
                idxs = idxs[:val_per_class] if split == "val" else idxs[val_per_class:]

            imgs, lbls = [], []
            for i in idxs:
                img, lbl = self.dataset[i]
                imgs.append(img); lbls.append(lbl)
            self.data.append(torch.stack(imgs))
            self.targets.append(torch.tensor(lbls))

    @property
    def num_classes(self):
        return len(self.dataset.classes)

    def __len__(self):
        return self.num_classes

    def __getitem__(self, c):
        d = self.data[c]
        if self.split == "train":
            d = _batch_augment(d)
        return d, self.targets[c]


class CL_DataLoader:
    """
    loader over a class-indexed dataset. set_epoch_data selects task classes,
    then iterating concatenates their images and yields batches.
    """

    def __init__(self, dataset, batch_size, *, shuffle=True, seed=42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self._generator = torch.Generator().manual_seed(seed)
        self._tasks = None

    def __deepcopy__(self, memo):
        new = CL_DataLoader(self.dataset, self.batch_size, shuffle=self.shuffle, seed=self.seed)
        new._tasks = deepcopy(self._tasks, memo)
        return new

    def reset_shuffle_state(self):
        """reseed the shuffle generator."""
        self._generator = torch.Generator().manual_seed(self.seed)

    @staticmethod
    def _to_int(x):
        if isinstance(x, torch.Tensor):
            return int(x.item())
        return int(x)

    def _parse_tasks(self, task_labels):
        if isinstance(task_labels, (int, np.integer)) and not isinstance(task_labels, bool):
            return [[self._to_int(task_labels)]]
        if isinstance(task_labels, torch.Tensor) and task_labels.ndim == 0:
            return [[self._to_int(task_labels)]]
        if isinstance(task_labels, torch.Tensor):
            task_labels = task_labels.view(-1).tolist()
        elif isinstance(task_labels, np.ndarray):
            task_labels = task_labels.reshape(-1).tolist()
        elif isinstance(task_labels, tuple):
            task_labels = list(task_labels)
        elif not isinstance(task_labels, list):
            raise TypeError(f"unsupported task_labels type: {type(task_labels)}")
        if not task_labels:
            raise ValueError("empty task_labels not allowed")
        if isinstance(task_labels[0], (list, tuple)):
            return [[self._to_int(c) for c in t] for t in task_labels]
        return [[self._to_int(c) for c in task_labels]]

    def set_epoch_data(self, task_labels):
        """select the classes (one or more tasks) to iterate over."""
        self._tasks = self._parse_tasks(task_labels)

    def _get_task_data(self, task_classes):
        parts_x, parts_y = [], []
        for c in task_classes:
            d, t = self.dataset[c]
            parts_x.append(d); parts_y.append(t)
        full_x = torch.cat(parts_x, dim=0)
        full_y = torch.cat(parts_y, dim=0)
        if self.shuffle and full_x.size(0) > 1:
            perm = torch.randperm(full_x.size(0), generator=self._generator)
            full_x = full_x[perm]; full_y = full_y[perm]
        return full_x, full_y

    def __iter__(self):
        assert self._tasks is not None, "call set_epoch_data() before iterating"
        for task_classes in self._tasks:
            full_x, full_y = self._get_task_data(task_classes)
            for s in range(0, full_x.size(0), self.batch_size):
                yield full_x[s:s + self.batch_size], full_y[s:s + self.batch_size]
