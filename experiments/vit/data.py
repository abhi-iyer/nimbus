"""per-task image data: the 8 benchmark datasets.

each task exposes train / val / test splits of CLIP-preprocessed images plus
its class names in label order. tasks have disjoint label spaces, so everything
is per-task; the all-task scope is aggregated across per-task evals downstream.
"""
from functools import lru_cache

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms

from nimbus.seeds import device

from .config import (
    TASK_SPECS, DATA_DIR, IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD,
    BATCH_SIZE, VAL_FRAC, MAX_TRAIN_PER_TASK, GLOBAL_SEED,
)


def build_transform():
    """CLIP preprocessing: RGB, resize + center-crop to 224, normalize."""
    return transforms.Compose([
        transforms.Lambda(lambda im: im.convert("RGB")),
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
    ])


# --- raw torchvision dataset construction per task ---

def _tv_dataset(tv_name, root, split, transform):
    """
    torchvision dataset for split in {'train', 'test'}. datasets without a
    native test split (EuroSAT) are handled by the caller.
    """
    import torchvision.datasets as D
    root = str(root)
    if tv_name == "MNIST":
        return D.MNIST(root, train=(split == "train"), download=True, transform=transform)
    if tv_name == "SVHN":
        return D.SVHN(root, split=("train" if split == "train" else "test"),
                      download=True, transform=transform)
    if tv_name == "GTSRB":
        return D.GTSRB(root, split=("train" if split == "train" else "test"),
                       download=True, transform=transform)
    if tv_name == "DTD":
        return D.DTD(root, split=("train" if split == "train" else "test"),
                     partition=1, download=True, transform=transform)
    if tv_name == "EuroSAT":
        return D.EuroSAT(root, download=True, transform=transform)
    raise ValueError(f"unsupported torchvision dataset {tv_name}")


class HFImageDataset(torch.utils.data.Dataset):
    """
    adapts a HuggingFace image split (e.g. timm/resisc45) to the torchvision
    (image_tensor, int_label) interface, exposing .classes in label order.
    """

    def __init__(self, hf_split, transform):
        self.hf = hf_split
        self.tf = transform
        self._lab = next(k for k in hf_split.features if "label" in k.lower())
        self.classes = list(hf_split.features[self._lab].names)

    def __len__(self):
        return len(self.hf)

    def __getitem__(self, i):
        ex = self.hf[i]
        return self.tf(ex["image"]), int(ex[self._lab])


def _split_train_val(train_full):
    """
    carve a deterministic VAL_FRAC validation subset out of train, capping
    train size at MAX_TRAIN_PER_TASK.
    """
    n = len(train_full)
    n_val = max(1, int(round(n * VAL_FRAC)))
    g = torch.Generator().manual_seed(GLOBAL_SEED)
    perm = torch.randperm(n, generator=g).tolist()
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    if len(train_idx) > MAX_TRAIN_PER_TASK:
        train_idx = train_idx[:MAX_TRAIN_PER_TASK]
    return Subset(train_full, train_idx), Subset(train_full, val_idx)


def _raw_splits(task, transform):
    """(train, val, test) datasets for task."""
    spec = TASK_SPECS[task]
    kind, name = spec["loader"]
    if kind == "hf":
        from datasets import load_dataset
        ds = load_dataset(name)
        test_split = "test" if "test" in ds else "validation"
        train_full = HFImageDataset(ds["train"], transform)
        test = HFImageDataset(ds[test_split], transform)
        train, val = _split_train_val(train_full)
        return train, val, test
    if kind != "tv":
        raise NotImplementedError(f"{task}: loader kind {kind!r} not supported")
    if name == "EuroSAT":
        full = _tv_dataset(name, DATA_DIR / task, "train", transform)
        g = torch.Generator().manual_seed(GLOBAL_SEED)
        n = len(full)
        n_test = int(round(0.2 * n))
        n_val = max(1, int(round(VAL_FRAC * (n - n_test))))
        n_train = n - n_test - n_val
        train, val, test = random_split(full, [n_train, n_val, n_test], generator=g)
        return train, val, test
    train_full = _tv_dataset(name, DATA_DIR / task, "train", transform)
    test = _tv_dataset(name, DATA_DIR / task, "test", transform)
    train, val = _split_train_val(train_full)
    return train, val, test


# --- class names ---

@lru_cache(maxsize=None)
def classnames(task):
    """class names in label order (explicit spec, else dataset.classes)."""
    spec = TASK_SPECS[task]
    if spec["classnames"] is not None:
        return list(spec["classnames"])
    # derive from the dataset label order.
    kind, name = spec["loader"]
    if kind == "hf":
        from datasets import load_dataset
        ds = load_dataset(name)
        sp = "test" if "test" in ds else list(ds)[0]
        lab = next(k for k in ds[sp].features if "label" in k.lower())
        raw = list(ds[sp].features[lab].names)
    else:
        d = _tv_dataset(name, DATA_DIR / task, "test", build_transform())
        base = d.dataset if isinstance(d, Subset) else d
        raw = list(base.classes)
    return [c.replace("_", " ").replace("-", " ").strip().lower() for c in raw]


# --- loaders ---

def task_loaders(task, batch_size=BATCH_SIZE, num_workers=0):
    """
    dict(train, val, test) of DataLoaders plus classnames for task. num_workers
    defaults to 0.
    """
    tf = build_transform()
    train, val, test = _raw_splits(task, tf)
    mk = lambda ds, sh: DataLoader(ds, batch_size=batch_size, shuffle=sh,
                                   num_workers=num_workers, pin_memory=True,
                                   drop_last=sh)
    return {
        "train": mk(train, True),
        "val": mk(val, False),
        "test": mk(test, False),
        "classnames": classnames(task),
    }


class CyclingLoader:
    """infinite (x, y) batch stream over a DataLoader, tensors moved to device."""

    def __init__(self, loader):
        self.loader = loader
        self._it = iter(loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            x, y = next(self._it)
        except StopIteration:
            self._it = iter(self.loader)
            x, y = next(self._it)
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
