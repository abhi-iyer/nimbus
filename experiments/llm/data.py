import gzip
import json
import lzma
import time
import urllib.request

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import (
    DATA_DIR, RAW_DIR, TEXTS_DIR, TOKENS_DIR, TOKENIZER_PATH,
    TASK_NAMES, VOCAB_SIZE, MAX_CHARS_PER_TASK,
    MAX_TOKENIZER_CHARS, MAX_TOKENS_PER_TASK, VAL_FRAC,
)
from ._dataset_urls import SOURCES
from .tokenizer import BPETokenizer


def fetch_url(url, timeout=120, max_bytes=None):
    req = urllib.request.Request(url, headers={"User-Agent": "nimbus-llm/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if max_bytes is None:
            return r.read().decode("utf-8", errors="replace")
        buf = bytearray()
        chunk = 1 << 20
        while len(buf) < max_bytes:
            piece = r.read(min(chunk, max_bytes - len(buf)))
            if not piece:
                break
            buf.extend(piece)
        return bytes(buf).decode("utf-8", errors="replace")


def _hf_url(repo, path):
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def _download_file(url, dest, timeout=300):
    """stream a source file to `dest`, cached. downloads to a .part file first."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "nimbus-llm/1.0"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            piece = r.read(1 << 22)
            if not piece:
                break
            f.write(piece)
            got += len(piece)
            if total and got % (1 << 26) < (1 << 22):
                print(f"    {dest.name}: {got/1e6:.0f}/{total/1e6:.0f} MB", flush=True)
    part.rename(dest)
    print(f"    {dest.name}: {dest.stat().st_size/1e6:.0f} MB in {time.time()-t0:.0f}s")
    return dest


def _read_parquet_text(path, field, max_chars):
    """concatenate `field` over parquet row batches, stopping at `max_chars`."""
    import pyarrow.parquet as pq

    pieces, n = [], 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=[field]):
        for v in batch.column(field).to_pylist():
            if not v:
                continue
            pieces.append(v)
            n += len(v)
            if n >= max_chars:
                return "\n".join(pieces)
    return "\n".join(pieces)


def _open_text(path):
    if path.suffix == ".xz":
        return lzma.open(path, "rt", encoding="utf-8", errors="replace")
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _read_jsonl_text(path, field, max_chars):
    """concatenate `field` over JSON lines, stopping at `max_chars`."""
    pieces, n = [], 0
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            v = json.loads(line).get(field)
            if not v:
                continue
            pieces.append(v)
            n += len(v)
            if n >= max_chars:
                break
    return "\n\n".join(pieces)


def read_source(spec, max_chars):
    """read up to `max_chars` characters from one source spec (see _dataset_urls)."""
    kind = spec[0]
    if kind == "url":
        return fetch_url(spec[1], max_bytes=max_chars * 4)   # utf-8 worst case
    repo, path, field = spec[1], spec[2], spec[3]
    local = _download_file(_hf_url(repo, path), RAW_DIR / repo.replace("/", "__") / path.split("/")[-1])
    if kind == "hf_parquet":
        return _read_parquet_text(local, field, max_chars)
    if kind == "hf_jsonl":
        return _read_jsonl_text(local, field, max_chars)
    raise ValueError(f"unknown source kind: {kind}")


def collect_task_text(name, specs, budget=MAX_CHARS_PER_TASK):
    """
    gather up to `budget` chars for one task, split fairly across its sources by
    water-filling (each source capped at an equal share of the remaining budget).
    """
    pieces = []
    for spec in specs:
        try:
            pieces.append(read_source(spec, budget))
        except Exception as e:
            print(f"  WARN: failed {spec}: {e}")
    if not pieces:
        raise RuntimeError(f"No data fetched for {name}")

    # water-fill: shortest sources first, each capped at an equal share of the
    # remaining budget.
    order = sorted(range(len(pieces)), key=lambda i: len(pieces[i]))
    taken, remaining = {}, budget
    for j, i in enumerate(order):
        share = remaining // (len(order) - j)
        taken[i] = pieces[i][:share]
        remaining -= len(taken[i])
    return _interleave([taken[i] for i in range(len(pieces))])


# rounds of round-robin interleaving across sources.
INTERLEAVE_ROUNDS = 40


def _interleave(pieces, rounds=INTERLEAVE_ROUNDS):
    """round-robin the sources in `rounds` equal slices each, so any contiguous
    fraction f of the output contains ~f of every source."""
    pieces = [p for p in pieces if p]
    if len(pieces) <= 1:
        return "\n\n".join(pieces)
    cuts = [[p[k * len(p) // rounds:(k + 1) * len(p) // rounds] for k in range(rounds)]
            for p in pieces]
    out = []
    for k in range(rounds):
        for src in cuts:
            if src[k]:
                out.append(src[k])
    return "\n\n".join(out)


def download_texts():
    TEXTS_DIR.mkdir(parents=True, exist_ok=True)
    texts = {}
    for name in TASK_NAMES:
        path = TEXTS_DIR / f"{name}.txt"
        if not (path.exists() and path.stat().st_size > 0):
            specs = SOURCES[name]
            print(f"building {name} ({len(specs)} source(s))", flush=True)
            path.write_text(collect_task_text(name, specs), encoding="utf-8")
        t = (TEXTS_DIR / f"{name}.txt").read_text(encoding="utf-8")
        if len(t) > MAX_CHARS_PER_TASK:
            t = t[:MAX_CHARS_PER_TASK]
        texts[name] = t
        print(f"  {name}: {len(t):,} chars")
    return texts


def get_or_train_tokenizer(texts):
    tok = BPETokenizer()
    if TOKENIZER_PATH.exists():
        tok.load(str(TOKENIZER_PATH))
        print(f"loaded tokenizer with vocab_size={len(tok.vocab)}")
        return tok
    per_task = MAX_TOKENIZER_CHARS // len(texts)
    combined = "\n\n".join(t[:per_task] for t in texts.values())
    print(f"training BPE tokenizer on {len(combined):,} chars (vocab={VOCAB_SIZE})")
    tok.train(combined, vocab_size=VOCAB_SIZE)
    tok.save(str(TOKENIZER_PATH))
    return tok


def encode_all(texts, tokenizer):
    TOKENS_DIR.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name, text in texts.items():
        path = TOKENS_DIR / f"{name}.bin"
        if path.exists():
            ids = np.fromfile(path, dtype=np.uint16)
        else:
            t0 = time.time()
            print(f"encoding {name} ({len(text):,} chars)")
            ids = np.array(tokenizer.encode(text), dtype=np.uint16)
            if len(ids) > MAX_TOKENS_PER_TASK:
                ids = ids[:MAX_TOKENS_PER_TASK]
            ids.tofile(path)
            print(f"  -> {len(ids):,} tokens in {time.time() - t0:.1f}s")
        if len(ids) > MAX_TOKENS_PER_TASK:
            ids = ids[:MAX_TOKENS_PER_TASK]
        arrays[name] = ids
    return arrays


def prepare_data():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    texts = download_texts()
    tokenizer = get_or_train_tokenizer(texts)
    arrays = encode_all(texts, tokenizer)
    return tokenizer, arrays


class TokenChunks(Dataset):
    def __init__(self, tokens, block_size):
        self.tokens = torch.from_numpy(tokens.astype(np.int64))
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.tokens) - self.block_size - 1)

    def __getitem__(self, i):
        c = self.tokens[i:i + self.block_size + 1]
        return c[:-1], c[1:]


def split_train_val(arr, val_frac=VAL_FRAC):
    cut = int(len(arr) * (1 - val_frac))
    return arr[:cut], arr[cut:]


def make_loader(token_array, block_size, batch_size, shuffle=True, seed=42):
    ds = TokenChunks(token_array, block_size)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, drop_last=True,
        generator=g if shuffle else None, num_workers=0,
    )


class CyclingLoader:
    def __init__(self, loader):
        self.loader = loader
        self.it = iter(loader)

    def __next__(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.loader)
            return next(self.it)


def mixed_val_with_tasks(val_arrays, block_size):
    per_task = []
    for name in TASK_NAMES:
        a = val_arrays[name]
        n = (len(a) - 1) // block_size
        if n <= 0:
            per_task.append(np.empty((0, block_size + 1), dtype=np.int64))
            continue
        idx = np.arange(n)[:, None] * block_size + np.arange(block_size + 1)[None, :]
        per_task.append(a[idx].astype(np.int64))
    N_ = len(per_task)
    n_min = min(c.shape[0] for c in per_task) if per_task else 0
    chunks = np.empty((n_min * N_, block_size + 1), dtype=np.int64)
    task_chunks = np.empty((n_min * N_, block_size), dtype=np.int64)
    for ti, c in enumerate(per_task):
        chunks[ti::N_] = c[:n_min]
        task_chunks[ti::N_] = ti
    return chunks, task_chunks


def val_batches(tokens, block_size, batch_size, max_batches=None, task_ids=None):
    from nimbus.seeds import device
    if np.ndim(tokens) == 2:
        N = tokens.shape[0]
        if max_batches is not None:
            N = min(N, max_batches * batch_size)
        if N == 0:
            return
        arr = torch.from_numpy(np.asarray(tokens[:N], dtype=np.int64))
        tarr = (torch.from_numpy(np.asarray(task_ids[:N], dtype=np.int64))
                if task_ids is not None else None)
        for i in range(0, N, batch_size):
            b = arr[i:i + batch_size]
            if len(b) < batch_size:
                break
            t = tarr[i:i + batch_size].to(device) if tarr is not None else None
            yield b[:, :-1].to(device), b[:, 1:].to(device), t
        return

    n = len(tokens) - block_size - 1
    if n <= 0:
        return
    arr = torch.from_numpy(tokens.astype(np.int64))
    starts = torch.arange(0, n, block_size)
    bi = 0
    for s in range(0, len(starts), batch_size):
        if max_batches is not None and bi >= max_batches:
            return
        batch_starts = starts[s:s + batch_size]
        x = torch.stack([arr[i:i + block_size] for i in batch_starts]).to(device)
        y = torch.stack([arr[i + 1:i + block_size + 1] for i in batch_starts]).to(device)
        yield x, y, None
        bi += 1
