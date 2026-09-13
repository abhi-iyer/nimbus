"""byte-level BPE tokenizer."""

import heapq
import json

from tqdm import tqdm


class BPETokenizer:
    """byte-level BPE with a heap-based encoder."""

    def __init__(self):
        self.merges = {}
        self.vocab = {}

    def train(self, text, vocab_size):
        assert vocab_size >= 256
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        ids = list(text.encode("utf-8"))
        n_merges = vocab_size - 256
        for k in tqdm(range(n_merges), desc="bpe-train"):
            counts = {}
            for j in range(len(ids) - 1):
                p = (ids[j], ids[j + 1])
                counts[p] = counts.get(p, 0) + 1
            if not counts:
                break
            best = max(counts, key=counts.get)
            new_id = 256 + k
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            new_ids = []
            j = 0
            L = len(ids)
            while j < L:
                if j < L - 1 and ids[j] == best[0] and ids[j + 1] == best[1]:
                    new_ids.append(new_id)
                    j += 2
                else:
                    new_ids.append(ids[j])
                    j += 1
            ids = new_ids

    def encode(self, text, chunk_chars=256_000):
        out = []
        for s in range(0, len(text), chunk_chars):
            out.extend(self._encode_bytes(list(text[s:s + chunk_chars].encode("utf-8"))))
        return out

    def _encode_bytes(self, byte_list):
        n = len(byte_list)
        if n < 2:
            return list(byte_list)
        prev = list(range(-1, n - 1))
        nxt  = list(range(1, n + 1))
        nxt[-1] = -1
        tokens = list(byte_list)
        alive = [True] * n
        heap = []
        for i in range(n - 1):
            mid = self.merges.get((tokens[i], tokens[i + 1]))
            if mid is not None:
                heapq.heappush(heap, (mid, i))
        while heap:
            mid, i = heapq.heappop(heap)
            if not alive[i]:
                continue
            j = nxt[i]
            if j == -1:
                continue
            if self.merges.get((tokens[i], tokens[j])) != mid:
                continue
            tokens[i] = mid
            alive[j] = False
            nn_idx = nxt[j]
            nxt[i] = nn_idx
            if nn_idx != -1:
                prev[nn_idx] = i
                m2 = self.merges.get((tokens[i], tokens[nn_idx]))
                if m2 is not None:
                    heapq.heappush(heap, (m2, i))
            pp = prev[i]
            if pp != -1:
                m2 = self.merges.get((tokens[pp], tokens[i]))
                if m2 is not None:
                    heapq.heappush(heap, (m2, pp))
        out = []
        i = 0
        while i != -1:
            out.append(tokens[i])
            i = nxt[i]
        return out

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        data = {"merges": [[a, b, v] for (a, b), v in self.merges.items()]}
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        for a, b, v in data["merges"]:
            self.merges[(a, b)] = v
            self.vocab[v] = self.vocab[a] + self.vocab[b]
