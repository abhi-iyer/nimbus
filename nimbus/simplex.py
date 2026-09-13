"""bezier simplex interpolation over N vertex models."""

from itertools import combinations

import numpy as np
import torch
import torch.nn as nn

from .seeds import device


def enumerate_alphas(N, r):
    """
    all degree-r face indicators over N vertices.

    returns a list of N-tuples, each with r ones marking the vertices of one
    r-subset (face) of the simplex.
    """
    return [
        tuple(1 if i in sub else 0 for i in range(N))
        for sub in combinations(range(N), r)
    ]


def choose_face(N, r, rng, face=None):
    """
    pick an r-subset of the N vertices. returns `face` verbatim if given, else
    samples one without replacement.
    """
    if face is not None:
        return tuple(face)
    return tuple(sorted(rng.choice(N, size=r, replace=False).tolist()))


def sample_w_for_degree(N, r, rng=None, device=None, face=None):
    """
    sample a simplex weight supported on one degree-r face.

    draws dirichlet weights over the r face vertices (concentration all-ones or
    r-ones, each with probability 0.5) and embeds them into a N-vector.

    returns the weight (torch.Tensor) and the face (tuple).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    assert 1 <= r <= N
    face = choose_face(N, r, rng, face=face)
    conc = np.ones(r) if rng.random() < 0.5 else r * np.ones(r)
    u = rng.dirichlet(conc)
    w = np.zeros(N)
    w[list(face)] = u
    return torch.tensor(w, dtype=torch.float32, device=device), face


def build_bump_params(vertex_models, degrees, share_by_degree=True):
    """
    build zero-initialized bump parameters, one set per bump key.

    degrees is the set of face degrees r to include. with share_by_degree, all
    faces of the same degree share one bump set keyed by r; otherwise each face
    keys on its own indicator.

    returns (bump_params, alpha_to_key, alphas_by_degree).
    """
    N = len(vertex_models)
    template = [(n, p) for n, p in vertex_models[0].named_parameters() if p.requires_grad]

    alphas_by_degree = {r: enumerate_alphas(N, r) for r in sorted(degrees)}
    all_alphas = [a for r in sorted(degrees) for a in alphas_by_degree[r]]
    if share_by_degree:
        alpha_to_key = {a: sum(a) for a in all_alphas}
    else:
        alpha_to_key = {a: a for a in all_alphas}

    unique_keys = list(dict.fromkeys(alpha_to_key[a] for a in all_alphas))
    bump_params = {
        key: {n: nn.Parameter(torch.zeros_like(p)) for n, p in template}
        for key in unique_keys
    }
    return bump_params, alpha_to_key, alphas_by_degree


class SimplexInterpolator:
    """
    maps a simplex weight w to a full parameter dict.

    theta(w) = w @ vertex_flat + sum over faces of
    gate_r * r^r * \\prod_{j in face} w_j * bump_r,

    a linear barycentric blend of the N vertices plus bezier bumps whose
    coefficients vanish on face boundaries and peak at the face centroid. bumps
    are grouped by key, scaled by per-key log-gates, and added to the blend.
    tied parameters (shared data_ptr) and non-trainable buffers pass through
    unchanged.
    """

    def __init__(self, vertex_params, bump_params, alpha_to_key,
                 alphas_by_degree, dummy, degree_gates=None):
        self.bump_params = bump_params
        self._bump_cache = None

        self.param_names  = list(vertex_params[0].keys())
        self.param_shapes = [vertex_params[0][n].shape  for n in self.param_names]
        self.param_sizes  = [vertex_params[0][n].numel() for n in self.param_names]

        dev = next(iter(vertex_params[0].values())).device

        # stack each vertex's trainable params into one flat row
        self.vertex_flat = torch.stack([
            torch.cat([vp[n].reshape(-1) for n in self.param_names])
            for vp in vertex_params
        ])

        # non-trainable keys tied (shared storage) to a trainable param
        sd = dummy.state_dict()
        ptr_to_keys = {}
        for k, v in sd.items():
            if torch.is_tensor(v) and v.numel() > 0:
                ptr_to_keys.setdefault(v.data_ptr(), []).append(k)
        trainable_names = set(vertex_params[0].keys())
        self.tied_aliases = set()
        for keys in ptr_to_keys.values():
            if len(keys) > 1 and any(k in trainable_names for k in keys):
                for k in keys:
                    if k not in trainable_names:
                        self.tied_aliases.add(k)
        # buffers and untied params held fixed across all w
        self.fixed_sd = {
            k: v for k, v in sd.items()
            if k not in vertex_params[0] and k not in self.tied_aliases
        }

        all_alphas = [a for r in sorted(alphas_by_degree) for a in alphas_by_degree[r]]
        n_alphas = len(all_alphas)
        N = len(vertex_params)

        self.key_list = list(dict.fromkeys(alpha_to_key[a] for a in all_alphas))
        key_to_idx = {k: i for i, k in enumerate(self.key_list)}
        self.n_keys = len(self.key_list)

        self.degree_gates   = degree_gates or {}
        self._gate_key_list = self.key_list

        mask  = torch.zeros(n_alphas, N, device=dev)
        r_vec = torch.zeros(n_alphas, device=dev)
        self.key_indices = torch.zeros(n_alphas, dtype=torch.long, device=dev)

        for i, alpha in enumerate(all_alphas):
            r_vec[i] = sum(alpha)
            self.key_indices[i] = key_to_idx[alpha_to_key[alpha]]
            for j, a in enumerate(alpha):
                if a > 0:
                    mask[i, j] = 1.0

        self.alpha_mask   = mask
        # log(r^r) term of the bump coefficient
        self.log_r_powers = r_vec * torch.log(r_vec.clamp(min=1.0))

    def cache_bump_flat(self):
        """flatten and stack the current bump parameters, one row per key."""
        self._bump_cache = torch.stack([
            torch.cat([p.reshape(-1) for p in self.bump_params[key].values()])
            for key in self.key_list
        ])

    def __call__(self, w):
        """
        interpolate the parameters at simplex weight w (torch.Tensor).

        returns a param dict of trainable params plus fixed state.
        """
        if self._bump_cache is None:
            self.cache_bump_flat()

        # linear barycentric blend
        flat = w @ self.vertex_flat

        # bump coefficient r^r * \prod_{j in face} w_j in log space
        log_w      = torch.log(w.clamp(min=1e-30))
        log_coeffs = self.log_r_powers + self.alpha_mask @ log_w
        coeffs     = torch.exp(log_coeffs)

        # sum face coefficients into their bump keys
        key_coeffs = torch.zeros(self.n_keys, device=w.device, dtype=flat.dtype)
        key_coeffs.scatter_add_(0, self.key_indices, coeffs)

        # apply per-key log-gates: strength = exp(gate)
        _zero = torch.tensor(0.0, device=w.device)
        log_gates  = torch.stack([self.degree_gates.get(k, _zero) for k in self._gate_key_list])
        key_coeffs = key_coeffs * torch.exp(log_gates)
        flat       = flat + key_coeffs @ self._bump_cache

        # unflatten back into named params
        param_map = {}
        offset = 0
        for name, shape, sz in zip(self.param_names, self.param_shapes, self.param_sizes):
            param_map[name] = flat[offset:offset + sz].view(shape)
            offset += sz

        return {**param_map, **self.fixed_sd}


def clone_bp(bp):
    """deep-copy bump parameters."""
    return {k: {n: nn.Parameter(p.data.clone()) for n, p in pd.items()}
            for k, pd in bp.items()}


def clone_dg(dg):
    """deep-copy degree gates."""
    return {k: nn.Parameter(g.data.clone()) for k, g in dg.items()}


def bp_to_cpu(bp):
    """detach bump parameters to cpu tensors."""
    return {k: {n: p.data.detach().cpu().clone() for n, p in pd.items()}
            for k, pd in bp.items()}


def bp_from_cpu(d, dev=device):
    """rebuild bump parameters from cpu tensors on `dev`."""
    return {k: {n: nn.Parameter(t.to(dev)) for n, t in pd.items()}
            for k, pd in d.items()}


def dg_to_cpu(dg):
    """detach degree gates to cpu tensors."""
    return {k: g.data.detach().cpu().clone() for k, g in dg.items()}


def dg_from_cpu(d, dev=device):
    """rebuild degree gates from cpu tensors on `dev`."""
    return {k: nn.Parameter(t.to(dev)) for k, t in d.items()}


def pack_simplex_state(bp, dg, degrees):
    """bundle bumps, gates, and degree config for checkpointing."""
    return {
        "bp": bp_to_cpu(bp),
        "dg": dg_to_cpu(dg),
        "config": {"degrees": tuple(sorted(degrees))},
    }


def unpack_simplex_state(state, dev=device):
    """restore bumps and gates from a packed state."""
    return bp_from_cpu(state["bp"], dev), dg_from_cpu(state["dg"], dev)


def load_simplex_if_match(path, expected_cfg, dev=device):
    """load a packed simplex state if it exists and its config matches, else None."""
    if not path.exists():
        return None
    state = torch.load(path, map_location=dev)
    if state.get("config") != expected_cfg:
        return None
    return state
