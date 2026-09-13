"""
config for the LLM (NanoGPT) leg: N=7 from-scratch text specialists.

budgets follow the sample-passes convention:

    passes = (passes per step) * (iters per degree) * (batch size)
             * sum_{r in D} min(C(N, r), N) * r

degree_work is the sum term. every baseline gets the same sample-passes as
Nimbus phase 1 and converts them to optimizer steps via its passes-per-step.

only the vertices get an LR schedule (cosine + warmup); Nimbus bumps/gates and
every baseline train at a constant LR == SIMPLEX_LR.
"""
from math import comb
from pathlib import Path

import numpy as np


GLOBAL_SEED = 42

# --- tasks -----------------------------------------------------------------
# seven disjoint text domains (see _dataset_urls.py).
N = 7
TASK_NAMES = ["early_modern_english", "wikitext", "tinystories", "code",
              "arxiv", "legal", "multilingual"]
TASK_DISPLAY = ["Early Modern English", "WikiText", "TinyStories", "Code",
                "arXiv", "Legal", "French/German"]

ROOT_DIR     = Path(__file__).resolve().parent
DATA_DIR     = ROOT_DIR / "data_dir"
RAW_DIR      = DATA_DIR / "raw"          # downloaded parquet / jsonl.xz sources
TEXTS_DIR    = DATA_DIR / "texts"
TOKENS_DIR   = DATA_DIR / "tokens"
CKPT_DIR     = DATA_DIR / "checkpoints"
TOKENIZER_PATH = DATA_DIR / "tokenizer.json"
RESULTS_DIR  = ROOT_DIR / "results"
FIGURES_DIR  = ROOT_DIR / "figures"

# --- data ------------------------------------------------------------------
# 2048 BPE merges.
VOCAB_SIZE          = 2048
# 60M chars -> ~20M tokens/task (~18.5 tokens per parameter).
MAX_CHARS_PER_TASK  = 60_000_000
MAX_TOKENIZER_CHARS = 2_000_000
MAX_TOKENS_PER_TASK = 20_000_000
VAL_FRAC            = 0.05

BLOCK_SIZE = 256
BATCH_SIZE = 64

# --- vertices --------------------------------------------------------------
# vertices train outside the matched budget, to convergence via early stopping.
VERTEX_ITERS   = 100_000        # cap; early stopping usually binds first
VERTEX_LR      = 6e-4           # constant after warmup
VERTEX_WARMUP  = 200
VERTEX_EVAL_EVERY = 2000        # val-NLL check cadence
VERTEX_PATIENCE   = 5           # stop after this many checks with no improvement
VERTEX_VAL_BATCHES = 40

# --- Nimbus: phase 1 (bumps) + phase 2 (degree gates) ----------------------
SIMPLEX_ITERS_PER_DEGREE = 6000
SIMPLEX_LR               = 8e-3     # constant; also every baseline's LR

CAL_ITERS     = 1000
CAL_LR        = 2e-3                 # gate calibration lr
N_CAL_SAMPLES = 4
CAL_CHUNKS    = 4

N_ENS      = 100
N_FACE_ENS = 100

DISTILL_TEMPERATURE = 1.0

DEGREES = (2, 6, 7)                 # the {2, N-1, N} family


# --- sample-pass budgets ---------------------------------------------------

def degree_work(D=DEGREES, k=N):
    """sum_r min(C(k, r), k) * r, the task-batches forwarded per phase-1 iteration."""
    return sum(min(comb(k, r), k) * r for r in sorted(set(D)))


def sample_passes(D=DEGREES, k=N, iters=SIMPLEX_ITERS_PER_DEGREE,
                  batch=BATCH_SIZE, passes_per_step=1):
    """the budget unit: (passes/step) * (iters/degree) * batch * degree_work."""
    return passes_per_step * iters * batch * degree_work(D, k)


def steps_for_passes(passes, passes_per_step=1, batch=BATCH_SIZE):
    """optimizer steps that spend `passes` sample-passes at `passes_per_step`."""
    return int(passes) // (passes_per_step * batch)


NIMBUS_PHASE1_PASSES = sample_passes()          # printed by run.py at startup


def baseline_ens_steps(D=DEGREES, k=N):
    """per-member steps for the |D| baseline ensemble: phase-1 passes split |D| ways."""
    return steps_for_passes(sample_passes(D, k) // len(set(D)))


N_BASELINE = len(set(DEGREES))         # |D| compute+param-matched ensemble members
# |D|+N ensemble: N extra members at the same per-member budget as the |D| ones.
N_BASELINE_LARGE = N_BASELINE + N         # |D|+N ensemble size

# merge baselines; task vectors are defined against the mean of the vertices.
TIES_KEEP_FRACTION = 0.8            # TIES trim keep-fraction
FISHER_BATCHES = 50                 # batches per vertex for the diagonal Fisher


# --- robustness sweeps -----------------------------------------------------
N_TRIALS         = 3
MAX_BATCHES      = 50
INPUT_RATES      = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
WEIGHT_EPSILONS  = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
