from math import comb
from pathlib import Path


GLOBAL_SEED         = 42
BATCH_SIZE          = 256

EPOCHS              = 100
LR                  = 8e-3
CAL_EPOCHS          = 50
CAL_LR              = 1e-3
N_CAL_SAMPLES       = 5
CAL_CHUNKS          = 4
N_ENS               = 100
N_FACE_ENS          = 100
DISTILL_TEMPERATURE = 1.0

# (N, degrees, N_BASELINE) for the {2, N-1, N} degree family. N_BASELINE = |D|,
# the number of baseline-ensemble members (one full ConvNeXt per Nimbus bump-key).
EXPERIMENT_CONFIGS = [
    (3,  [2, 3],      2),
    (5,  [2, 4, 5],   3),
    (7,  [2, 6, 7],   3),
    (10, [2, 9, 10],  3),
]


def baseline_ens_epochs(N, degrees, base_epochs=EPOCHS):
    """
    epochs per baseline-ensemble member so the |D|-member ensemble's total
    forward/backward passes match Nimbus phase-1.

    E' = base_epochs * \\sum_r min(C(N,r),N)*r / (|D|*N).
    """
    degs = sorted(set(degrees))
    M = len(degs)
    work = sum(min(comb(N, r), N) * r for r in degs)
    return round(base_epochs * work / (M * N))


def _phase1_work(N, degrees):
    """phase-1 task-units per epoch: \\sum_r min(C(N,r),N)*r."""
    return sum(min(comb(N, r), N) * r for r in sorted(set(degrees)))


# the "single4x" baseline: one wide model given 4x Nimbus phase-1 compute
# (2x trainable params times 2x sample-passes). both factors below.

def baseline_single4x_epochs(N, degrees, base_epochs=EPOCHS):
    """
    epochs for single4x so its sample-passes are 2x Nimbus phase-1.

    E = 2 * base_epochs * work / N.
    """
    return round(2 * base_epochs * _phase1_work(N, degrees) / N)


def baseline_single4x_param_mult(degrees):
    """param multiple for single4x = 2*|D|, twice Nimbus's dense-bump params."""
    return 2 * len(set(degrees))


ROOT_DIR    = Path(__file__).resolve().parent
DATA_DIR    = ROOT_DIR / "data_dir"
RESULTS_DIR = ROOT_DIR / "results"


STAGE_OFFSET = {
    "vertex":            0,
    "baseline":        100,
    "single4x":        200,
    "fisher":          300,
    "ties":            400,
    "pcgrad":          500,
    "asam":            600,
    "swa":             700,
    "simplex_pre":     800,
    "simplex_post":    900,
    "baseline_extra": 1000,
}


def stage_seed(N, name, idx=0):
    """per-stage seed from GLOBAL_SEED, N, the stage name, and an index."""
    return GLOBAL_SEED + N * 10000 + STAGE_OFFSET[name] + idx
