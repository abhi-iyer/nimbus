"""config for the ViT (CLIP) merging experiment.

backbone: CLIP ViT-B/32 (openai/clip-vit-base-patch32, HuggingFace transformers).
vertices are trained by experiments/vit/vertices_lora.py: base CLIP plus a
rank-r LoRA adapter per task, materialized into a full vision tower
(vision_model.*).

  * interpolated = the vision tower (vision_model.*, ~87.5M params)
  * fixed/shared = the base visual_projection, logit_scale, and the per-task
    zero-shot heads built from CLIP text embeddings of class names

the Nimbus bump/gate hyperparameters follow the LLM leg, see
experiments.llm.config.
"""
from math import comb
from pathlib import Path

import numpy as np


GLOBAL_SEED = 42

# --- backbone + vertex source ---
CLIP_BASE = "openai/clip-vit-base-patch32"

# full 8-task benchmark. loader is how data.py fetches it; template is the CLIP
# zero-shot prompt; classnames=None derives names from the dataset label order.
TASK_SPECS = {
    "mnist":    dict(loader=("tv", "MNIST"),
                     template='a photo of the number: "{}".',
                     classnames=[str(i) for i in range(10)]),
    "svhn":     dict(loader=("tv", "SVHN"),
                     template='a photo of the number: "{}".',
                     classnames=[str(i) for i in range(10)]),
    "gtsrb":    dict(loader=("tv", "GTSRB"),
                     template='a zoomed in photo of a "{}" traffic sign.',
                     # torchvision GTSRB exposes no .classes; descriptions in
                     # label order (0..42).
                     classnames=[
                         "red and white circle 20 kph speed limit",
                         "red and white circle 30 kph speed limit",
                         "red and white circle 50 kph speed limit",
                         "red and white circle 60 kph speed limit",
                         "red and white circle 70 kph speed limit",
                         "red and white circle 80 kph speed limit",
                         "end / de-restriction of 80 kph speed limit",
                         "red and white circle 100 kph speed limit",
                         "red and white circle 120 kph speed limit",
                         "red and white circle red car and black car no passing",
                         "red and white circle red truck and black car no passing",
                         "red and white triangle road intersection warning",
                         "white and yellow diamond priority road",
                         "red and white upside down triangle yield right-of-way",
                         "stop",
                         "empty red and white circle",
                         "red and white circle no truck entry",
                         "red circle with white horizontal stripe no entry",
                         "red and white triangle with exclamation mark warning",
                         "red and white triangle with black left curve approaching warning",
                         "red and white triangle with black right curve approaching warning",
                         "red and white triangle with black double curve approaching warning",
                         "red and white triangle rough / bumpy road warning",
                         "red and white triangle car skidding / slipping warning",
                         "red and white triangle with narrow bottom road narrows warning",
                         "red and white triangle with person digging / construction / road work warning",
                         "red and white triangle with traffic light approaching warning",
                         "red and white triangle with person walking warning",
                         "red and white triangle with child and person walking warning",
                         "red and white triangle with bicyle warning",
                         "red and white triangle with snowflake / ice warning",
                         "red and white triangle with deer warning",
                         "white circle with gray strike bar no speed limit",
                         "blue circle with white right turn arrow mandatory",
                         "blue circle with white left turn arrow mandatory",
                         "blue circle with white forward arrow mandatory",
                         "blue circle with white forward or right turn arrow mandatory",
                         "blue circle with white forward or left turn arrow mandatory",
                         "blue circle with white keep right arrow mandatory",
                         "blue circle with white keep left arrow mandatory",
                         "blue circle with white arrows indicating a traffic circle",
                         "white circle with gray strike bar indicating no passing for cars has ended",
                         "white circle with gray strike bar indicating no passing for trucks has ended",
                     ]),
    "dtd":      dict(loader=("tv", "DTD"),
                     template='a photo of a {} texture.',
                     classnames=None),
    "eurosat":  dict(loader=("tv", "EuroSAT"),
                     template='a centered satellite photo of {}.',
                     classnames=["annual crop land", "forest",
                                 "brushland or shrubland", "highway or road",
                                 "industrial buildings", "pasture land",
                                 "permanent crop land", "residential buildings",
                                 "river", "sea or lake"]),
    # cars/sun397/resisc45 come from HuggingFace rather than torchvision.
    "cars":     dict(loader=("hf", "tanganke/stanford_cars"),
                     template='a photo of a {}.',
                     classnames=None),
    "resisc45": dict(loader=("hf", "timm/resisc45"),
                     template='satellite imagery of {}.',
                     classnames=None),
    "sun397":   dict(loader=("hf", "tanganke/sun397"),
                     template='a photo of a {}.',
                     classnames=None),
}

# the N=8 benchmark.
ALL_TASKS = ["mnist", "svhn", "gtsrb", "dtd", "eurosat",
             "cars", "resisc45", "sun397"]

TASK_NAMES = ALL_TASKS                           # N=8 full benchmark
TASK_DISPLAY = {"mnist": "MNIST", "svhn": "SVHN", "gtsrb": "GTSRB", "dtd": "DTD",
                "eurosat": "EuroSAT", "cars": "Cars", "resisc45": "RESISC45",
                "sun397": "SUN397"}
N = len(TASK_NAMES)


# --- degree set + rank: the single shipped config ---
# D = {2, N-1, N}, the near-dense degree family (N=8 gives |D|=3), at LoRA rank r=64.
DEGREES = (2, 7, 8)
RANK = 64


# --- LoRA bump targeting (HuggingFace CLIP vision-tower matrix names) ---
# LoRA the attention + MLP matrices; keep everything else dense (embeddings,
# class/position emb, LayerNorms, all biases).
LORA_TARGET_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.out_proj.weight",
    "mlp.fc1.weight",
    "mlp.fc2.weight",
)
LORA_INIT_STD = 0.02


# --- image / data ---
IMAGE_SIZE = 224
# CLIP normalization.
IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)
BATCH_SIZE = 64   # per-task forward; faces run per-task, not concatenated
VAL_FRAC = 0.1            # fraction of train carved off for val
MAX_TRAIN_PER_TASK = 50_000   # cap on train examples per task


# --- Nimbus bump/gate training (anchored to experiments/llm/config.py) ---
SIMPLEX_ITERS_PER_DEGREE = 1000
SIMPLEX_LR = 5e-5
CAL_ITERS = 300
CAL_LR = 2e-3
N_CAL_SAMPLES = 4
CAL_BATCHES = 8                   # fixed calibration set
DISTILL_TEMPERATURE = 1.0
N_ENS = 100
N_FACE_ENS = 100

# --- vertices ---
# base CLIP + rank-r LoRA, one per task (see experiments/vit/vertices_lora.py).
# VERTEX_SUFFIX tags the Nimbus simplex-checkpoint filenames.
VERTEX_SUFFIX = "_loraV"


# --- baseline ensemble (compute/param-matched) ---
# M = |D| independent members, each parameterized like one Nimbus bump-key
# (rank-r LoRA on LORA_TARGET_SUFFIXES + dense deltas on the rest) and trained as
# a plain multitask fine-tune of base CLIP with per-task CE. parameter-matched to
# Nimbus's learnable params; total steps == Nimbus phase-1 split across M members.
# trained with Nimbus's bump optimizer: constant SIMPLEX_LR, AdamW(0.9, 0.95), no
# warmup or decay.


def nimbus_phase1_steps(D, k=None, S=None):
    """
    total student forward+backward steps in Nimbus phase-1 for degree set D:
    S iters per degree times faces/step times members/face, with
    faces/step = min(C(k,r), k) and members = r.
    """
    k = N if k is None else k
    S = SIMPLEX_ITERS_PER_DEGREE if S is None else S
    return S * sum(min(comb(k, r), k) * r for r in sorted(set(D)))


def baseline_ens_steps(D, k=None, S=None):
    """per-member step budget: Nimbus phase-1 steps split across |D| members."""
    return nimbus_phase1_steps(D, k, S) // len(set(D))


# --- robustness sweeps (vision: gaussian input noise + weight noise) ---
N_TRIALS = 3
INPUT_EPSILONS = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                           0.6, 0.7, 0.8, 0.9, 1.0])
WEIGHT_EPSILONS = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                            0.6, 0.7, 0.8, 0.9, 1.0])


# --- paths (everything stays inside experiments/vit/) ---
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data_dir"
CKPT_DIR = DATA_DIR / "checkpoints"      # vertices + baseline ensemble (cached)
HEAD_DIR = DATA_DIR / "heads"            # cached zero-shot heads
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = ROOT_DIR / "figures"
SIMPLEX_CKPT_DIR = CKPT_DIR / "simplex"  # Nimbus bump/gate state


def dtag(D):
    """filesystem-safe tag for a degree set, e.g. (2,7,8) -> '278'."""
    return "".join(str(x) for x in sorted(D))
