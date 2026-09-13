# Nimbus

*Nimbus* stands for **N**-expert **I**nterpolating **M**anifold: **B**ézier surface over an **U**nderlying **S**implex  

Each experiment trains N single-task "vertex" models, then fits a low-dimensional manifold of models over them: a linear 
blend of the vertices plus learned Bézier "bump" corrections, indexed by a simplex weight w. An infinite set of models 
can be sampled from the manifold. When these models are then ensembled, they demonstrate superior robustness under 
input and weight perturbations (vs. compute- and parameter-matched baselines). 

tl;dr: *Nimbus* finds _functionally_ and _weight_-flat solutions through infinite ensembling 

## install

```bash
conda env create -f environment.yml
conda activate nimbus
```

this installs the `nimbus` package editable (`pip install -e .`). training each experiment/mode expects a cuda gpu; 
the figure and table scripts run on cpu.

## layout

```
nimbus/                     core library (modality-agnostic)
  simplex.py                Bézier simplex interpolator and bump construction
  ensemble.py               sampling weights and building model pools from the manifold
  metrics.py                accuracy, top-k, nll, ece, scoped by task
  robustness.py             ensemble robustness sweeps under perturbation
  tables.py                 ascii summary and robustness table writers
  seeds.py                  seeding and device selection

experiments/<mode>/         modalities: cifar, vit, llm
  config.py                 hyperparameters, task split, degree set, budgets
  data.py                   dataset loading and per-task batching
  model.py                  the vertex architecture
  baselines.py              merge/ensemble baselines and robustness sweeps
  train.py                  vertex training and the two Nimbus phases (bumps, gate calibration)
  run.py                    end-to-end orchestration; writes results/ and checkpoints
  figures/                  figure and table generation scripts
  results/                  generated tables (*.txt)

misc/                       mostly ablations; some extra analyses
  misc_figs.py              manifold-plateau figure (CIFAR-100 K=3)
  sweep.py                  degree-set sweeps
  ablations/                simplex-design and seed-sensitivity ablations
```

per-mode specifics:
- cifar: from-scratch ConvNeXt vertices; input noise in `perturbations.py`.
- vit: CLIP ViT-B/32 with LoRA vertices; adds `eval.py` (evaluation and
  perturbations), `tables.py`, and the LoRA pieces `lora_bump.py`,
  `lora_simplex.py`, `vertices_lora.py`.
- llm: from-scratch nanoGPT vertices; adds `tokenizer.py` and `_dataset_urls.py`;
  token corruptions in `perturbations.py`.

## train from scratch

```bash
python -m experiments.cifar.run     # all N in {3,5,7,10}; --N to pick one
python -m experiments.vit.run       # --batch to fit smaller gpus
python -m experiments.llm.run       # --data-only to just fetch + tokenize the corpora
```

each run caches checkpoints and writes the result tables under the mode's
`results/`. re-running reuses whatever cache exists; use `--refresh` to ignore
the cache and recompute.

## generate the figures and tables

run these after the corresponding mode has been trained (they read `results/` and
the cached checkpoints). all scripts run as modules from the repo root.

cifar:

```bash
python -m experiments.cifar.figures.radar_plots                   # all N; --N 3 for one
python -m experiments.cifar.figures.robustness_line_plots         # all N, both noises; --N, --noise
python -m experiments.cifar.figures.loss_surface                  # N=3; --refresh to re-evaluate
python -m experiments.cifar.figures.metrics_tables                # markdown tables; --N
```

vit:

```bash
python -m experiments.vit.figures.radar_plots
python -m experiments.vit.figures.robustness_line_plots
python -m experiments.vit.figures.perturbation_viz                # --eps to set the noise levels
python -m experiments.vit.figures.metrics_tables
```

llm:

```bash
python -m experiments.llm.figures.radar_plots
python -m experiments.llm.figures.robustness_line_plots           # both noises; --noise for one
python -m experiments.llm.figures.input_perturbation_viz
python -m experiments.llm.figures.metrics_tables
```


## misc analyses

```bash
# manifold-plateau figure (computes the ray cache if missing, then draws)
python -m misc.misc_figs                          # --refresh to recompute the cache

# degree-set sweep: one command runs every stage on cache, then draws + dumps tables
python -m misc.sweep --exp cifar7                 # --all for every experiment; --refresh to recompute

# ablations
python -m misc.ablations.simplex_design           # --variant, --n_w
python -m misc.ablations.seed_sensitivity         # --seeds / --n_seeds
```
