# Minitron-SSM Reproduction (Scaled)

A scaled, public-source reproduction of the compression pipeline from
[**Minitron-SSM: Efficient Hybrid Language Model Compression through
Group-Aware SSM Pruning**](https://arxiv.org/abs/2504.11409) (NVIDIA,
NeurIPS 2025).

The canonical implementation plan lives in
[`minitron_ssm_reproduction_plan.md`](minitron_ssm_reproduction_plan.md).
A frozen, human-readable summary used for course submission is in
[`ReproducingPlan.txt`](ReproducingPlan.txt) and is **read-only**.

This repository contains a project skeleton: configs, package layout,
pipeline scripts, and tests. Cheap CPU-runnable utilities (config
loading, parameter counting, candidate enumeration, KD loss math,
shape-check helpers) are fully implemented and tested. Everything that
requires the 8B teacher or GPU runtime is stubbed with typed signatures
and `TODO(stage-N)` markers tying back to the plan.

## Repository layout

```text
configs/                YAML configs (base, data, importance, search_space, kd, eval)
src/minitron_ssm/
    models/             load + introspect Nemotron-H parent
    data/               streaming data + mixture + tokenization
    importance/         activation hooks + Mamba/FFN/embedding scoring
    pruning/            group-aware Mamba, FFN, embedding, depth pruning
    search/             candidate enumeration + parameter counting + budget filter
    kd/                 KD loss, online trainer, top-k logit cache
    eval/               LM loss, throughput, lm-eval-harness wrapper
    utils/              config loader, shape checks, logging, checkpoints
scripts/                01..10 pipeline CLIs
tests/                  CPU-only pytest suite
outputs/                run artifacts (gitignored)
```

## Pipeline

```text
01 baseline -> 02 importance -> 03 generate candidates -> 04 prune candidates
   -> 05 eval candidates -> 06 select top 3 -> 07 KD smoke -> 08 KD train
   -> 09 final eval -> 10 (optional) mini final KD
```

## Quickstart

Use the **conda** environment `minitron` (dependencies are installed there).

`pip install -e ".[dev,eval]"` must be run **inside** this repo (where `pyproject.toml` lives), not from `~/Desktop`.

```bash
conda activate minitron
cd /arf/scratch/skantar/MinitronSSM   # repo root — required for pip -e
pip install -e ".[dev,eval]"              # editable install of this repo only
pytest
python scripts/03_generate_candidates.py
python scripts/01_baseline.py --dry-run
```

On the A100 cluster (same `minitron` env), additionally install:

```bash
conda activate minitron
pip install "mamba-ssm==2.*" "causal-conv1d>=1.4" --no-build-isolation
```

## Status

- [x] Skeleton, configs, CPU utilities, tests
- [ ] Load parent and measure baseline (Stage 1)
- [ ] Importance estimation (Stage 2)
- [ ] Candidate pruning + zero-shot eval (Stages 3-6)
- [ ] Short KD on top-3 (Stages 7-8)
- [ ] Final internal evaluation (Stage 9)

## Hardware target

2-4 A100 (80GB), BF16, sequence length 8192 (fallback 4096). See
[`minitron_ssm_reproduction_plan.md`](minitron_ssm_reproduction_plan.md)
sections 5 and 14 for full settings.

## What we are NOT reproducing

- The 380B-token final KD stage from the paper.
- The exact Phase-3 NVIDIA training mixture.
- Long-context, instruct, or safety evaluations.

See section 16 of the plan for the full list.
