# Runtime Estimates per Pipeline Script

All measurements assume **one A100 (80 GB), bf16, palamut-cuda partition**.
Observed numbers come from SLURM job **1280919** (run May 31 – Jun 1 2026).
KD estimates are calculated from calibration data (see Notes).

---

## Summary table

| # | Script | Token / sample budget | Est. wall time | Status | Slurm script |
|---|--------|-----------------------|----------------|--------|--------------|
| 01 | `01_baseline.py` | 1M val tokens + throughput | ~50 min | ✅ done (job 1280917) | `run_baseline.slurm` |
| 02 | `02_importance.py` | 1024 calibration batches × 8192 | ~30–60 min | ✅ done (prior job) | — |
| 03 | `03_generate_candidates.py` | CPU-only enumeration | < 1 min | ✅ done (job 1280919) | `run_overnight.slurm` |
| 04 | `04_prune_candidates.py` | CPU-side slicing × 20 candidates | ~6 min | ✅ done (job 1280919) | `run_overnight.slurm` |
| 05 | `05_eval_candidates.py` | 1M val tokens × 21 candidates | ~11 h | ✅ done (job 1280919) | `run_overnight.slurm` |
| 06 | `06_select_top3.py` | read/sort JSON | < 1 min | ✅ done (job 1280919) | `run_overnight.slurm` |
| 07 | `07_kd_smoke.py` | **2M tokens**, 1 candidate | ~1–2 h | ⏳ pending | `run_kd_smoke.slurm` |
| 08 | `08_kd_train.py` | **20M tokens × 3 candidates** | ~18–22 h | ⏳ pending | `run_kd_train.slurm` |
| 09 | `09_final_eval.py` | lm-eval-harness 7 tasks × 3 models | ~4–8 h | ⏳ pending | (separate job needed) |
| 10 | `10_optional_mini_kd.py` | 1B tokens (default), 1 candidate | ~50 h | ⏳ optional | (separate job, skip if no time) |

---

## Per-script breakdown

### 01 — baseline (observed)
- Load teacher: ~10s  
- LM loss on 1M tokens (seq 8192, batch 1): ~2 min  
- Throughput benchmark (2 configs × 7 iters): ~45 min  
- **Total observed: ~50 min**

### 02 — importance scoring (estimated)
- Load teacher: ~10s  
- 1024 calibration forward passes × 8192 tokens at ~4 tok/s (8B model): 1024 × 8192 tokens / ~5000 tok/s ≈ 1680s
- Score computation: ~few seconds  
- **Estimate: ~30–60 min**

### 03 — generate candidates
- Pure Python arithmetic, no GPU.  
- **Observed: < 1 min**

### 04 — prune candidates (observed)
- CPU deepcopy + weight slicing per candidate. No forward pass.
- **Observed: ~6 min for 20 candidates**

### 05 — eval candidates (observed)
- Each candidate: build model shell + load state dict + 1M tok LM loss + throughput.
- ~32 min/candidate × 21 candidates = 11.2 h  
- **Observed: 10 h 51 min**

### 06 — select top 3
- Read 05_candidates.json, sort, write.  
- **Observed: < 1 min**

### 07 — KD smoke (new budget: 2M tokens)
- Load teacher (8B): ~2 min  
- Build + load student (~4B): ~1 min  
- Training steps: 2M / 8192 ≈ **244 steps**  
- Per-step cost (teacher forward + student forward+backward, torch_forward path): ~12–20s  
- Training: 244 × 16s ≈ 3,900s ≈ **65 min**  
- Save checkpoint: ~1 min  
- **Total estimate: ~1–2 h** (safe within 24h limit)

### 08 — KD train (new budget: 20M tokens × 3 candidates)
- Load teacher (8B): ~2 min  
- Per candidate: build/load student + 20M/8192 ≈ **2,441 steps** × ~16s ≈ 39,000s ≈ **6.8 h**  
- 3 candidates: ~20.4 h  
- Model save per candidate: ~2 min  
- **Total estimate: ~20–22 h** (fits within 24h with ~2–4h margin)

### 09 — final eval (estimated)
- Runs `lm-evaluation-harness` on 3 models (parent 8B, best pruned, best KD).
- 7 benchmark tasks per model at ~20 min/task (zero-shot, batch=auto): 7 × 20 × 3 = 420 min  
- **Estimate: ~4–8 h** (needs a separate 24h slot; do not chain after 08)

### 10 — optional mini final KD (1B tokens default)
- 1B / 8192 ≈ 122,000 steps × 16s = ~1,952,000s ≈ **540 h** at 1 candidate  
- **Not feasible at default budget.** Use `--target-tokens 200_000_000` (200M) for a realistic ~27h run if a second overnight job remains.

---

## Notes on step-time estimate

The per-step timing (16s) is derived from the stage 05 eval calibration:
- Stage 05 observed ~10s per eval batch (forward-only, 8192 tokens, 4B student).
- KD training adds: teacher forward (~5s for 8B) + student backward with activation checkpointing (~6s).
- `torch_forward` path (pure PyTorch SSM) is used instead of the fused CUDA training kernel.
  This is ~10–20% slower than the CUDA eval path for the student but avoids the
  `causal_conv1d_fwd incompatible arguments` crash during training.
- Conservative estimate: **~15–20s/step**; table uses 16s as the mid-point.

---

## Token budget changes (vs original plan)

| Parameter | Original | New |
|-----------|---------|-----|
| `smoke_test_tokens` | 10M | **2M** |
| `tokens_per_candidate` (stage 08) | 200M | **20M** |

Changes are in `configs/kd.yaml`.
