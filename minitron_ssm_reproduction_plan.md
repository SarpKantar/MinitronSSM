# Implementation Plan: Scaled Reproduction of “Minitron-SSM: Efficient Hybrid Language Model Compression through Group-Aware SSM Pruning”

## 0. Core Interpretation of the Reproduction Target

The goal is **not** to exactly reproduce NVIDIA’s released `Nemotron-H-4B-Base-8K` checkpoint. The exact released checkpoint depends on NVIDIA’s internal/partially specified training mixture, a 380B-token final knowledge distillation run, and their full production training infrastructure. The paper states that the final 4B model is obtained by compressing `Nemotron-H 8B` to 4B through pruning and knowledge distillation, with up to 40× fewer training tokens and about 2× faster inference compared to similarly sized models. ([arxiv.org](https://arxiv.org/html/2504.11409v2))

The correct target is a **scaled methodological reproduction**:

> Reproduce the main compression pipeline of the paper at reduced scale: activation-based importance estimation, group-aware Mamba/SSM pruning, architecture candidate search, short knowledge distillation recovery, and internal evaluation of parent vs. pruned vs. distilled models.

The main result to reproduce is a reduced version of the architecture search behind **Table 1**, plus a limited internal version of **Table 4** where only our own models are compared. Table 1 in the paper compares compressed 4B candidate configurations using layers, embedding size, FFN size, Mamba heads, Mamba head channels, LM validation loss, and relative throughput. ([arxiv.org](https://arxiv.org/html/2504.11409v2))

## 1. Final Deliverable

Produce a codebase and report that include:

1. A script to load the parent `Nemotron-H-8B-Base-8K`.
2. A script to collect activation-based importance scores.
3. A pruning module implementing or invoking:
   - Mamba head pruning,
   - Mamba head-channel pruning,
   - FFN width pruning,
   - embedding width pruning,
   - optional depth pruning only as an ablation.
4. A candidate architecture search over approximately **20 compressed candidates**.
5. Zero-shot LM validation loss and throughput measurements for all candidates.
6. Short KD training for the **top 3 candidates**.
7. Final evaluation comparing:
   - parent 8B model,
   - pruned but non-distilled candidates,
   - pruned + KD candidates,
   - optionally NVIDIA’s released 4B model as a reference-only upper bound.
8. A reproduction report explaining what was reproduced, what was approximated, and what could not be reproduced exactly.

## 2. Source Models and Public Assets

Use these public assets:

### Parent / Teacher Model

Use `nvidia/Nemotron-H-8B-Base-8K` as the teacher model. Its model card describes it as an 8B hybrid Mamba-Transformer model with primarily Mamba-2 and MLP layers plus four Attention layers, 8K context length, and support for several languages. ([huggingface.co](https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K))

### Released 4B Model

Use `nvidia/Nemotron-H-4B-Base-8K` only as a reference, not as the target of training. Its model card says it was pruned and distilled from `Nemotron-H-8B-Base-8K` using **380B tokens**. ([huggingface.co](https://huggingface.co/nvidia/Nemotron-H-4B-Base-8K))

### Implementation Tooling

Prioritize NVIDIA’s public tooling:

1. **NVlabs/Minitron**  
   The repository describes Minitron as a family of small language models obtained via pruning and knowledge distillation. It explicitly mentions pruning embedding size, attention heads, and MLP intermediate dimension, followed by continued training with distillation. ([github.com](https://github.com/NVlabs/Minitron))

2. **NVIDIA Model Optimizer**  
   Its pruning README says Minitron-style pruning has been extended to Mamba, MoE, and hybrid Transformer-Mamba models, using activation magnitudes to prune hidden size, FFN size, attention heads, Mamba heads, Mamba head dimension, and model depth. ([github.com](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/pruning/README.md))

3. **NVIDIA NeMo / Megatron-Bridge / Megatron-LM**  
   NeMo documentation supports LLM pruning across depth and width dimensions such as embedding hidden size, FFN hidden size, attention heads, and attention query groups, powered by NVIDIA Model Optimizer. ([docs.nvidia.com](https://docs.nvidia.com/nemo-framework/user-guide/latest/model-optimization/pruning/pruning.html))

If NVIDIA tooling supports `Nemotron-H` directly, use it. If not, implement missing pruning logic manually around the Hugging Face model, but keep this as a fallback because manual hidden-dimension pruning across a hybrid model is fragile.

## 3. Main Experimental Scope

Use the following reduced scope:

### Original Paper Scale

The paper explores many 4B candidate architectures. It reports **125 checkpoints** under a fixed 4B parameter budget and evaluates LM loss, time to first token, and throughput. ([arxiv.org](https://arxiv.org/html/2504.11409v2))

The paper then selects candidates for lightweight KD and finally performs extended KD on the best candidate. The final selected model uses width-only pruning and is trained with **380B tokens**. ([arxiv.org](https://arxiv.org/html/2504.11409v2))

### Reproduction Scale

Do this instead:

```text id="4v6j6k"
Candidate search:
    original: ~125 candidate checkpoints
    reproduction: ~20 candidate checkpoints

Lightweight KD:
    original: top candidates trained with much larger KD budget
    reproduction: top 3 candidates × 200M tokens each = 600M KD tokens

Final KD:
    original: 380B tokens
    reproduction: optional 200M to 1B extra tokens on best candidate if compute remains
```

The reproduction should test the **trend**, not match the final benchmark values.

Expected trend:

```text id="du3e2b"
Parent 8B:
    best accuracy, slowest inference

Pruned 4B-ish model before KD:
    faster inference, worse LM loss

Pruned + short KD model:
    faster than parent, partially recovered LM loss

Released NVIDIA 4B model, optional reference only:
    much better than our reproduction because it used 380B-token KD
```

## 4. Dataset Plan

### Important Limitation

Do not claim exact data reproduction. The paper says it uses a random sample from the **Phase 3 data mixture used for training Nemotron-H models** for both importance estimation and KD. It uses 1024 samples with sequence length 8192 for importance estimation, and KD with sequence length 8192. ([arxiv.org](https://arxiv.org/html/2504.11409v2))

The exact Phase 3 mixture is not fully specified in enough detail to recreate the checkpoint exactly.

### Public Data Substitute

Use public pretraining-style data instead. Recommended options:

1. **Nemotron-CC-v2**  
   This is NVIDIA’s public pretraining dataset for generative AI model training, with math, code, multilingual Q&A, and general text. ([huggingface.co](https://huggingface.co/datasets/nvidia/Nemotron-CC-v2?utm_source=chatgpt.com))

2. **FineWeb**  
   Public Common Crawl-derived English pretraining data, created from 96 Common Crawl dumps. ([huggingface.co](https://huggingface.co/datasets/HuggingFaceFW/fineweb?utm_source=chatgpt.com))

3. **SlimPajama-627B or SlimPajama-6B subset**  
   Public deduplicated pretraining data derived from RedPajama. The full SlimPajama split is large, so use a smaller sampled variant if storage is limited. ([huggingface.co](https://huggingface.co/datasets/MBZUAI-LLM/SlimPajama-627B-DC?utm_source=chatgpt.com))

Recommended practical mixture:

```text id="8ws0b6"
60% Nemotron-CC-v2 or FineWeb
20% code data
10% math/science text
10% QA / instruction-like text
```

Keep a held-out validation set of at least:

```text id="0tszjp"
10M to 50M tokens for LM validation loss
```

Do not train on evaluation benchmark data.

## 5. Baseline Setup

First implement the baseline pipeline before pruning.

### Step 5.1: Load Tokenizer and Teacher

Load:

```text id="nqkcip"
nvidia/Nemotron-H-8B-Base-8K
```

Use BF16. Test inference with a short prompt.

### Step 5.2: Measure Parent Model Metrics

Measure:

```text id="68uvrd"
1. Parameter count
2. LM validation loss on held-out data
3. Throughput
4. Latency / time-to-first-token if possible
5. GPU memory usage
```

Use fixed settings for all models:

```text id="jig9q4"
precision: bf16
sequence length for LM loss: 8192 if possible, otherwise 4096
generation benchmark input length: 4096 or 8192
output length: 256 or 512
batch sizes: 1, 2, 4 if memory allows
hardware: same GPU type for all models
```

Do not compare throughput numbers directly to the paper unless you reproduce the same hardware and sequence lengths. The paper reports throughput using very long input/output settings in some figures, so our numbers should be treated as internal comparisons only. ([arxiv.org](https://arxiv.org/html/2504.11409v2))

## 6. Importance Estimation

The paper’s pruning procedure starts by computing importance/sensitivity scores for Mamba heads, Mamba head channels, FFN neurons, embedding channels, and layers using an activation-based strategy requiring only forward passes. After scoring, components are sorted and low-importance components are pruned, then the pruned model is distilled. ([arxiv.org](https://arxiv.org/html/2504.11409v2))

### Step 6.1: Calibration Dataset

Use:

```text id="vp1viy"
num_samples: 1024
sequence_length: 8192 if possible
fallback_sequence_length: 4096
```

This follows the paper’s importance-estimation setup as closely as possible. ([arxiv.org](https://arxiv.org/html/2504.11409v2))

### Step 6.2: Collect Activations

For each calibration batch, collect activations from:

```text id="bgcppo"
Mamba layers:
    W_x-related activations for Mamba head and head-channel scoring

FFN layers:
    input activations to the first FFN projection

Embedding / hidden channels:
    layer-normalized hidden states across FFN, Mamba, Attention, and LayerNorm components
```

Use forward hooks or the internal activation API from NVIDIA Model Optimizer if available.

### Step 6.3: Mamba Group-Aware Constraint

This is the core novelty. Unlike FFN and embedding dimensions, Mamba heads cannot be freely permuted globally. The paper says Mamba heads must preserve group structure because cross-group head permutation changes the broadcast pattern in the SSM computation. ([arxiv.org](https://arxiv.org/html/2504.11409v2))

Implement:

```text id="d5gaxq"
For Mamba head pruning:
    rank heads only within each Mamba group
    keep the same number of heads per group
    concatenate group-wise rankings afterward

For Mamba head-channel pruning:
    use a shared channel ranking across all heads
    prune or keep each head-channel index uniformly across all heads
```

The paper states that head-channel pruning must maintain consistency across all heads, meaning each channel index is preserved or pruned uniformly. ([arxiv.org](https://arxiv.org/html/2504.11409v2))

### Step 6.4: FFN and Embedding Importance

For FFN and embedding channels, use activation-based importance metrics. The paper computes scores from FFN and LayerNorm activations and keeps the top-k neurons/channels according to the target compression ratio. ([arxiv.org](https://arxiv.org/html/2504.11409v2))

Implement:

```text id="bnfb33"
FFN neuron score:
    aggregate activation magnitude over calibration samples

Embedding hidden-dim score:
    aggregate hidden channel importance across all layers that use the hidden dimension
```

Use mean over sequence length and L2 or absolute magnitude over batch/samples, matching the paper’s Minitron-style activation aggregation where possible.

## 7. Pruning Implementation

### Preferred Route

Use NVIDIA Model Optimizer / NeMo if possible.

Target pruning dimensions:

```text id="mir0mp"
Mamba heads
Mamba head channels
FFN hidden size
Embedding hidden size
Optional depth/layer pruning
```

Model Optimizer explicitly supports Minitron pruning for Mamba heads and head dimension in hybrid Transformer-Mamba models. ([github.com](https://github.com/NVIDIA/Model-Optimizer/blob/main/examples/pruning/README.md))

### Manual Fallback

If manual pruning is required, implement carefully.

#### Mamba Head Pruning

For each Mamba layer:

```text id="1hftw2"
1. Determine Mamba group count.
2. Rank heads inside each group.
3. Select top heads per group.
4. Trim all tensors whose dimensions depend on Mamba head count.
```

Likely affected tensors include:

```text id="z0zzvt"
W_x / x projection
W_z / gating projection
dt projection
A and D SSM parameters
causal convolution channels associated with pruned heads
output projection columns corresponding to pruned heads
```

Verify tensor names from the actual `Nemotron-H` implementation before writing slicing code.

#### Mamba Head-Channel Pruning

For each Mamba layer:

```text id="st8kgy"
1. Compute shared head-channel ranking.
2. Select top channel indices.
3. Apply the same selected channel indices to every head.
4. Trim input/output projections and convolution channels consistently.
```

#### FFN Pruning

For each FFN block:

```text id="jlbykx"
1. Rank FFN intermediate neurons.
2. Keep top-k neurons.
3. Slice gate/up projection output rows.
4. Slice down projection input columns.
```

#### Embedding Hidden-Dimension Pruning

This is the most dangerous manual pruning path because the hidden dimension appears almost everywhere.

If doing manual hidden-dim pruning:

```text id="8v6qhz"
1. Create a global hidden-dim keep index.
2. Slice token embedding output dimension.
3. Slice all projection input/output dimensions consistently.
4. Slice LayerNorm/RMSNorm parameters.
5. Slice LM head input dimension.
6. Update model config hidden_size.
7. Run strict shape checks after every layer.
```

Because this is brittle, prefer Model Optimizer for embedding pruning.

#### Attention Pruning

Do **not** prune attention layers in the main reproduction. The paper says attention layers are not pruned because they are only 8% of total layers. ([arxiv.org](https://arxiv.org/html/2504.11409v2))

## 8. Candidate Architecture Search

### Step 8.1: Candidate Budget

Generate approximately **20 candidates** instead of 125.

Use mostly width-only pruning because the paper finds width-only pruning is much better than depth-only pruning at 50% compression. In Table 1, candidate #1, a width-pruned model, outperforms depth-only pruning; the paper also states that depth-pruned models perform worse despite sometimes having more parameters. ([arxiv.org](https://arxiv.org/html/2504.11409v2))

### Step 8.2: Candidate Search Space

Base parent configuration from the paper:

```text id="amjyph"
layers: 52
embedding: 4096
FFN: 21504
Mamba heads: 128
Mamba head channels: 64
```

Candidate #1 in the paper:

```text id="2ze8js"
layers: 52
embedding: 3072
FFN: 12288
Mamba heads: 112
Mamba head channels: 64
LM val loss after lightweight KD: 1.380
relative throughput: 1.00
```

Candidate #2 in the paper:

```text id="c6lwnz"
layers: 52
embedding: 3072
FFN: 10752
Mamba heads: 128
Mamba head channels: 64
LM val loss after lightweight KD: 1.380
relative throughput: 0.98
```

These are useful anchor candidates. ([arxiv.org](https://arxiv.org/html/2504.11409v2))

Recommended reduced search space:

```text id="fevujl"
layers:
    main: 52
    optional depth-ablation: 48 or 44 only for comparison

embedding:
    3072, 3328, 3584

FFN:
    9984, 10752, 11520, 12288, 13056, 13824, 14592

Mamba heads:
    96, 104, 112, 120, 128

Mamba head channels:
    48, 56, 60, 62, 64
```

Filter candidates so that parameter count is approximately 4B. Use exact parameter counting after pruning.

### Step 8.3: Candidate Evaluation Before KD

For each candidate:

```text id="jxk8xi"
1. Apply pruning from the parent model.
2. Save checkpoint.
3. Validate shape correctness.
4. Run a short generation smoke test.
5. Compute LM validation loss on fixed held-out set.
6. Measure throughput/latency.
7. Log parameter count and memory usage.
```

Select top 3 candidates using a combined rule:

```text id="mx7b31"
primary: lowest zero-shot LM validation loss
secondary: higher throughput
tertiary: simpler pruning pattern / fewer risky operations
```

## 9. Short Knowledge Distillation

The paper uses logit-based KD: the student learns from the teacher’s output probability distribution across tokens. ([arxiv.org](https://arxiv.org/html/2504.11409v2))

### Step 9.1: KD Objective

Use:

```text id="p4afsu"
loss = KLDiv(student_logits / T, teacher_logits / T) * T^2
```

Optional:

```text id="s2uijc"
loss = alpha * KD_loss + (1 - alpha) * CE_loss_on_true_tokens
```

Recommended:

```text id="turvmx"
temperature: 1.0 or 2.0
alpha: 0.9 for KD-heavy training
```

If teacher logits over the full vocabulary are too expensive, store or compute **top-k teacher logits**:

```text id="km6ils"
top_k: 100
```

This is consistent with the paper’s later SFT-KD stage, where the 4B base model is fine-tuned using top-k teacher logits. ([arxiv.org](https://arxiv.org/html/2504.11409v2))

### Step 9.2: KD Budget

Use:

```text id="w64z56"
top_candidates: 3
tokens_per_candidate: 200M
total_lightweight_KD_tokens: 600M
```

Also add a smoke-test mode:

```text id="cfellb"
smoke_test_tokens: 5M to 20M
```

Run smoke test first before committing to 200M-token runs.

### Step 9.3: KD Training Practicalities

Teacher + student together may not fit comfortably on limited GPUs.

Try these options in order:

1. **Online KD**  
   Run teacher forward pass and student forward pass in the same training step.

2. **Teacher logits cache**  
   Precompute top-k teacher logits for the KD dataset and train the student from cached logits.

3. **Hybrid cache**  
   Cache logits for a smaller KD subset first, then switch to online KD if feasible.

Use:

```text id="9obwhd"
precision: bf16
activation checkpointing: enabled
gradient accumulation: enabled
sequence length: 8192 if feasible, otherwise 4096
optimizer: AdamW
scheduler: cosine decay
warmup: short warmup, e.g. 1% to 3% of total steps
```

The paper reports using sequence length 8192, batch size 768, cosine learning-rate schedule, and 60-step warmup for KD, but the exact parsed final learning-rate value should be verified from the PDF before copying because the parsed HTML appears potentially ambiguous. ([arxiv.org](https://arxiv.org/html/2504.11409v2))

## 10. Optional Mini Final KD

After the top-3 short KD runs:

```text id="oh4gpc"
1. Pick the best candidate using LM validation loss and throughput.
2. Continue KD on the best candidate for an additional 200M to 1B tokens if compute remains.
3. Report this as “mini final KD”, not as reproduction of the 380B-token final stage.
```

Do not claim equivalence to the paper’s final model.

## 11. Evaluation Plan

### Main Metrics

Report:

```text id="uqvjdi"
LM validation loss
perplexity
parameter count
throughput
latency / time-to-first-token
GPU memory
```

### Internal Table 1 Reproduction

Create a table like:

```text id="r4ezcn"
Candidate ID
Layers
Embedding dim
FFN dim
Mamba heads
Mamba head channels
Parameter count
Zero-shot LM loss
Post-KD LM loss
Relative throughput
Relative latency
Notes
```

Relative throughput should be normalized to the best reproduced candidate or to parent 8B.

### Internal Table 4 Reproduction

Do not compare against external community models unless you use their official reported values only for context. Instead compare:

```text id="ma6s4p"
Nemotron-H 8B parent
Best pruned candidate before KD
Best pruned candidate after short KD
Optional mini-final KD model
Optional NVIDIA released 4B model as reference-only
```

Use a small subset of the paper’s Table 4 benchmarks:

```text id="mq8042"
ARC Challenge
ARC Easy
HellaSwag
PIQA
Winogrande
OpenBookQA
MMLU subset
GSM8K optional
HumanEval / MBPP optional if time allows
```

The paper’s Table 4 includes ARC, CommonsenseQA, GSM8K, HellaSwag, HumanEval, MBPP, MMLU, OpenBookQA, PIQA, RACE, Social IQA, TruthfulQA, and Winogrande. ([arxiv.org](https://arxiv.org/html/2504.11409v2))

Use **EleutherAI lm-evaluation-harness** for standardized evaluation. It is widely used for evaluating language models and supports many common academic benchmarks. ([github.com](https://github.com/EleutherAI/lm-evaluation-harness?utm_source=chatgpt.com))

## 12. Minimal Ablations

Do not reproduce all ablations. Only include small ablations that support the main story.

Recommended:

### Ablation A: Width vs. Depth

Compare:

```text id="dxcauc"
one width-pruned candidate
one depth-pruned candidate
```

Expected result:

```text id="2hz9up"
Width pruning should preserve accuracy better than depth pruning.
```

The paper explicitly states width-only pruning significantly outperforms depth-only pruning at 50% compression. ([arxiv.org](https://arxiv.org/html/2504.11409v2))

### Ablation B: Mamba Heads vs. Mamba Head Channels

Compare:

```text id="tpxkuv"
one candidate with fewer Mamba heads
one candidate with fewer Mamba head channels
```

Expected result:

```text id="rv44u6"
Mamba head pruning should be more attractive for speed/accuracy trade-off.
```

The paper says pruning Mamba heads consistently gives lower LM loss, lower latency, and higher throughput than pruning Mamba head channels in isolation. ([arxiv.org](https://arxiv.org/html/2504.11409v2))

### Ablation C: FLAP

Skip FLAP unless extra time remains. The paper reports that FLAP gives mixed results before KD and no clear advantage after KD for candidate #1. ([arxiv.org](https://arxiv.org/html/2504.11409v2))

## 13. Success Criteria

The reproduction is successful if it shows the following pattern:

```text id="j8dthp"
1. Pruning reduces parameter count and improves throughput.
2. Pruning alone worsens LM validation loss.
3. Short KD partially recovers LM validation loss.
4. Width-pruned candidates are better than depth-pruned candidates.
5. Mamba head pruning is a useful pruning axis in hybrid Mamba-Transformer models.
6. The reproduced model does not match the released NVIDIA 4B model, but follows the same qualitative trend.
```

Do **not** use the final benchmark score as the only success criterion. The reproduction uses dramatically fewer KD tokens than the paper.

## 14. Compute-Aware Schedule

Assume:

```text id="qbd2l6"
hardware: 2–4 A100 GPUs
time: about 1 month
```

### Week 1: Infrastructure and Baselines

```text id="56s1rx"
- Set up environment.
- Load teacher model.
- Build tokenized data pipeline.
- Run parent LM validation.
- Run parent throughput benchmark.
- Confirm pruning tooling works on a small test.
```

### Week 2: Importance Estimation and Candidate Search

```text id="u0l25w"
- Implement activation collection.
- Implement group-aware Mamba ranking.
- Generate ~20 candidate configs.
- Prune and save candidate checkpoints.
- Run zero-shot LM loss and throughput.
- Select top 3 candidates.
```

### Week 3: Short KD

```text id="qykzpw"
- Run KD smoke test.
- Train top 3 candidates with ~200M tokens each.
- Track LM loss curves.
- Evaluate post-KD validation loss.
```

### Week 4: Final Evaluation and Report

```text id="h5kyq0"
- Select best candidate.
- Optional mini-final KD if time remains.
- Run internal benchmark suite.
- Produce tables and plots.
- Write final report.
```

## 15. Risk Management and Fallbacks

### Risk 1: Nemotron-H Training Support Is Difficult

If the Hugging Face model can run inference but not training/pruning cleanly:

```text id="er9x4q"
Fallback 1:
    Use NVIDIA NeMo / Model Optimizer conversion route.

Fallback 2:
    Use a smaller NVIDIA-supported hybrid/Mamba model if available.

Fallback 3:
    Reproduce the pruning mechanics on a smaller Mamba2/hybrid toy model and clearly label it as a methodological reproduction, not a Nemotron-H reproduction.
```

### Risk 2: Teacher + Student KD Does Not Fit in GPU Memory

Use cached top-k teacher logits.

```text id="4dxf5w"
top_k: 100
dtype: bf16 or fp16
storage: sharded files
```

### Risk 3: 200M Tokens per Candidate Is Too Expensive

Reduce to:

```text id="q60gdp"
50M tokens per candidate
```

Then run a longer KD only for the best candidate.

### Risk 4: Hidden-Dimension Manual Pruning Breaks Model Shapes

Avoid manual embedding pruning if possible. Use Model Optimizer. If manual pruning is unavoidable, first implement only:

```text id="5p8l6z"
Mamba head pruning
FFN pruning
```

Then add embedding pruning later.

## 16. What Not to Claim

Do not claim:

```text id="6l3ezg"
- exact reproduction of Nemotron-H-4B-Base-8K
- reproduction of NVIDIA’s 380B-token KD
- reproduction of the full Phase 3 training mixture
- full Table 4 benchmark reproduction
- full instruct or 128K long-context reproduction
```

Instead claim:

```text id="gxgfjp"
- scaled reproduction of the architecture search and compression pipeline
- public-source approximation of the paper’s method
- internal validation of pruning + KD trends
```

## 17. Final Report Structure

Use this report outline:

```text id="ulah12"
1. Introduction
   - What Minitron-SSM does
   - Why hybrid Mamba-Transformer pruning is difficult

2. Reproduction Target
   - Reduced Table 1 architecture search
   - Partial internal Table 4-style evaluation

3. Differences from Original Paper
   - Fewer candidates
   - Public substitute data
   - Much smaller KD budget
   - No 380B-token final KD
   - No instruct/long-context/safety reproduction

4. Method
   - Importance estimation
   - Group-aware Mamba pruning
   - FFN and embedding pruning
   - Candidate generation
   - KD objective

5. Experiments
   - Parent baseline
   - Candidate search
   - Short KD
   - Internal benchmark evaluation

6. Results
   - LM loss table
   - Throughput table
   - Benchmark table
   - Loss curves
   - Pareto plot

7. Discussion
   - Did KD recover accuracy?
   - Did pruning improve efficiency?
   - Which pruning axes were most useful?
   - Why exact reproduction is not possible from public sources alone?

8. Conclusion
   - Summary of reproduced trends
   - Limitations
   - Future work
```

## 18. One-Sentence Summary for the Implementing LLM

Implement a **scaled public-source reproduction** of Minitron-SSM by starting from `Nemotron-H-8B-Base-8K`, applying activation-based group-aware Mamba/SSM + FFN + embedding pruning to generate about 20 compressed candidates, selecting the top 3 by validation loss and throughput, running short KD with the 8B teacher, and reporting internal comparisons showing that pruning improves inference efficiency while KD partially recovers accuracy.

## 19. Skeleton Mapping (Current Repository)

This appendix maps core plan sections to the scaffolded implementation files in this repository.

- Section 5 (Baseline setup): `scripts/01_baseline.py`, `src/minitron_ssm/models/load.py`, `src/minitron_ssm/eval/lm_loss.py`, `src/minitron_ssm/eval/throughput.py`
- Section 6 (Importance estimation): `scripts/02_importance.py`, `src/minitron_ssm/importance/hooks.py`, `src/minitron_ssm/importance/mamba_scores.py`, `src/minitron_ssm/importance/ffn_scores.py`, `src/minitron_ssm/importance/embed_scores.py`
- Section 7 (Pruning implementation): `scripts/04_prune_candidates.py`, `src/minitron_ssm/pruning/mamba.py`, `src/minitron_ssm/pruning/ffn.py`, `src/minitron_ssm/pruning/embed.py`, `src/minitron_ssm/pruning/depth.py`, `src/minitron_ssm/pruning/apply.py`
- Section 8 (Candidate search): `scripts/03_generate_candidates.py`, `scripts/05_eval_candidates.py`, `scripts/06_select_top3.py`, `src/minitron_ssm/search/param_count.py`, `src/minitron_ssm/search/space.py`, `src/minitron_ssm/search/filter.py`
- Section 9 (Short KD): `scripts/07_kd_smoke.py`, `scripts/08_kd_train.py`, `src/minitron_ssm/kd/losses.py`, `src/minitron_ssm/kd/trainer.py`, `src/minitron_ssm/kd/cache.py`
- Section 10 (Optional mini final KD): `scripts/10_optional_mini_kd.py`
- Section 11 (Evaluation plan): `scripts/09_final_eval.py`, `src/minitron_ssm/eval/harness.py`, `configs/eval.yaml`
- Shared configuration and utilities: `configs/base.yaml`, `configs/data.yaml`, `configs/importance.yaml`, `configs/search_space.yaml`, `configs/kd.yaml`, `src/minitron_ssm/utils/config.py`, `src/minitron_ssm/utils/shape_check.py`

Implementation status rule: modules marked with `TODO(stage-N)` are intentionally scaffold-only and get completed during the corresponding execution stage.
