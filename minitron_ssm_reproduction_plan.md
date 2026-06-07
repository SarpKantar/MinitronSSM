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
- Section 9 (Short KD): `scripts/07_kd_smoke.py`, `scripts/08_kd_train.py`, `scripts/08b_kd_train_round2.py`, `src/minitron_ssm/kd/losses.py`, `src/minitron_ssm/kd/trainer.py`, `src/minitron_ssm/kd/cache.py`
- Section 10 (Optional mini final KD): `scripts/10_optional_mini_kd.py`, `scripts/10b_mini_kd_round2.py`
- Section 11 (Evaluation plan): `scripts/09_final_eval.py`, `scripts/09b_final_eval_from_results.py`, `scripts/11_paper_table4_eval.py`, `scripts/12_report_suite_eval.py`, `scripts/12_report_efficiency.py`, `scripts/13_finalize_report_artifacts.py`, `src/minitron_ssm/eval/harness.py`, `configs/eval.yaml`
- Report-focused launchers: `slurm/run_paper_table4_eval.slurm`, `slurm/submit_paper_table4_eval.sh`, `slurm/run_report_campaign.slurm`, `slurm/submit_report_campaign.sh`
- Shared configuration and utilities: `configs/base.yaml`, `configs/data.yaml`, `configs/importance.yaml`, `configs/search_space.yaml`, `configs/kd.yaml`, `src/minitron_ssm/utils/config.py`, `src/minitron_ssm/utils/shape_check.py`

Implementation status rule: modules marked with `TODO(stage-N)` are intentionally scaffold-only and get completed during the corresponding execution stage.

## 20. Live Implementation Status (Updated in Chat)

This section tracks *actual* repository progress so future chats can resume quickly.

---

### Pipeline execution status (as of 2026-06-07)

| Stage | Script | Status | Output artifact |
|-------|--------|--------|-----------------|
| 01 | `01_baseline.py` | ✅ Done (job 1280917) | `outputs/eval/01_baseline.json` |
| 02 | `02_importance.py` | ✅ Done (prior job) | `outputs/scores/importance.pt` |
| 03 | `03_generate_candidates.py` | ✅ Done (job 1280919) | `outputs/candidates/candidates.json` — 20 candidates |
| 04 | `04_prune_candidates.py` | ✅ Done (job 1280919) | `outputs/checkpoints/*/` — 21 checkpoints |
| 05 | `05_eval_candidates.py` | ✅ Done (job 1280919) | `outputs/eval/05_candidates.json` |
| 06 | `06_select_top3.py` | ✅ Done (job 1280919) | `outputs/eval/06_top3.json` — cand-009, cand-016, cand-006 |
| 07 | `07_kd_smoke.py` | ✅ Done (job 1285584) | `outputs/eval/07_kd_smoke.json` — cand-009, 2M tokens |
| 08 | `08_kd_train.py` | ✅ Done (array 1285594) | `outputs/eval/08_kd_results.json` — 15M/candidate, last_loss ~0.94 |
| 08b | `08b_kd_train_round2.py` | ✅ Done (array 1286270) | `outputs/eval/08_kd_results_round2.json`, `cand-*-kd2` — +15M/candidate |
| 09 | `09_final_eval.py` | ✅ Done (job 1286114, ~7h) | `outputs/eval/09_final_eval.json` — parent / pruned / round-1 KD |
| 09b | `09b_final_eval_from_results.py` | ✅ Done (job 1287433_0, 2h58m) | `outputs/eval/09_final_eval_round2.json` — cand-016-kd2 |
| 09c | `09b` + `--mini-kd-json` | ✅ Done (job 1287433_1, 2h58m) | `outputs/eval/09_final_eval_mini_kd.json` — cand-016-mini-final |
| 10 | `10_optional_mini_kd.py` | ✅ Done (job 1286115, 10M) | `outputs/eval/10_mini_kd.json` — cand-016, last_loss 0.77 |
| 10b | `10b_mini_kd_round2.py` | ✅ Done (job 1287432, 12h55m, +10M) | `outputs/eval/10_mini_kd_round2.json`, `cand-016-mini-final2` |
| 11 | `11_paper_table4_eval.py` | ⚠️ Partial evidence preserved | Parent/final zero-shot + 5-shot MMLU completed; reference zero-shot completed; generation tasks did not fit the deadline |
| 12a | `12_report_suite_eval.py` | ✅ Done (array 1290826 tasks 0–2, 2h58m43s each) | pre-KD, 15M KD, and 35M final on the stable seven-task suite |
| 12b | `12_report_suite_eval.py` | ✅ Done (job 1290826_3, 55m37s) | `outputs/eval/12_report_reference4b_native.json` |
| 12c | `12_report_efficiency.py` | ✅ Done (job 1290826_4, 34m57s) | `outputs/eval/12_report_final35m_efficiency.json` |

**Reproduction target from `ReproducingPlan.txt`:** reduced Table-1 search +
short KD on top-3 + optional mini final KD + internal Table-4 comparison.
**Core pipeline, all planned training branches, and all final report-evaluation
jobs are complete.** No additional training or benchmark jobs are required
before report writing. Artifact consolidation, stage-09 compaction, and final
verification are also complete.

Final report writing can use all completed stage-09 results, the preserved
stage-11 partial results, the complete aligned stage-12 recovery comparison,
the native official-4B reference, and the final efficiency run.

**Round-1 harness trend (`09_final_eval.json`, acc):** parent ≫ pruned; KD recovers on several tasks (e.g. hellaswag 0.25→0.36, piqa 0.49→0.67 vs pruned).

**Final report jobs (submitted 2026-06-06, completed 2026-06-07):**
- `1290826_0`: `cand-016` pre-KD, completed in 2h58m43s, exit `0:0`
- `1290826_1`: `cand-016-kd` at 15M tokens, completed in 2h58m43s, exit `0:0`
- `1290826_2`: `cand-016-mini-final2` at ~35M lineage tokens, completed in
  2h58m43s, exit `0:0`
- `1290826_3`: official NVIDIA 4B reference, completed in 55m37s, exit `0:0`
- `1290826_4`: final 35M efficiency, completed in 34m57s, exit `0:0`

All five passed clean-GPU preflight and published completed JSON. The log scan
found no traceback, CUDA OOM, BrokenPipeError, non-finite result, or partial
benchmark write. The benchmark `.err` files contain lm-eval progress bars and
expected zero-shot override notices. The efficiency `.err` contains non-fatal
AMP deprecation, long-tokenization, cache, and right-padding warnings; the
loss, perplexity, throughput, TTFT, and memory measurements are finite.

**Checkpoint lineage (do not add branches together):**

```text
cand-016 (pruned)
  └─ stage 08: +15M → cand-016-kd
       ├─ stage 08b: +15M → cand-016-kd2
       │    total on this lineage: ~30M tokens
       └─ stage 10: +10M → cand-016-mini-final
            └─ stage 10b: +10M → cand-016-mini-final2
                 total on this lineage: ~35M tokens
```

---

### Latest additions and finalized results (2026-06-06/07)

**New evaluation/report tooling:**

- `scripts/11_paper_table4_eval.py`: resumable mixed-shot Table-4-style
  evaluator. It checkpoints after each shot group and includes a parquet-backed
  Social IQA override for compatibility with `datasets>=4`.
- `slurm/run_paper_table4_eval.slurm` / `slurm/submit_paper_table4_eval.sh`: parent,
  official-4B, pre-KD, and final-model launchers with dependencies and clean-GPU
  preflight.
- `scripts/12_report_suite_eval.py`: focused, stable seven-task zero-shot
  evaluator matching the successful stage-09 protocol.
- `scripts/12_report_efficiency.py`: final-checkpoint LM loss, throughput,
  latency, and peak-memory measurement.
- `scripts/13_finalize_report_artifacts.py`: validates completed source JSON,
  builds the report tables, creates a compact stage-09 copy, reads Safetensors
  headers for effective checkpoint parameter counts, and records Slurm/job/
  checkpoint provenance.
- `slurm/run_report_campaign.slurm` / `slurm/submit_report_campaign.sh`: five independent
  A100 tasks, so one model failure does not block the other report artifacts.
- All new evaluators publish JSON atomically through node-local staging.

**Preserved partial Table-4-style evidence:**

- Job `1287577` parent: zero-shot group and 5-shot MMLU completed; GSM8K was
  only 120/1319 after ~16.5 hours and could not finish inside 24 hours.
- Job `1287580` final 35M model: zero-shot group and 5-shot MMLU completed;
  GSM8K was only 41/1319 after ~4.8 hours and was stopped.
- Job `1287578` official 4B: zero-shot group completed; 5-shot MMLU failed with
  CUDA OOM.
- Timestamped copies were saved as
  `outputs/eval/11_table4_*.partial-preserved-20260606-224106.json`; no completed
  measurements were discarded.
- These stage-11 parent/final scores use the safe `torch_forward` fallback and
  are diagnostic only. Their parent MMLU (24.59%) conflicts with the successful
  native stage-09 parent MMLU (71.26%), demonstrating an evaluation execution
  path mismatch rather than a real quality collapse.

**Latest finalized job: official NVIDIA 4B native stable suite
(`1290826_3`):**

| Task | Accuracy | Normalized accuracy |
|------|---------:|--------------------:|
| ARC-Challenge | 47.95% | 51.54% |
| ARC-Easy | 80.22% | 78.66% |
| HellaSwag | 54.94% | 73.83% |
| PIQA | 78.13% | 78.94% |
| Winogrande | 69.93% | — |
| OpenBookQA | 32.60% | 43.60% |
| MMLU (0-shot) | 55.89% | — |

Artifact: `outputs/eval/12_report_reference4b_native.json`. This run used the
official model's native inference kernels and completed without OOM.

**Latest finalized job: final 35M efficiency (`1290826_4`):**

- Held-out LM loss: **3.28844**
- Perplexity: **26.801**
- Tokens evaluated: **1,007,493**
- Batch 1, 4096→256: **4.679 tok/s**, TTFT **202.1 ms**, peak **15.48 GiB**
- Batch 4, 4096→256: **4.862 tok/s**, TTFT **791.5 ms**, peak **34.72 GiB**

Artifact: `outputs/eval/12_report_final35m_efficiency.json`. The LM-loss phase
and throughput phase both completed successfully with finite results.

**Final aligned cand-016 recovery results (`1290826_0`–`1290826_2`):**

| Checkpoint | ARC-C norm | ARC-E norm | HellaSwag norm | PIQA norm | Winogrande | OpenBookQA norm | MMLU |
|------------|-----------:|-----------:|---------------:|----------:|-----------:|----------------:|-----:|
| pre-KD | 25.68% | 26.18% | 25.88% | 49.95% | 50.75% | 26.00% | 23.49% |
| 15M KD | 29.78% | 50.59% | 47.24% | 66.43% | 55.96% | 30.20% | 23.81% |
| final ~35M | 31.14% | 53.87% | 48.94% | 67.68% | 54.22% | 33.40% | 24.56% |

The final model improves six of the seven headline metrics over pre-KD.
Winogrande peaks at 15M and then falls by 1.74 percentage points, but remains
3.47 points above pre-KD. MMLU improves only 1.08 points from pre-KD to 35M,
consistent with the limited KD scale and the expectation that this training
does not strongly improve static knowledge recall.

Artifacts:
- `outputs/eval/12_report_pre_kd.json`
- `outputs/eval/12_report_kd15m.json`
- `outputs/eval/12_report_final35m.json`

---

### Known gaps (document in final report)

1. **Mamba structural pruning not applied** — stage 04 silently no-op'd on `block.mixer`; only FFN + embedding pruned. Success criterion #5 (Mamba head axis) is **not demonstrated**.

2. **KD scale far below paper/plan** — 30M tokens on the round-2 branch and
   approximately 35M on the final mini-KD lineage, vs plan 200M and paper 200M
   lightweight / 380B final. `seq_len` 1024 vs 8192; `torch_forward` ~218 tok/s
   on one A100.

3. **Harness coverage by eval artifact** — `09_final_eval.json` = parent +
   pruned + round-1 KD. `09_final_eval_round2.json` = cand-016 round-2 KD
   (~30M branch). `09_final_eval_mini_kd.json` = cand-016 mini-final (~25M
   branch). `12_report_final35m.json` contains the final ~35M checkpoint.
   Their consolidated recovery table is `outputs/report/kd_recovery.{csv,json}`.

4. **Slurm array task-2 cosmetic failures** — jobs `1285594_2` and `1286270_2` exit 6 (post-Python abort) after successful training; artifacts are valid. Broke `afterok` dependency for job 1286273.

5. **Depth-pruning ablation** — not run as a separate comparison (plan optional).

6. **Tables 2, 3, 5–8** — intentionally skipped per `ReproducingPlan.txt`.

7. **NFS/Lustre and shared-GPU fragility** — JSON reads and Slurm log writes failed intermittently on compute nodes; job 1287295 also landed on a GPU with only 4.65 GiB free and OOM'd. Follow-up launchers now embed metadata JSON, publish compact results atomically, use stable home-backed Slurm logs, require a clean visible GPU, automatically requeue contaminated allocations, and exclude problematic nodes `palamut3`, `palamut5`, and `palamut6`.

8. **Benchmark protocol remains primarily internal** — the stable comparison
   suite is zero-shot for all seven tasks, including MMLU. A paper-aligned
   mixed-shot campaign was attempted and produced useful partial evidence, but
   GSM8K generation was far too slow and the reference 5-shot MMLU run OOM'd.
   Therefore, use stable-suite results for quantitative model comparisons and
   present stage-11 mixed-shot results as protocol/runtime diagnostics rather
   than a complete Table-4 reproduction.

---

### Token / compute budget history — planned vs. actual vs. paper

This table is the single reference for every budget decision made during
implementation. Training reductions were forced by the memory/speed constraints
of one 80 GB A100 per model and the `torch_forward` fallback instead of the
fused CUDA kernel. Independent final evaluations were later parallelized across
five A100s.

| Stage | Parameter | Paper target | Original plan | Actual implemented | Reason for cut |
|-------|-----------|-------------|---------------|-------------------|---------------|
| 02 | calibration batches | 1024 × seq 8192 | same | same | no cut needed |
| 05 | val tokens per candidate | ~1M (typical) | 1M | **1M** | no cut |
| 07 | `smoke_test_tokens` | not in paper | **10M** | **2M** | OOM at seq 8192; OOM at seq 2048 with GA=768 (no updates); after GA fix, 2M ran cleanly in ~2.5 h |
| 08 | `tokens_per_candidate` | **200M** | **200M** | **15M × 2 rounds = 30M unique** | torch_forward ~4.7 s/step; 15M ≈ 19h/slot; round 2 via `08b_kd_train_round2.py` (+15M, seed+1000) |
| 08 | `seq_len` | 8192 | 8192 | **1024** | forward OOM at 8192 (64 GiB transient); OOM at 2048 after optimizer step (16 GiB transient + ~56 GiB resident > 80 GB); 1024 → 8 GiB transient, ~12 GB headroom |
| 08 | `grad_accumulation` | ~768 (global/micro) | `"auto"` = 768 | **8** | GA=768 with 15M budget = 11 updates total (effectively no training); GA=8 → 1832 updates per candidate |
| 10 | `target_tokens` | 380B (full paper final KD) | 1B (script default) | **10M** from round-1 KD | 1B at 4.7 s/step ≈ 56 days; 10M ≈ 13h |
| 10b | additional continuation | — | optional | **+10M, completed** | resumed stage-10 checkpoint on a fresh seed; final mini-KD lineage reached ~35M total |

**Throughput reality check:** The paper trains on multiple A100s with the fast
fused Mamba CUDA kernel. Our single-GPU `torch_forward` fallback runs at
~218 tok/s (seq 1024, micro 1, GA 8), making token budgets much more expensive
in wall-clock time. Throughput does not change the value of a token already
processed; it limits how many tokens can be processed before the deadline.
The dominant quality gaps are therefore the much smaller token budget, shorter
sequence length, public substitute data, and missing Mamba structural pruning.

**Fidelity summary vs. paper:**
- Embedding + FFN pruning: ✅ implemented correctly
- Mamba head/channel pruning: ❌ silently no-op'd (see blocker note below) — checkpoints retain parent Mamba dims (128 heads, 64 head_dim)
- KD loss (forward KL + CE blend): ✅ implemented
- KD token budget: **30M** on the round-2 branch and **~35M** on the completed
  mini-final branch, versus 200M lightweight KD in the paper
- `seq_len` during KD: 1024 / 8192 = **12.5% of paper**; affects gradient diversity

---

### Known blocker: Mamba pruning silently no-op'd in stage 04

**Root cause:** `apply_candidate` in `src/minitron_ssm/pruning/apply.py` passes a `NemotronHBlock` (the full transformer block) to `prune_mamba_heads` and `prune_mamba_head_channels`, but those functions call `_get_int_attr(layer, ["n_heads","num_heads"], 0)` which returns `0` on the block because `num_heads` lives on `block.mixer`, not on the block itself. Both functions then hit the early-return guard `if n_heads <= 0: return` and silently do nothing.

**Effect:** The saved checkpoints in `outputs/checkpoints/*/` have:
- ✅ FFN neurons pruned correctly
- ✅ Embedding/hidden dim pruned correctly (e.g. 4096 → 3328 for cand-009)
- ❌ `mamba_num_heads` still 128 (parent value, not pruned to 120)
- ❌ `mamba_head_dim` still 64 (parent value, not pruned to 48)

**Evidence:** Stage 07 (job 1285548) failed with:
```
size mismatch for backbone.layers.0.mixer.dt_bias:
    checkpoint shape [128], model shell shape [120]
```
because `build_pruned_model` was updated to override `mamba_num_heads=120` in the config but the actual checkpoint weight has 128 heads.

**Fix options for next chat:**

Option A — Fastest (no re-run needed): Remove the Mamba-dimension overrides from `build_pruned_model` (`mamba_num_heads`, `mamba_head_dim`, `mamba_groups`). The model shell will be built with the actual saved dimensions (128/64), matching the checkpoint. KD will train a model that has FFN + embedding pruned but not Mamba-pruned. Still a valid partial reproduction.

Option B — Correct but expensive (re-run stage 04+05+06): Fix `apply_candidate` to drill into `block.mixer` when calling `prune_mamba_heads` / `prune_mamba_head_channels`, then re-run stage 04 (6 min), 05 (11 h), 06 (<1 min), 07, 08. Costs another overnight run just for stage 05.

**Recommendation given 5-day deadline:** Apply Option A immediately so KD can proceed on existing artifacts. Document Mamba pruning as a known limitation of this reproduction run.

**Fix location:**
- File: `src/minitron_ssm/models/load.py`, function `build_pruned_model`
- Remove these three override entries from the `overrides` dict:
  ```python
  "mamba_heads": ("mamba_num_heads", "num_heads"),
  "mamba_head_channels": ("mamba_head_dim", "head_dim"),
  "mamba_groups": ("n_groups",),
  ```

---

### Known blocker (RESOLVED): CUDA OOM in stage 07/08 KD forward pass

**Symptom:** Both `slurm/run_kd_smoke.slurm` (job 1285550) and `slurm/run_kd_train.slurm`
(job 1285551) crashed with:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 GiB.
... at modeling_nemotron_h.py:647
    G_intermediate = C[:, :, :, None, :, :] * B[:, :, None, :, :, :]  # (b, c, l, s, h, n)
```

**Root cause:** `disable_fast_mamba_kernels` repoints the student's Mamba mixers
to `torch_forward` (the naive SSD reference path) to avoid the
`causal_conv1d_fwd() incompatible arguments` TypeError in the fused training
kernel. That naive path materialises a single fp32 tensor of shape
`(b, c, l, s, h, n)` = `(1, seq_len/128, 128, 128, 128, 128)` =
**`(seq_len/128)` GiB**, i.e. exactly 64 GiB at `seq_len=8192`. It is one
contiguous allocation, so it cannot be sharded or checkpointed away — the only
lever is `seq_len`. (The teacher is unaffected: it still uses the fused
`cuda_kernels_forward` which never builds this tensor.)

A second latent OOM was also waiting at the first `optimizer.step()`:
teacher (~17 GB) + student (~10 GB) + grads (~10 GB) + AdamW moments (~19 GB,
bf16 since the student params are bf16) ≈ 56 GB fixed, leaving only ~24 GB for
the transient `G_intermediate`. `seq_len=2048` (16 GiB) fits this budget;
`seq_len=4096` (32 GiB) would not.

**Fix applied:**
- `configs/kd.yaml`: `training.seq_len` 8192 → 2048 (with an explanatory comment).
- `slurm/run_kd_smoke.slurm` / `slurm/run_kd_train.slurm`: export
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce allocator
  fragmentation (explicitly recommended by the OOM message).

If 2048 still proves tight in practice, the next step down is `seq_len=1024`
(8 GiB transient). Installing `bitsandbytes` for 8-bit AdamW would free a
further ~10 GB and is the path to restoring a longer `seq_len`.

---

### Known blocker (RESOLVED): KD loss flat + second OOM after enabling updates

**Symptom 1 (flat loss):** stages 07/08 ran but the loss just oscillated in the
10–17 band with no downward trend, and the smoke and train jobs printed
*byte-identical* loss values step-for-step.

**Root cause 1:** `grad_accumulation` was `"auto"` => `global_batch_size /
micro_batch_size = 768/1 = 768` micro-steps per optimizer update. With the
slashed token budgets the runs reached only ~1 (smoke) / ~12 (train) actual
weight updates, so the model was effectively frozen — identical losses across
jobs were just deterministic batch noise on the un-updated init weights. A
secondary scheduler bug tied the cosine-decay horizon to `warmup_steps+1`, so
the LR would have collapsed to its floor within ~120 updates regardless of run
length.

**Fix 1:**
- `configs/kd.yaml`: `grad_accumulation: 8` (explicit; ~tokens/update = 8 *
  micro * seq_len). Now ~245 updates (smoke) / ~2442 updates per candidate
  (train).
- `src/minitron_ssm/kd/trainer.py`: scheduler is built in `train()` from the
  real total optimizer-step count (`ceil(target_tokens / tokens_per_update)`),
  warmup-then-cosine over the whole run.
- `src/minitron_ssm/kd/losses.py`: CE term now uses the causal next-token shift
  (was comparing logits at position t with the token *at* t).

**Symptom 2 (second OOM):** once updates actually fired, the first
`optimizer.step()` allocated AdamW moment buffers (~19 GB, bf16) and the *next*
backward's `(seq_len/128)` GiB transient overflowed the 80 GB card by ~1 GiB at
seq_len=2048 (observed 64.2 GB resident + 16 GiB alloc).

**Fix 2:** `configs/kd.yaml` `seq_len` 2048 → **1024** (transient 16→8 GiB,
leaving ~12 GB headroom on top of teacher+student+grads+optimizer ≈ 56 GB).

**Verification (jobs 1285584 smoke / 1285585 train, seq_len=1024, grad_accum=8):**
no OOM past the optimizer step; loss falls 12.3 → 7.3 over the first 60 steps
with the LR correctly ramping through warmup (1.7e-6 → 6.7e-6). ~4.7 s/step.

**Stage 08 wall-time (RESOLVED):** at seq_len=1024 a candidate is ~step-bound at
~4.7 s/step, so 20M tokens ≈ 25 h overflowed a 24 h slot — and the old script
trained all 3 candidates sequentially in one job (checkpoints saved only after
each `train()` returns). Restructured to **one candidate per job**:
- `configs/kd.yaml`: `tokens_per_candidate` 20M → **15M** (~14.6k steps ≈ 19 h,
  inside 24 h with margin).
- `scripts/08_kd_train.py`: new `--candidate-index N` trains only `top[N]`,
  writes `outputs/eval/08_kd_results.candNN.json`, then atomically re-merges the
  aggregate `08_kd_results.json` (ordered by candidate index, so `[0]` stays the
  top candidate for stage 09). Omitting the flag keeps the legacy all-in-one loop.
- `slurm/run_kd_train.slurm`: now a Slurm array `--array=0-2`; each task gets its own
  GPU + 24 h window. Logs go to `slurm-%A_%a.{out,err}` with timestamped symlinks.

Verified (array job 1285594): task 0→cand-009, 1→cand-016, 2→cand-006, all
training (1832 optimizer steps each) with no OOM.

The durable fix for both memory and speed remains repairing the fused
`causal_conv1d_fwd` kernel so the fast Mamba path can run instead of torch_forward.

---

### Final 12-hour report campaign (implemented)

With training complete and only 12 hours left before report writing, the
highest-value choice was evaluation rather than another short KD continuation.
The report needed an aligned recovery curve and final efficiency evidence more
than another checkpoint with too little time for complete evaluation.

The implemented five-GPU campaign is:

```text
1290826_0: cand-016 pre-KD          → stable seven-task suite
1290826_1: cand-016-kd, 15M         → stable seven-task suite
1290826_2: cand-016-mini-final2     → stable seven-task suite
1290826_3: official NVIDIA 4B       → stable seven-task suite, native kernels
1290826_4: cand-016-mini-final2     → LM loss + throughput + TTFT + memory
```

This fills the most important report gaps:

1. A directly aligned pre-KD → 15M KD → final ~35M KD comparison.
2. A native official-4B reference under the same stable protocol.
3. Final selected-model LM loss and efficiency measurements.
4. Independent jobs, allowing up to five A100s without one failure blocking the
   other evidence.

All five jobs completed successfully. The three candidate jobs finished in
2h58m43s each; the official reference finished in 55m37s; final efficiency
finished in 34m57s.

---

### Implementation closure and report artifacts

All implementation tasks requested before report writing are complete:

1. The final Slurm logs and JSON outputs were verified.
2. The candidate/Table-1 summary was generated as
   `outputs/report/candidate_table1.{csv,json}`.
3. The aligned pre-KD → 15M → 25M/30M → 35M recovery table was generated as
   `outputs/report/kd_recovery.{csv,json}`.
4. The benchmark table was generated as
   `outputs/report/benchmark_compact.{csv,json}`.
5. The efficiency table was generated as
   `outputs/report/efficiency.{csv,json}`.
6. The 407 MB `09_final_eval.json` was preserved unchanged and compacted to
   `outputs/eval/09_final_eval_compact.json` (approximately 483 KB) by removing
   only per-example `samples`; all 68 result entries per model remain.
7. Exact source hashes, generated-artifact hashes, Slurm accounting records,
   and 29 checkpoint records were written to
   `outputs/report/job_checkpoint_manifest.json`.

The candidate table records both planned search-space parameter counts and
effective Safetensors checkpoint counts. For the selected candidates, the
planned values are approximately 4.02B while the effective saved checkpoints
are 4.864B, which makes the Mamba structural-pruning limitation explicit.

No implementation work remains except final report writing. The report must
retain the documented limitations: no Mamba structural pruning, reduced/public
data, short KD context, small KD budget, zero-shot primary benchmark protocol,
and partial stage-11 mixed-shot evidence.

---

### KD loss trajectory (stages 07/08) — analysed, NOT a blocker

Per-step training loss decreases smoothly through warmup, briefly spikes at the
LR peak (~step 610, recovers), then "oscillates" 0.1–1.6 for the rest. This is
expected noise: `micro_batch_size=1` means each logged value is the loss on one
1024-token sequence, and the trainer logs the *instantaneous* (un-averaged)
loss. The smoothed trend keeps improving — for cand-009, first-half mean 1.93 →
second-half mean 0.92, sub-1.0 fraction 31%→52%; all 3 candidates converge to
last_loss ~0.94. The saved checkpoints are valid; proceed to stage 09.
(Optional future polish: log a running-mean loss, and/or lower the 5e-5 peak LR
or lengthen warmup to remove the step-610 spike.)

### Stage 09/10 enablement (this session)

- **Checkpoint-format blocker (fixed):** `save_candidate` writes `config.json`
  as candidate metadata, not an HF config, so the harness's
  `lm_eval(model="hf", pretrained=<path>)` loader (`from_pretrained`) cannot load
  the pruned/KD checkpoints. Fixed by evaluating in-memory:
  - `src/minitron_ssm/eval/harness.py`: `run_harness` now accepts an in-memory
    `transformers` model (+ tokenizer) and wraps it in lm-eval's `HFLM`; the
    string/path branch still uses the `from_pretrained` loader (for the parent).
  - `scripts/09_final_eval.py`: loads the parent once, rebuilds `best_pruned` /
    `best_kd` via `build_pruned_model` + `load_candidate` + `disable_fast_mamba_kernels`
    (same path as stages 07/08), evaluates one at a time, freeing GPU between.
- **`scripts/10_optional_mini_kd.py` (fixed):** replaced `copy.deepcopy(teacher)`
  + `load_state_dict(strict=False)` (a full 8B student that OOMs and silently
  drops shape-mismatched weights) with the `build_pruned_model` + `strict=True`
  path, resuming from the best stage-08 KD checkpoint.
- **Follow-up Slurm scripts:** `slurm/run_final_eval_followups.slurm` runs round-2 and
  mini-final harness evaluation as separate array tasks;
  `slurm/run_mini_kd_round2.slurm` runs stage 10b for another 10M tokens.
  `slurm/submit_final_eval_followups.sh` and `slurm/submit_mini_kd_round2.sh` embed metadata
  JSON on the login node. They use GPU preflight/requeue logic and stable
  home-backed logs under `/arf/home/skantar/minitron_job_logs/`.

### Stage 11/12 report enablement (latest session)

- The first paper-aligned evaluator failed on Social IQA because
  `datasets==4.8.5` no longer executes repository dataset scripts. The new
  evaluator supplies the same task configuration against Hugging Face's
  `refs/convert/parquet` revision; all 1954 Social IQA validation records load.
- Mixed-shot evaluation is resumable by group: zero-shot, 5-shot MMLU, 8-shot
  GSM8K, 0-shot HumanEval, and 3-shot MBPP. Completed groups are written before
  the next group starts.
- HumanEval/MBPP are explicitly gated through
  `confirm_run_unsafe_code=True` and `HF_ALLOW_CODE_EVAL=1`.
- Long generation tasks proved incompatible with the final wall-clock budget,
  so the completed groups were preserved and the campaign was replaced with
  the stable seven-task report suite.
- Before cancelling jobs `1287577`, `1287579`, and `1287580`, their JSON files
  were copied to timestamped `partial-preserved` artifacts.
- Array `1290826` launched five independent jobs on five A100s. After the
  requested five-minute observation window, every task had loaded its model,
  entered application work, published a valid JSON status file, and showed no
  traceback, OOM, shape mismatch, or non-finite loss.
- All five array tasks later completed with exit code `0:0`; the final logs and
  completed JSON were re-checked before artifact consolidation.

### Code changes made in this chat session

- `src/minitron_ssm/models/load.py`:
  - Added `disable_fast_mamba_kernels(model)` — monkey-patches `cuda_kernels_forward → torch_forward` on all Mamba mixer blocks to avoid `TypeError: causal_conv1d_fwd() incompatible arguments` during training.
  - `build_pruned_model` reconstructs saved candidate shapes and intentionally
    does not override Mamba dimensions because stage-04 checkpoints retained
    the parent Mamba tensor shapes.
- `scripts/07_kd_smoke.py`: calls `disable_fast_mamba_kernels` on the KD student.
- `scripts/08_kd_train.py`: aligned with stage-05 loading pattern (`build_pruned_model` + `strict=True`); calls `disable_fast_mamba_kernels`.
- `configs/kd.yaml`: reduced token budgets (`smoke_test_tokens` 10M→2M,
  `tokens_per_candidate` 200M→15M), sequence length 8192→1024, and gradient
  accumulation to 8.
- `slurm/run_kd_smoke.slurm`: new Slurm script for stage 07 only.
- `slurm/run_kd_train.slurm`: new Slurm script for stage 08 only.
- Follow-up and report Slurm scripts use stable home-backed logs plus
  repository symlinks.
- `runtime_estimates.md`: per-script wall-time estimates table.
- Added `scripts/11_paper_table4_eval.py`,
  `scripts/12_report_suite_eval.py`, `scripts/12_report_efficiency.py`, and
  their Slurm submitters.
- Added `scripts/13_finalize_report_artifacts.py` and generated the final
  compact tables, stage-09 compact copy, and provenance manifest.

---

### Non-blocking warnings (safe to ignore)

- `FutureWarning` from `mamba_ssm` AMP decorators (`custom_fwd`/`custom_bwd`)
- `torch.load weights_only=False` future warning
- `Token indices sequence length > 8192` tokenizer warning
- `right-padding detected` generation warning
- `FFN scores insufficient, falling back to weight-based scoring` — expected when activation-based scores don't cover the target FFN width

---

### Verification done

- `python -m compileall -q src scripts` passed (no syntax errors).
- `python scripts/07_kd_smoke.py --dry-run` and `08_kd_train.py --dry-run` both passed, showing new token budgets correctly.
- `python -m py_compile scripts/11_paper_table4_eval.py
  scripts/12_report_suite_eval.py scripts/12_report_efficiency.py` passed.
- `python -m compileall -q src scripts tests` passed after finalization.
- `bash -n` passed for all new Slurm and submit scripts.
- `sbatch --test-only` accepted both report launchers.
- The stage-12 five-minute startup check found all five array tasks healthy;
  tasks 3 and 4 later completed with exit code 0.
- Final Slurm accounting confirms all stage-12 tasks completed with exit
  `0:0`; strict JSON parsing passed for all report artifacts and the compact
  stage-09 result.
- `/arf/home/skantar/anaconda3/envs/minitron/bin/python -m pytest -q` passed:
  **35 tests passed** (final cached run: 6.02 seconds).
- `git diff --check` passed for the final script, test, and plan edits.
