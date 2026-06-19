# Concept analysis — linking every measured number to Lecture 08

> Course: **Orchestration of AI Agents (203.3763)**, Dr. Yoram Segal. HW5.
> Every claim below is tied to a measured number in `results/*.json` (the ledger) or
> to the lecture. Nothing is fabricated; the run was on a free **Kaggle Tesla T4**.

## The measured ledger (recap)

| scenario | success | throughput | TTFT | TPOT | peak VRAM |
|---|---|---|---|---|---|
| FP16 baseline | **OOM** | — | — | — | needs 29.4 GB |
| AirLLM FP16 | ✓ | 0.0070 tok/s | 142.3 s | 142.4 s | 1.60 GB |
| AirLLM 8-bit | ✓ | 0.0138 tok/s | 70.1 s | 72.7 s | 2.35 GB |
| AirLLM 4-bit | ✓ | 0.0410 tok/s | 45.7 s | 23.3 s | 3.15 GB |

## 1. VRAM capacity — why the naive run dies (D2)

A parameter costs **bytes = params × bytes/param**. Qwen2.5-14B has ≈14.7 B params, so
in FP16 (2 bytes) the weights alone are **≈29.4 GB**, before activations, the KV-cache,
or the CUDA context. The T4 exposes ≈15.6 GB of usable VRAM. `29.4 GB > 15.6 GB`, so
`AutoModelForCausalLM(..., device_map="cuda")` raises a real
`torch.cuda.OutOfMemoryError` (`results/fp16_baseline.json`: *"GPU 0 has a total
capacity of 14.56 GiB of which 102.81 MiB is free"*). **The bottleneck is memory
capacity, not compute** — the GPU never reaches the first GEMM. This is the lecture's
VRAM wall.

## 2. AirLLM = the virtual-memory / Paging analogy (D3)

An OS runs a program larger than physical RAM by keeping only the *active* pages
resident and **paging** the rest from disk on demand. AirLLM applies the identical
idea to a transformer: it keeps **only the active layer** in VRAM and streams the
others from per-layer **SafeTensors** shards. So:

- **a layer = a page**; loading a layer = a **page fault**; the `mmap` mapping is the
  zero-copy page-in; freeing the layer after use is **eviction / page replacement**.
- **SafeTensors** is the right on-disk format precisely because it is `mmap`-able and
  carries no executable code — a clean, safe page store.

The measured proof is **peak VRAM = 1.60–3.15 GB** while the model "needs" 29.4 GB:
at no instant is the whole model resident. AirLLM converts an *impossible* 29.4 GB
allocation into a steady ≈2–3 GB working set — it **trades VRAM capacity for time**.

## 3. The price of paging — disk bandwidth vs HBM (D9, Roofline)

Why ≈0.007–0.04 tokens/second? Because autoregressive decode is **memory-bound**:
every output token re-reads (essentially) all the weights. Normally those weights are
read from **HBM at ≈320 GB/s**. Under AirLLM they are read from **disk**. From the
ledger, the *effective sustained bandwidth* is

`bytes_per_token × throughput = (14.7e9 × bytes/param) × tok/s`
→ **FP16 ≈ 0.21 GB/s, 8-bit ≈ 0.20 GB/s, 4-bit ≈ 0.30 GB/s**.

That is **~1000–1550× below the T4's 320 GB/s HBM** (`figures/roofline.png`). On the
Roofline, decode sits at arithmetic intensity ≈1–4 FLOPs/byte — far left of the
T4 **ridge ≈203** (65 TFLOP/s ÷ 320 GB/s), squarely in the memory-bound region — and
the measured points sit **three orders of magnitude *below* even the memory roof**,
because the relevant roof is not HBM but the disk read path. **AirLLM trades the
HBM-memory wall for a disk-bandwidth wall.** That single sentence is the whole story.

## 4. Prefill vs Decode — and why AirLLM flattens them (D8)

The lecture splits inference into **Prefill** (process the prompt in one pass — a big
matrix-matrix multiply, **GEMM**, compute-bound, surfaced as **TTFT**) and **Decode**
(generate tokens one at a time — matrix-vector, **GEMV**, memory-bound, surfaced as
**TPOT**). On a normal resident model TTFT ≪ per-token-TPOT in wall terms only because
prefill is parallel; the two regimes have very different arithmetic intensity.

Our numbers show something subtle: **TTFT ≈ TPOT** for FP16 (142.3 vs 142.4 s) and
8-bit (70.1 vs 72.7 s). Under AirLLM, *both* a prefill step and a decode step pay the
**same dominant cost — one full 51-shard sweep off disk**. The disk-load term is so
large that it swamps the GEMM-vs-GEMV compute difference, so the usual Prefill/Decode
asymmetry **collapses**: every step, prompt or token, is disk-bound. AirLLM doesn't
just slow inference down — it *re-shapes* it into a single disk-bound regime.

The 4-bit row breaks the symmetry the other way (TTFT 45.7 s > TPOT 23.3 s). This is
the **OS page cache warming up**: the first token (TTFT) pays cold reads for every
shard; later tokens (TPOT) hit shards still warm in the host page cache, and the 4-bit
shards are small enough (≈0.5 bytes/param) that more of them stay cached — so decode
runs ~2× faster than the first token. A textbook paging effect, visible in the data.

## 5. Quantization — memory, speed, and the accuracy red line (D4)

Quantization lowers **bytes/param**: FP16 = 2 → 8-bit = 1 → 4-bit (NF4-style) = 0.5.
Fewer bytes per weight means **less data to stream off disk per layer**, which is why
throughput rises **≈6×** from FP16 (0.0070) to 4-bit (0.0410 tok/s) and TPOT drops
from 142 s to 23 s. The lecture's "FP32→FP16 halves memory; Q4 is aggressive; NF4 is
the high-quality 4-bit" maps directly onto this monotone speed-up.

Crucially, **the output stayed coherent at every level** — all three answers begin
"The attention mechanism is a key component of …" and read sensibly (`output` field in
each `results/airllm_*.json`). So for this prompt and 20 tokens we **did not cross the
accuracy "red line"**: 4-bit bought a 6× speed-up at no visible quality cost. (A
rigorous red line would need a perplexity/benchmark sweep — see the honesty note.)

### The counter-intuitive VRAM result

Peak VRAM *rises* with more aggressive quantization (1.60 → 2.35 → 3.15 GB) even though
the weights get smaller. The resolution ties back to paging: because only one layer is
ever resident, the **weights are not what dominates VRAM** — the bitsandbytes
**dequantization buffers** (and faster, more-overlapped compute) do. Quantization
shrinks the *disk* footprint (the thing that matters for speed) while *adding* a small
constant VRAM overhead for on-the-fly dequant. It's a real, measured, explainable
trade-off, and it's why the quant Pareto (D10, `figures/pareto.png`) is **speed-vs-VRAM**
with all three points non-dominated.

## 6. KV-cache

Decode also grows the **KV-cache** (the stored keys/values for past tokens), which adds
to VRAM as the sequence lengthens. We capped `max_new_tokens=20` and `max_seq_len=128`
precisely so the KV-cache stays negligible next to the per-layer weight streaming — at
these lengths the run is dominated by paging weights, not by the KV-cache, keeping the
"layer = page" story clean.

## 7. What the lecture predicted, confirmed by measurement

| Lecture concept | Measured confirmation |
|---|---|
| VRAM capacity wall | FP16 baseline OOMs (29.4 > 15.6 GB) |
| Paging / "layer = page" | runs in 1.6–3.2 GB vs 29.4 GB needed |
| Decode is memory-bound | AI ≈1–4 ≪ ridge 203; far left on the Roofline |
| Disk BW ≪ HBM BW | effective ≈0.2–0.3 GB/s vs 320 GB/s (~1500×) |
| Quantization speeds memory-bound work | 4-bit ≈6× FP16 throughput |
| Prefill vs Decode | flattened to one disk-bound regime (TTFT ≈ TPOT) |
