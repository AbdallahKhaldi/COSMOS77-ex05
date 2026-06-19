# AirLLM — the same model runs, layer by layer (D3)

**Scenario `airllm_none` — the treatment condition (FP16, no quantization).**

`from airllm import AutoModel` shards the model into 51 per-layer **SafeTensors** files
and streams them through VRAM one at a time (`mmap`). The same 29.4 GB model that OOM'd
now **runs and produces coherent text**:

> *"The attention mechanism is a key component of transformer models, which are widely
> used for natural language processing …"* (`output` in `results/airllm_none.json`)

Measured: **peak VRAM 1.60 GB** (vs 29.4 GB "needed"), throughput **0.0070 tok/s**,
TTFT **142.3 s**, TPOT **142.4 s/token**, total **2848 s** for 20 tokens.

This is the **virtual-memory / Paging analogy** made literal: *a layer = a page*, a
layer load = a *page fault*, `mmap` = the zero-copy page-in, free-after-use = eviction.
AirLLM **trades VRAM capacity for time** — it runs the impossible model in ≈1.6 GB, at
the cost of brutal latency (see `reports/CONCEPTS.md` for the disk-bandwidth analysis).
