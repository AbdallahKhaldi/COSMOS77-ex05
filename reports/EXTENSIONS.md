# Original extension — the quantization speed-vs-VRAM Pareto (D10)

> Built from the committed real ledger (`results/*.json`); figure
> `figures/pareto.png`, code `src/cosmos77_ex05/extensions/pareto.py`.

## What and why

The assignment asks for ≥1 original initiative beyond the required sweep. Ours turns
the three measured AirLLM scenarios into a **Pareto analysis of the quantization
trade-off** — the kind of decision an engineer actually makes when choosing a
precision for a deployment: *which level is "best" depends on what you are optimising.*

We plot **throughput (tokens/s)** against **peak VRAM (GB)** for FP16 / 8-bit / 4-bit
and compute the Pareto frontier (a point is *dominated* only if another is at least as
fast **and** uses no more VRAM, with one strict inequality). Output quality is the
third axis and is **flat** here — all three answers were coherent (no accuracy "red
line" crossed at Q4), so the decision reduces to speed vs VRAM.

## The result (measured)

| Level | Throughput (tok/s) | Peak VRAM (GB) | Dominated? |
|---|---|---|---|
| FP16 (`none`) | 0.0070 | 1.60 | no |
| 8-bit | 0.0138 | 2.35 | no |
| 4-bit | 0.0410 | 3.15 | no |

**All three points are Pareto-non-dominated** — the frontier is the whole set. This is
the counter-intuitive finding: people reach for quantization to *save* VRAM, but under
AirLLM lower precision *raises* peak VRAM (1.6 → 2.4 → 3.2 GB) while *raising*
throughput (≈6× from FP16 to 4-bit).

## Why it happens (ties back to the concepts)

Under AirLLM only one transformer layer is resident at a time, so the **resident
weights are tiny regardless of precision** — they are not what sets peak VRAM. What
quantization shrinks is the **on-disk** footprint (≈0.5 bytes/param at 4-bit vs 2 at
FP16), and *that* is the quantity that governs speed, because the run is bound by
**streaming weights off disk** (see `reports/CONCEPTS.md` §3, §5). Meanwhile
bitsandbytes adds a small, roughly constant **dequantization-buffer** overhead in VRAM
for the lower-precision paths. Net: precision trades a little VRAM for a lot of disk
bandwidth — hence a real, non-trivial Pareto rather than a single dominant choice.

## How to read it for a decision

- **Tightest VRAM budget** (e.g. sharing the GPU): pick **FP16-AirLLM** — 1.6 GB.
- **Best latency / throughput**: pick **4-bit** — ≈6× faster, still coherent, at 3.2 GB.
- **8-bit** is the balanced middle and is itself on the frontier.

Reproduce: `uv run cosmos77-airllm analyze` regenerates `figures/pareto.png` from the
committed ledger — no GPU, no re-run, every number traceable to `results/*.json`.
