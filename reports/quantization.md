# Quantization sweep — FP16 → 8-bit → 4-bit (D4)

**Scenarios `airllm_8bit`, `airllm_4bit` — bitsandbytes compression on the T4 (Turing).**

| level | bytes/param | throughput | TTFT | TPOT | peak VRAM | output |
|---|---|---|---|---|---|---|
| FP16 | 2.0 | 0.0070 tok/s | 142.3 s | 142.4 s | 1.60 GB | coherent |
| 8-bit | 1.0 | 0.0138 tok/s | 70.1 s | 72.7 s | 2.35 GB | coherent |
| 4-bit | 0.5 | 0.0410 tok/s | 45.7 s | 23.3 s | 3.15 GB | coherent |

**Memory/speed:** fewer bytes per weight ⇒ less data to stream off disk per layer ⇒
**throughput rises ≈6×** from FP16 to 4-bit (TPOT 142 s → 23 s). The lecture's
"FP32→FP16 halves memory; Q4 is aggressive; NF4 is the high-quality 4-bit" maps directly.

**Quality / the accuracy "red line":** every level produced a coherent answer beginning
"The attention mechanism is a key component …" — so for this prompt at 20 tokens **the
red line was not crossed at Q4**. (A rigorous red line needs a perplexity/benchmark
sweep — stated honestly as future work.)

**The counter-intuitive bit:** peak VRAM *rises* with more aggressive quantization
(1.6 → 3.2 GB) because, under paging, the resident weights are tiny and the
**bitsandbytes dequantization buffers** dominate VRAM. Quantization shrinks the *disk*
footprint (which governs speed) while adding a small constant VRAM overhead — the basis
of the quant Pareto (`reports/EXTENSIONS.md`).
