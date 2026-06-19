# AirLLM measured metrics

**Hardware:** Tesla T4 (15.6 GB VRAM), 33.7 GB host RAM, platform `Linux-6.12.90+-x86_64-with-glibc2.35`.

## Scenario comparison

| scenario | success | throughput_tok_s | ttft_s | tpot_s | peak_vram_gb | peak_ram_gb | total_s | est_power_wh | requested_vram_gb |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FP16 baseline (OOM) | False |  |  |  |  |  |  |  | 29.4 |
| AirLLM FP16 | True | 0.007022 | 142.337 | 142.424 | 1.595 | 4.788 | 2848.4 | 55.385 |  |
| AirLLM 8-bit | True | 0.013784 | 70.094 | 72.679 | 2.351 | 4.805 | 1451.0 | 28.214 |  |
| AirLLM 4-bit | True | 0.040964 | 45.653 | 23.294 | 3.154 | 4.815 | 488.2 | 9.494 |  |

## Takeaways

AirLLM runs the 29.4 GB FP16 model at 1.60–3.15 GB peak VRAM by paging one layer at a time — quantization shrinks the per-layer page further. The cost is throughput: TPOT runs 23–142 s/token, the disk-bandwidth price of streaming weights from storage. 4-bit is ~6x the FP16-AirLLM throughput, while the FP16 baseline OOMs (29.4 GB > 16.0 GB available) and never reaches the compute stage.
