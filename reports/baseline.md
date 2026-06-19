# Baseline — the naive FP16 direct load (D2)

**Scenario `fp16_baseline` — the control condition that must fail.**

We loaded `Qwen/Qwen2.5-14B-Instruct` the obvious way —
`AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=float16, device_map="cuda")`
— on the Kaggle **Tesla T4 (≈15.6 GB usable VRAM)**. It raised, for real:

> `torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 136.00 MiB.
> GPU 0 has a total capacity of 14.56 GiB of which 102.81 MiB is free.`
> *(verbatim from `results/fp16_baseline.json`)*

**The bottleneck is memory capacity, not compute.** The param→memory math predicted it:
14.7 B params × 2 bytes (FP16) ≈ **29.4 GB** of weights alone — before activations, the
KV-cache, or the CUDA context — which is **1.8× the T4's 16 GB**. The GPU never reached
the first matrix multiply; it failed while *placing the weights*. This is the VRAM wall
the rest of the experiment is built to climb over. Recorded with `success=false`.
