# SETUP — reproducing the AirLLM benchmark on a free GPU (D13)

The heavy live runs happen on a **free cloud GPU**, not on the Mac (`bitsandbytes`
quantization is CUDA-only). The notebook `airllm_benchmark.ipynb` runs the WHOLE
experiment top-to-bottom and writes every artifact under `/kaggle/working/results`
(the downloadable output). Our tested `cosmos77_ex05` library is pip-installed
from this public repo, so the notebook contains **no copy-pasted logic** — it just
calls the SDK.

## A. Kaggle (preferred — 16 GB T4, 32 GB RAM, 20 GB persistent, background exec)

1. **Account + GPU.** Create a free Kaggle account and **verify your phone**
   (Settings → Phone) — this unlocks GPU **and** internet inside notebooks.
2. **New notebook.** Either upload `airllm_benchmark.ipynb`, or let the API push it
   (see below). In the notebook settings panel:
   - **Accelerator → GPU T4 x1**
   - **Internet → On** (needed to `pip install airllm` and clone the repo).
3. **HF token (OPTIONAL).** `Qwen/Qwen2.5-14B-Instruct` is **ungated**, so no token
   is required. If you want to avoid download rate limits, add `HF_TOKEN` under
   **Add-ons → Secrets** (never hard-code it).
4. **Run.** `Run All` (or `Save & Run All (Commit)` for background execution — it
   keeps running after you close the tab). AirLLM is slow (~1–3 tok/s) and the
   14B shard step takes ~10–20 min and ~56 GB of temporary disk, so a full run is
   tens of minutes to a couple of hours.
5. **Where the shards land.** `layer_shards_saving_path = /kaggle/temp/shards`
   (config) — the **scratch** disk, NOT `/kaggle/working` (whose ~20 GB cap is on the
   *saved* output). The notebook **clears the shards between scenarios** and keeps the
   ~29 GB HF download once (`HF_HOME=/kaggle/temp/hf`, `delete_original=False`), so
   peak disk stays bounded. The ledger is written **directly** to
   `/kaggle/working/results/*.json` as each stage completes (robust to a mid-run crash).
6. **Outputs.** The committed notebook's rendered cells (the OOM traceback, the
   AirLLM output, the summary table) are the "screenshots"; `results/*.json` are the
   measured ledger.

### Automated push (what this repo uses)

```bash
uv run kaggle kernels push -p experiments/        # push the notebook + metadata
uv run kaggle kernels status abdallahkhaldi07/cosmos77-ex05-airllm-benchmark   # poll until complete
uv run kaggle kernels output abdallahkhaldi07/cosmos77-ex05-airllm-benchmark -p ./kaggle_out
# then copy kaggle_out/**/results/*.json -> results/, and the executed .ipynb -> experiments/
```

## B. Colab fallback (T4 not guaranteed)

`Runtime → Change runtime type → T4 GPU`. Upload the notebook, then run the cells
top to bottom. Shards go to `/content/shards`. Colab sessions time out and the GPU
is not guaranteed; Kaggle is preferred for the background, persistent run.

## Disk-overflow avoidance (important)

On Kaggle, `/kaggle/temp` and `/kaggle/working` are **one shared ephemeral overlay
disk** (~58–73 GB total, varies); the 20 GB on `/kaggle/working` is a *saved-output*
cap, not a separate physical disk. A 14B model is ~29 GB downloaded + ~29 GB of
per-layer FP16 shards, so the FP16 AirLLM run peaks near ~58 GB on that single pool.

The notebook keeps this bounded: the HF download stays once in `/kaggle/temp/hf`
(`delete_original=False`), shards go to `/kaggle/temp/shards` and are **cleared
between every scenario** (so the FP16/8bit/4bit shard sets never accumulate), and
only the small `results/*.json` are saved to `/kaggle/working`. Cell 1 prints the
free disk and warns below 55 GB. If a stage still dies with *"No space left"*, switch
`config/setup.json` `model_id` to `Qwen/Qwen2.5-7B-Instruct` and say so in the report
— a smaller clean experiment beats a stuck one (honesty, D15).
