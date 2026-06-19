# PRD — Hardware Spec Capture + Param→Memory Math (D1)

> **Mechanism owner:** `src/cosmos77_ex05/hardware/`
> **Maps to acceptance criterion:** **D1** — *Hardware spec captured + documented, with a quantitative model-choice justification (params × bytes/param math showing it is too big for the hardware).*
> **Phase:** 2 (shared infra). **Status:** specified, not yet implemented.
> **Binding rules:** CLAUDE.md §1 (17 rules). 150-line cap/file; single `class SDK` entry; `uv` only; TDD with ALL psutil/torch/shutil I/O mocked; coverage ≥ 85%; config-driven; every result → `results/*.json` ledger; English + lecture vocabulary.

---

## 1. Purpose

Before we can claim Qwen2.5-14B "doesn't fit," we must **measure the box** and **do the arithmetic**. This mechanism delivers both halves of D1:

1. **Hardware spec capture** — a deterministic probe of CPU, RAM, GPU + VRAM, and disk, written to `results/hardware.json`. It must run identically (no crash) on **two very different hosts**: the student's CUDA-less Mac (development) and a Kaggle **T4** (the experiment env). On the Mac it reports `"none / CPU-only"` for the GPU; on the T4 it reports the full `Tesla T4` / 16 GB VRAM spec.
2. **Param→memory math** — the *quantitative model-choice justification*. Given a parameter count and a precision (`fp16` / `q8` / `q4`), compute the model's footprint in GB and emit the **OOM verdict** against the host VRAM. This is the number that proves, on paper, why we need AirLLM + quantization at all: it is the gate that makes the rest of HW5 necessary.

The captured spec also feeds the **economic CAPEX** section (D7) — the on-prem cost story needs the real Mac specs — and the **Roofline** section (D9) needs the T4's VRAM/BW figures. So this module is upstream of half the report.

---

## 2. Inputs / Outputs

### Inputs
| Source | Used for |
|---|---|
| `platform` module (stdlib) | CPU model string, OS/arch |
| `psutil.cpu_count(logical=…)` | physical + logical core counts |
| `psutil.virtual_memory()` | total / available RAM |
| `torch.cuda.is_available()` + `torch.cuda.get_device_properties(0)` | GPU name + total VRAM (only when CUDA present) |
| `shutil.disk_usage(path)` | disk total / used / free |
| `config/setup.json` → `experiment.model_id`, `paths.results_dir` | where to write; which model the math justifies |
| Caller-supplied `params` (e.g. `14.7e9`) + `vram_gb` (e.g. `16`) | the param→memory math inputs |
| `BYTES_PER_PARAM` (from `constants.py`) | `{"fp16": 2.0, "q8": 1.0, "q4": 0.5}` |

### Outputs
- **`results/hardware.json`** — a single artifact (the D1 ledger entry), e.g.:
  ```json
  {
    "captured_at": "2026-06-19T10:00:00Z",
    "cpu": {"model": "Apple M2", "physical_cores": 8, "logical_cores": 8},
    "ram": {"total_gb": 16.0, "available_gb": 9.2},
    "gpu": {"name": "none / CPU-only", "vram_gb": 0.0, "cuda_available": false},
    "disk": {"total_gb": 460.4, "free_gb": 120.1},
    "platform": {"system": "Darwin", "release": "25.3.0", "machine": "arm64"}
  }
  ```
  On Kaggle the same shape yields `gpu.name = "Tesla T4"`, `gpu.vram_gb = 16.0`, `cuda_available = true`.
- **Return value of `capture_hardware()`** — a dict containing both the spec **and** the `justify(...)` math block (so the notebook/README can render the OOM table without re-reading the file).

---

## 3. Module Design (files ≤ 150 lines, how to split)

```
src/cosmos77_ex05/hardware/
├── __init__.py        # re-export capture_spec, model_memory, justify
├── spec.py            # ≤120 lines — probe CPU/RAM/GPU/disk → results/hardware.json
└── model_math.py      # ≤80 lines  — param→bytes math + OOM verdict
```

**Why this split.** Two distinct responsibilities, two failure modes: `spec.py` does *impure* hardware probing (mockable I/O against psutil/torch/shutil); `model_math.py` is *pure arithmetic* (no I/O at all → trivially testable, no mocks). Keeping them apart honours rule 3 (no duplication, single responsibility) and keeps each well under the 150-line cap.

### `spec.py` (≤ 120 lines)
Public functions:
- `capture_spec(results_dir: str | Path) -> dict` — orchestrates the four probes, builds the dict above, writes `results/hardware.json`, returns the dict.
- Private helpers `_cpu()`, `_ram()`, `_gpu()`, `_disk()` keep `capture_spec` short and each probe independently mockable.

**GPU graceful-degradation contract (the load-bearing requirement):**
```python
def _gpu() -> dict:
    if not torch.cuda.is_available():
        return {"name": "none / CPU-only", "vram_gb": 0.0, "cuda_available": False}
    props = torch.cuda.get_device_properties(0)
    return {"name": props.name,
            "vram_gb": round(props.total_memory / 1024**3, 1),
            "cuda_available": True}
```
`torch` is an **optional/experiment** dependency. `spec.py` must `import torch` defensively (try/except `ImportError`) so the Mac — where torch may be absent or CPU-only — still produces a valid `"none / CPU-only"` record rather than crashing. **No real hardware assumption may break the test suite or the Mac run.**

### `model_math.py` (≤ 80 lines)
Pure, no I/O:
- `model_memory(params: float, dtype: str) -> float` — `round(params * BYTES_PER_PARAM[dtype] / 1024**3, 1)` → GB. Raises `KeyError`/`ValueError` on an unknown dtype.
- `justify(model_id: str, params: float, vram_gb: float) -> dict` — returns the full param→memory math block **and** the OOM verdict at each precision:
  ```json
  {
    "model_id": "Qwen/Qwen2.5-14B-Instruct",
    "params": 14.7e9,
    "vram_gb": 16.0,
    "footprint_gb": {"fp16": 29.4, "q8": 14.7, "q4": 7.4},
    "fits": {"fp16": false, "q8": true, "q4": true},
    "verdict": "FP16 OOMs (29.4 GB > 16 GB VRAM); Q4 fits (7.4 GB ≤ 16 GB)"
  }
  ```

**Worked example (the canonical D1 number):**
`14.7e9 params × 2 bytes (FP16) = 29.4 GB > 16 GB VRAM ⇒ OOM`; at Q4, `14.7e9 × 0.5 = 7.4 GB ≤ 16 GB ⇒ fits`; Q8 sits at `14.7 GB`, just under the line (the "barely fits, no headroom for KV-cache" boundary case the report should flag).

---

## 4. Public SDK API

All business logic is reached through the single `class SDK` (rule 2). This mechanism contributes one method:

```python
class SDK:
    def capture_hardware(self) -> dict:
        """Probe the host, write results/hardware.json, and return the spec
        plus the param→memory justification (D1).

        Reads model_id + results_dir from config/setup.json. On a CUDA-less
        host (the Mac) the gpu block reports "none / CPU-only"; on a Kaggle
        T4 it reports the full Tesla T4 / 16 GB spec. The returned dict embeds
        justify(model_id, params, vram_gb) so callers render the OOM table
        without re-reading the file.
        """
```

Behaviour:
1. Load `model_id`, `results_dir` (and the assumed `params` for the model) from config — nothing hardcoded (rule 4).
2. `spec = capture_spec(results_dir)` → writes the file.
3. `math = justify(model_id, params, vram_gb=spec["gpu"]["vram_gb"] or T4_REFERENCE_VRAM)` — on the CPU-only Mac, `vram_gb` is 0, so the math falls back to the **configured T4 reference VRAM (16 GB)** so the justification is still meaningful off-GPU. This fallback is config-driven and documented.
4. Return `{**spec, "model_math": math}` (also surfaced via the `cosmos77-airllm hardware` CLI stage).

The CLI verification (`uv run cosmos77-airllm hardware`) must print the Mac spec and emit `results/hardware.json` with no GPU and no crash.

---

## 5. Test Plan (everything mocked — no real hardware)

All tests live in `tests/unit/hardware/`. **psutil, torch, and shutil are mocked** via `pytest-mock` / `monkeypatch`; the suite never probes real hardware, never imports a real GPU, and is deterministic (rule 17). Two synthetic hosts drive `spec.py`:

| Test | Mocks | Asserts |
|---|---|---|
| `test_spec_fake_cuda_box` | `torch.cuda.is_available()→True`; `get_device_properties(0)` returns a fake obj `name="Tesla T4", total_memory=16*1024**3`; psutil/shutil canned | gpu block = `{"name": "Tesla T4", "vram_gb": 16.0, "cuda_available": True}`; full dict shape correct |
| `test_spec_cpu_only_box` | `torch.cuda.is_available()→False` (and a variant with `torch` absent → `ImportError`) | gpu block = `{"name": "none / CPU-only", "vram_gb": 0.0, "cuda_available": False}`; **no exception** |
| `test_spec_writes_json` | all probes mocked; `tmp_path` as results_dir | `results/hardware.json` exists, parses, round-trips the returned dict |
| `test_cpu_ram_disk_shapes` | psutil/shutil mocked with known values | physical vs logical cores distinct; RAM/disk GB rounded correctly from bytes |
| `test_model_memory_fp16` | none (pure) | `model_memory(14.7e9, "fp16") == 29.4` |
| `test_model_memory_all_dtypes` | none | q8 → `14.7`, q4 → `7.4`; unknown dtype raises |
| `test_justify_oom_verdict` | none | `justify("Qwen/Qwen2.5-14B-Instruct", 14.7e9, 16.0)`: `fits["fp16"] is False`, `fits["q4"] is True`, `footprint_gb["fp16"] == 29.4`, verdict mentions "OOM" and "16" |
| `test_justify_boundary_q8` | none | Q8 = 14.7 GB flagged as fits-but-no-headroom (≤ 16 yet leaves < 2 GB) |
| `test_sdk_capture_hardware_cpu_only` | spec + config mocked | returns `{...spec, "model_math": {...}}`; CPU-only fallback uses the configured T4 reference VRAM for the math |

**Coverage target:** ≥ 85 % on `spec.py` + `model_math.py`. Because `model_math.py` is pure and `spec.py`'s I/O is fully mocked, 100 % line coverage is realistic.

---

## 6. Acceptance-Criteria Mapping

| Requirement (D1) | Satisfied by | Evidence |
|---|---|---|
| CPU + physical/logical cores captured | `spec._cpu()` via `platform` + `psutil.cpu_count` | `cpu` block in `results/hardware.json` |
| RAM captured | `spec._ram()` via `psutil.virtual_memory` | `ram` block |
| GPU + VRAM captured (or "none") | `spec._gpu()` via `torch.cuda` guard | `gpu` block; `"none / CPU-only"` on Mac, `Tesla T4`/16 GB on Kaggle |
| Disk total/free captured | `spec._disk()` via `shutil.disk_usage` | `disk` block |
| **Quantitative model-choice justification** | `model_math.justify` | `model_math` block: 29.4 GB FP16 > 16 GB ⇒ OOM; 7.4 GB Q4 ⇒ fits |
| Documented artifact | `results/hardware.json` (the ledger entry) | committed file; rendered in README §3 |
| Runs on both hosts without crashing | defensive torch import + CUDA guard | `test_spec_cpu_only_box` + Kaggle live run |

Downstream consumers: README §3 (model-choice justification), D7 economics (CAPEX uses real Mac spec), D9 Roofline (T4 VRAM/BW).

---

## 7. Risks & Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| `torch` absent or CPU-only on the Mac | `import torch` crashes the probe | defensive `try/except ImportError`; CUDA guard returns `"none / CPU-only"`; covered by `test_spec_cpu_only_box` |
| Parameter count is approximate (14.7B is a rounded figure) | math could look "wrong" to a grader | document `params` as a config value with its source; the *method* (params × bytes) is what D1 grades, and 29.4 GB > 16 GB holds for any count ≥ 8B at FP16 |
| CPU-only host has `vram_gb = 0`, breaking the OOM math | division/verdict meaningless off-GPU | config-driven **T4 reference VRAM (16 GB)** fallback so the justification renders even on the Mac; documented in §4 |
| Byte/GiB confusion (`1000³` vs `1024³`) | footprint off by ~7 % | fix the convention: `/ 1024**3` (GiB) everywhere, stated in docstrings; tests pin exact expected values |
| `get_device_properties` API drift across torch versions | spec probe breaks on Kaggle | only touch the stable `.name` / `.total_memory` fields; mock the same shape in tests; pin torch in the `experiment` extra |
| File over 150 lines as probes grow | rule 1 violation | helpers already split; if `spec.py` nears the cap, extract `_gpu`/`_disk` into a tiny `probes.py` |
| Fabricated/hand-edited numbers | rule 13 / D15 violation | every number flows from `capture_spec` → `results/hardware.json` → README; the ledger is the single source of truth, never hand-edited |

---

## 8. Definition of Done

- [ ] `spec.py` ≤ 120 lines, `model_math.py` ≤ 80 lines; `ruff check` clean; line-cap check passes.
- [ ] `capture_hardware()` writes `results/hardware.json` and returns spec + math.
- [ ] Mac run reports `"none / CPU-only"`; (Kaggle) T4 run reports `Tesla T4` / 16 GB — both without crashing.
- [ ] `model_memory(14.7e9, "fp16") == 29.4`; `justify(...)` yields the correct OOM verdict vs 16 GB.
- [ ] All psutil/torch/shutil mocked; suite green CPU-only; coverage ≥ 85 % on this module.
- [ ] Docstrings + type hints on every public signature; values read from `config/setup.json` (nothing hardcoded).
