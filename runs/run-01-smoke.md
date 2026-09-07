# Run 01 — smoke: kyonjova r17-roce image, mtp3 qualified boot

Date: 2026-09-03 · Image: `local/vllm:glm53-flash-nvfp4-R17-roce` (built 1h10m,
status OK+VERIFIED, peak 103 GiB) · Launcher: `glm53_pair_serve.sh` (kyonjova)
· Profile: `rank-0.env` / `rank-1.env` (smoke: 262144 ctx, unpinned @ 0.85).

## Goal
First boot of the kyonjova-qualified r17 composition on the pair: prove the
image + launcher + fabric, get profiled KV geometry, sanity-check throughput
against the R12 reference table, and calibrate the 1M-KV pin.

## Config (from env files)
mtp@3 adaptive (1/3/32, humming MoE + B12X attention) · vision 4 img/1 video ·
batch 4096 · seqs 8 · capture 96 · `CUDAGRAPH_MODE=FULL` ·
`PREFILL_SCHEDULE_INTERVAL=8` · `LOAD_FORMAT=instanttensor` (BUFFERED, staged
bounds) · gmu 0.85 unpinned · `KV_CACHE_DTYPE=fp8` · B12X attention/MoE/linear
· FlashInfer autotune OFF · block 256 + align mamba · patched NCCL via
`LD_PRELOAD` + CUDA 13.2 compat shim · `VLLM_ENABLE_PCIE_ALLREDUCE=0`.

## Boot
- Health in **308 s** (worker first, head second via `pairctl.sh up smoke`).
- Backends: `PYNCCL` all-reduce (RoCEnante present but dormant ✓),
  `B12xMxfp8LinearKernel`, `B12X` NvFp4 MoE (+ `HUMMING` for the MTP head ✓).
- Geometry: attention block size **4,608 tokens**, 2 padding layers (5.88%
  waste) — matches our earlier r17 measurement.
- **`CUDAGRAPH_MODE=FULL` auto-downgraded to `FULL_DECODE_ONLY`** (GDN
  backend) → prefill runs EAGER. Graph capture 16 s (0.04 GiB) + 20 s
  (1.26 GiB).
- Profiled pool: **503,316 tokens in 4.69 GiB = 9.6 KiB/token**
  (1.92× concurrency at 262k). ⇒ 10 GiB pin ≈ **1.07M tokens** — the 1M
  target fits.
- Memory while serving: r0 111 GiB / r1 107 GiB used → +5.3 GiB pin headroom ✓.
- Text probe OK (reasoning + chat), vision probe answered the red image:
  **"Red"** ✓.

## Bench (compact sanity: C1/C8, ctx 0/8k/32k, prefill 8k/32k)
`llm_decode_bench.py --concurrency 1,8 --contexts 0,8192,32768 --duration 20
--max-tokens 512 --prefill-contexts 8k,32k --kv-budget 503316`

| Metric | Smoke (FULL→decode-only) | R12 reference (piecewise) |
|---|---|---|
| Prefill 8k | 1,408 tok/s | 1,930 |
| Prefill 32k | 1,673 tok/s | 1,896 |
| Decode CC1 (ctx 0) | **30.7** | 29.4 |
| Decode C8 aggregate (ctx 0) | 100.0 | 104.8 |
| Decode C1 @ 8k / 32k | 21.8 / 24.2 | — |
| MTP steps/s C1 (accept) | 11.7 (2.61) | 12.06 (2.44) |

Decode is on par with (slightly above) the R12 reference. Prefill gap is the
expected eager-mode cost of `FULL` on the GDN backend.

## Takeaway / next
- The kyonjova r17-roce image is healthy on the pair: backends, NCCL, vision,
  MTP, geometry all match the qualified recipe. `pairctl.sh up` worked
  end-to-end on the first try.
- Run 02 (1M KV): pin **10 GiB** (`10737418240`), `MAX_MODEL_LEN=1048576`,
  `CUDAGRAPH_MODE=FULL_AND_PIECEWISE` (their R12 measured config — expect
  prefill to recover to ~1,900+ and compare the two graph modes A/B).
