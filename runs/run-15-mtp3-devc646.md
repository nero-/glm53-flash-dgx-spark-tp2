# Run 15 — MTP3 on cluster 2 (`devc646`): PR #646 geometry + b12x as prefill KDA

Date: 2026-09-05 · Image `local/vllm:glm53-flash-nvfp4-devc646` (24.9 GB, id `4b9f8e5622ac…`) ·
built on r2 from upstream `blackwell-llm-docker` main + `dgx-spark-builder`, vLLM =
**dev/jovian-judgement a8c796f3 + PR #646 tip b4f16bd3** (frozen as nero-/vllm branch
`osd2/spark-glm53-dev-c646`), b12x = `osd2/spark-glm53-master-20260905` (b58f34ea),
LMCache = builder-default `release/v0.5.2-glm52-dcp-base` (0.5.2+glm52dcp.3, dormant under
CACHE_MODE=vram). KDA prefill = **b12x passed explicitly** (`--kda-prefill-backend b12x`;
R25's flashkda is not buildable by the Spark wrapper and was left out).

Head/API = r2 (LAN 192.168.50.90, fabric 10.100.120.2), worker = r3 (rank1). This is the
SECOND cluster standing up (cluster-1 = r0/r1 remains on r22b); boots and machine memory
are independent of the agent runtime pair (r0/r1 are NEVER touched by the agent).

## Config (rank-{2,3}-mtp3.env)
MTP adaptive 1/3/32 · batch 8192 · ctx 524,288 · `--block-size 2048` (launcher BLOCK_SIZE)
· `--prefix-match-unit 256` (PREFIX_MATCH_UNIT) · split `VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=2048`
/ `SPLIT_MAMMA…MAMBA_BLOCK_SIZE=256` · `--kda-prefill-backend b12x` · MTP MoE mode Marlin ·
MM images 4 + video 1 (LMO 0) · **KV pin 8,589,934,592 (8 GiB)** — byte-exact pin, so
vLLM skips profiling and `gpu_memory_utilization=0.85` is effectively ignored while the
pin is set.

## Boot behavior (the pin is now mandatory)
- Unpinned calibration boot: split-pages warning live
  (`target block size 2048 tokens, recurrent-state block size 256`), physical page sizes
  **[1,148,928, 2,244,608] B** (PR #646 numbers), model load **89.13 GiB** per rank
  (InstantTensor ~3 GB/s), graphs 1.4–2.2 GiB, then
  **`Available KV cache memory: 0.24 GiB` (r2) / 0.76 GiB (r3)** at util 0.85 and the
  engine **refuses to start**:
  `ValueError: To serve at least one request with the model's max seq len (524288), (3.75 GiB
  KV cache…)`. This r-line adds `_check_enough_kv_cache_memory`; unpinned boots on this
  reserve profile no longer fit a 512k sequence. Byte-pin is the only sane path (as on r22).
- Pin 6.5 GiB: healthy, **908,526 tokens** (1.73× @512k), steady used ~118 GiB (r2).
- **Pin 8 GiB (FINAL for mtp3): pool = 1,118,600 tokens, max concurrency 2.13× @512k,
  steady 119 used / 2 avail (r2), 114/7 (r3).** During the full matrix r2 peaked at
  120 used / 1 avail and survived 20 cells + coding probes with no OOM. **Do not pin
  above 8 GiB on this profile** (r22 lesson: 9 GiB + vision is the cliff; here 8 GiB
  + video measured stable).
- Boot latency: first-ever boot ~25 min (JIT + cold page cache); subsequent boots ~4–5 min.
  Markers verified: `B12xMxfp8LinearKernel`, NvFp4 MoE `B12X` target + `MARLIN` MTP experts,
  PYNCCL tp/ep, split-pages 2048/256, page sizes, `GPU KV cache size`, health 200.

## Decode matrix (aggregate tok/s; contexts 0/8k/32k/64k/128k)

| ctx \ conc | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 0 | 31.4 | 47.8 | 63.7 | **101.1** |
| 8k | 33.5 | 46.5 | 71.6 | 91.7 |
| 32k | 31.4 | 50.9 | 59.9 | 96.7 |
| 64k | 31.3 | 41.8 | 57.5 | 60.1 |
| 128k | 32.5 | 36.9 | 45.5 | 82.7 |

MTP-normalized steps/s (accept len): C1 13.3–13.6 (2.31–2.47) · C2 19.0–20.4 ·
C4 23.5–27.3 · C8 37.5–42.6 (accept 1.93–2.62; the 64k C8 cell ran at accept 1.56 —
its 60.1 tok/s cell is an outlier vs its 38.6-step cousin cells; all other long-ctx
cells hold 2.0–2.5).

## Prefill (integrated scouts)

| ctx | TTFT s | tok/s |
|---|---|---|
| 8k | 4.14 | 1,978 |
| 32k | 16.03 | 2,017 |
| 64k | 31.49 | 2,049 |
| 128k | 63.74 | 2,022 |

**Prefill is fully restored to the direct-cable era (1.95–2.05k tok/s), +90% over the
r22b-through-switch reference (1,037–1,080)** — the cluster-2 fabric (108.9 Gb/s RDMA,
run measured during bring-up) moves TP2 all-reduce at line rate again. TTFT at 128k
(63.7 s) matches the ~2k tok/s expectation.

## Coding peak
**34.2 tok/s median (3/3; max 34.6)** vs r22 36.7 / r22b-switch 36.1 → **−5.5% (≈ −2.2
tok/s)** on this line. Decode-only, C1-bound — the this-line's own cost (one-exchange
top-k + b12x sparse-MLA cache fix land in the attention path) is the likely source;
not a fabric artifact (decode latency path is fabric-independent).

## vs baselines
| | r22 (run-10, direct cable) | r22b (run-12, switch) | **this (run-15, cluster-2)** |
|---|---|---|---|
| KV pool | 961,194 @ 7 GiB (run-10 mtp3) / 1,235,254 @ 9 GiB (run-12) | 1,235,254 @ 9 GiB | **1,118,600 @ 8 GiB (139.8k tok/GiB vs 137.3 = +1.9% density; −9% absolute vs the 9-GiB ref because the pin is 1 GiB smaller)** |
| Prefill | ~1,950–2,000 | 1,037–1,080 (throttled) | **1,978–2,049 (restored)** |
| C1 | ~28–30 | 25.9–30 | **31.3–33.5 (slightly better)** |
| C8 ctx0 | 103.8 | 96.4 | **101.1 (par)** |
| C8 long-ctx | 128.5–131 | 88.9–94.1 | **60–97 (mixed: 8k/32k/128k ≈ r22b, 64k cell low)** |
| Coding peak | 36.7 | 36.1 | 34.2 (−5.5%) |
| 512k concurrency | ~1.9× | same | **2.13× @ the 8 GiB pin** |

C8 long-ctx: the r22-direct 125–131 numbers were flagged in run-12 as "possibly
inflated (b12x contract pin predates the locklock fix)". Against the honest r22b
reference (88.9–96.4), this line lands at 82.7–101.1 (ctx0/8k/32k/128k) — **par or
better everywhere except the C8-64k cell (60.1, accept-len collapse 1.56; treat as a
single-cell oddity until re-measured).

## Verdict
- Cluster 2 serves devc646 MTP3 correctly: exact 1→10 smoke, reasoning folded, no
  `locklock`, PYNCCL-clean, fabric at line rate.
- **KV density +1.9%/GiB (139.8k vs 137.3k tokens/GiB)** and 2.13× 512k concurrency
  at the 8-GiB pin — PR #646's 2048/256 split pages + one-exchange top-k; the pool
  is the capacity goal delivered at the video-qualified memory envelope.
- Prefill restored to ~2k tok/s on the new pair.
- C8 long-ctx holds 83–97 (r22b-par) with one low 64k cell; C1/C2 up ~5–15%; coding
  peak −5.5% — the only regression, hip-pocket for the next r-line.

Files: `runs/run-15-mtp3-devc646.json`. Log path on r2:
