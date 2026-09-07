# Run 02 — mtp3 @ 1M KV (kyonjova r17-roce, piecewise graphs)

Date: 2026-09-03 · Image: `local/vllm:glm53-flash-nvfp4-R17-roce` · Profile:
`rank-0-1m.env` / `rank-1-1m.env` · Launcher: `glm53_pair_serve.sh` +
`pairctl.sh up 1m`.

## Goal
The 1M-KV milestone on the kyonjova recipe: pin the KV pool, serve at
`MAX_MODEL_LEN=1048576`, and run the full apples-to-apples bench (the same
cells as the kyonjova R12 table).

## Config (deltas vs run 01)
`KV_CACHE_MEMORY_BYTES=10737418240` (10 GiB pin) · `MAX_MODEL_LEN=1048576` ·
**`CUDAGRAPH_MODE=FULL_AND_PIECEWISE`** (their R12 measured config; run 01's
`FULL` auto-downgraded to `FULL_DECODE_ONLY` = eager prefill). Everything
else unchanged: mtp@3 adaptive (1/3/32), vision 4/1, batch 4096, seqs 8,
capture 96, fp8 KV, instanttensor, B12X + HUMMING, autotune OFF.

## Boot
- Healthy in **225 s**; piecewise graphs captured in 38 s.
- **GPU KV cache size: 1,474,826 tokens** (10 GiB pin → 7.1 KiB/token;
  profiled mode had shown 9.6 KiB/token — pinned mode is much denser).
  = **1.41× concurrency for a full 1M-token request**. Target 1M ✓ (by 47%).
- Memory while serving: r0 ~120 GiB / r1 ~117 GiB used — same envelope as
  the R12 pair. No OOM across the whole bench ladder (128k prefills +
  65k-token coding runs), including the cells that killed the old
  self-composed image.

## Bench (full: C1-C8, ctx 0/8k/32k/64k, prefill 8k-128k, coding peak)
`llm_decode_bench.py --concurrency 1,2,3,4,8 --contexts 0,8192,32768,65536
--duration 30 --max-tokens 2048 --prefill-contexts 8k,32k,64k,128k
--coding-peak --coding-peak-runs 3 --coding-peak-max-tokens 65536
--kv-budget 1474826`

### Prefill tok/s (TTFT)
| ctx | TTFT s | tok/s | R12 ref |
|---|---|---|---|
| 8k | 4.53 | **1,810** | 1,930 |
| 32k | 19.24 | 1,679 | 1,896 |
| 64k | 38.13 | 1,691 | 1,890 |
| 128k | 77.88 | 1,654 | 1,813 |

Piecewise recovered +28% over run 01's eager prefill (1,408 → 1,810 @ 8k).
Remaining gap to R12 ≈ their batch 4096→8192 delta (+15–25% per their notes)
→ Run 04.

### Aggregate decode tok/s
| ctx \ conc | 1 | 2 | 3 | 4 | 8 |
|---|---|---|---|---|---|
| 0 | 28.0 | 44.6 | 52.0 | 61.3 | **101.1** |
| 8k | 22.2 | 42.3 | 48.6 | 49.5 | 96.6 |
| 32k | 28.0 | 46.8 | 45.2 | 50.3 | 88.2 |
| 64k | 29.6 | 37.9 | 47.0 | 50.1 | 89.2 |

C8 aggregate 101.1 @ ctx 0 (R12: 104.8) and still 88-97 at up to 64k
context — decode holds up under long contexts at the 1M geometry.

### MTP steps/s (accept len)
C8: 41.2 (2.45) @ ctx 0 · 39.4 (2.2x) @ 64k. C1: 12.4-13.2.

### Coding peak
| runs | median | mean | max |
|---|---|---|---|
| 3/3 | **37.0** | 35.9 | 37.8 |

Beats the R12 mtp3 reference (36.3). (The 40-50 coding-peak target is the
dflash2 mode — Run 03.)

## Takeaway / next
- **1M KV ✓ (1,474,826 tokens)**, stable across the full bench, and decode +
  coding peak match/beat the R12 mtp3 reference on the first try.
- Prefill at batch 4096 = 1,654-1,810. The ~2k PP target needs the 8192
  batch profile (their measured +15-25%) → Run 04.
- Run 03 (dflash2) next: TG/coding-peak mode, 12 GiB pin, split pages.
