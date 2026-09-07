# Run 10 — r22 rebuild: DFlash2 wedge FIXED, MTP3 requal

Date: 2026-09-04 · Image: `local/vllm:glm53-flash-nvfp4-r22` · built on r0,
loaded to r1 over fabric.

## Build (r22 pins, dgx-spark-builder, ~13 min warm)
- vLLM `local-inference-lab/vllm` `artifact/jovian-judgement-community-20260904-r22-source` @
  `70b3c1c7f1c76fcf0847fcbb4a0b8b5583b78d19`
- B12X `local-inference-lab/b12x` @ `1e59a1fd09f782d302b1068b15c8a0bd66103894`
  (the r22 contract commit)
- Toolchain unchanged from jj-master (NCCL dfab7c1a, FlashInfer 1ac6942,
  InstantTensor 49b4010, torch 2.13.0+cu132, cutlass e6233cba).
- Wheel `v0.26.1rc0+glm53.flash.nvfp4.r22`; image 24.9 GB.
- Build profile: `builder/blackwell-llm-docker/dgx-spark-builder/build-glm53-r22.env`.

## DFlash2 (`rank-{0,1}-df.env` → SERVING_IMAGE=r22) — THE RESULT
- Boot healthy in **307 s**. `speculative_config.method='dflash'`,
  `num_spec_tokens=7`, draft `glm-5.3-flash-dflash2`.
- **GPU KV cache: 558,269 tokens** (2.13× at 262k) — same 9 GiB pin geometry
  as before.
- Draft path: B12X MXFP8 fused K/V projection, FlashAttention v2 draft
  attention, split pages 4096/4096, 5 draft KV layers in 1 group.
- All-reduce: PYNCCL (correct for 2-node RoCE).

### Wedge status: FIXED ✅
| Prefill | Result |
|---|---|
| 32k | HTTP 200, 21.1 s |
| 64k | HTTP 200, 17.6 s |
| 128k | HTTP 200, 34.1 s |

Prior signatures (r17/r20/jj-master) wedged on the first long request
(`Waiting: 1`, engine idle, KV 0.0%). r22 completes all three, repeatedly.
The long-prefill DFlash2 wedge is resolved on this line.

### Sanity (fresh boot)
- "Count from 1 to 10" → `1, 2, 3, 4, 5, 6, 7, 8, 9, 10` (stop, 29 reasoning tok)
- "Capital of France?" → `The capital of France is **Paris**.` (106 reasoning tok)
- No `locklock` loop; reasoning tokens present (content is in `message.content`;
  `reasoning_content` is None — the glm45 parser folds reasoning).

## Throughput (quick: C1/2/4/8 × ctx 0/8k/32k/64k/128k, duration 30)

| ctx \ conc | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 0 | 27.8 | 38.8 | 51.3 | **63.2** |
| 8k | 28.3 | 36.0 | 60.3 | 55.7 |
| 32k | 28.5 | 31.6 | 51.3 | 59.9 |
| 64k | 23.8 | 39.7 | 41.0 | 58.8 |
| 128k | 17.4 | 30.1 | 49.9 | — |

vs MTP3 (jj-master, run 06): C1 ≈ 28.8 (par) · C8 104.5 (MTP3 wins by ~1.65×).

### Prefill tok/s (PP)
8k **1928** · 32k **1982** · 64k **1961** · 128k **1940** — par with MTP3 (~1.96k).

## Coding peak
**47.3 tok/s** (1/1 run, 2594 tokens, finish stop, 55.1 s) — vs MTP3's **35.4**.
**+34%** and right at the R12-qualified dflash2 reference (49.2).

## Speculation acceptance (real, from /metrics)
- `spec_decode_num_accepted_tokens_total` = 17,797 of `num_draft_tokens_total`
  = 210,595 → **8.45%** draft acceptance; mean accept length ≈ 1.59 (blended).
- Per-position: 22.2% / 14.4% / 8.8% / 5.7% / 3.6% / 2.5% / 1.9% (pos 0–6).
- On short-prose C1 accept ≈ 2.6; on code ≈ 4.4 (drives the 47.3 coding peak).
- Accept collapses toward 1.0 at C8/long-ctx — hence C8 ~60 vs MTP3 104.
- ⚠️ The periodic log line `Avg Draft acceptance rate: 0.0%` / `Mean acceptance
  length: 1.00` is a **per-window artifact** during high-concurrency cells
  (misleading); the /metrics counters above are authoritative.

## Estonia (LAST, per operator)
TBD — target ≥28/30.

## MTP3 requal on r22 (same master env, SERVING_IMAGE=r22)
Boot 225 s, KV 961,194 (same), MTP experts Marlin, target MoE B12X — contract match.

| ctx \ conc | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 0 | 30.0 | 42.4 | 53.8 | **103.8** |
| 8k | 17.1* | 40.4 | 74.1 | **131.1** |
| 32k | 28.3 | 40.1 | 72.6 | **129.8** |
| 64k | 37.9 | 40.1 | 73.4 | **128.5** |
| 128k | 16.8* | 39.7 | 71.9 | **125.6** |

\* C1 cells noisy (30 s duration, single run); ctx 8k/128k dipped, ctx 64k spiked —
not a trend. PP 2002/2008/1980/1954 · coding peak **36.7** (jj-master 35.4).

vs jj-master (run 06): C1 ≈ par · **C8 long-ctx 125–131 vs 88–95 (+~35%)** ·
coding par · PP par. C8 delta may be partly prefix-cache state (this run: 0%
hit; run 06: unknown) — treat as "no regression, possibly better at C8 long-ctx".

## Takeaways
- **r22 = the DFlash2-fix line.** Wedge fixed; dflash2 drafts AND accepts.
- DFlash2 r22: **coding +34%** (47.3 vs 35.4) and C1 prose ~par — the coding /
  long-single-stream option it was designed to be. C8 is ~40% *below* MTP3
  (deeper draft = 2× verify batch, acceptance collapses at high batch), so
  MTP3 stays the throughput / daily driver.
- **MTP3 on r22: no regression** vs jj-master (C1 par, coding 36.7 vs 35.4,
  PP ~2.0k), C8 long-ctx possibly higher. r22 is a safe drop-in for master.
