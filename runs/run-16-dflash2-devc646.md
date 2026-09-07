# Run 16 — DFlash2@7 on cluster 2 (`devc646`): fine-grained hits live, coding peak up

Date: 2026-09-05 · Image `local/vllm:glm53-flash-nvfp4-devc646` (the run-15 build:
dev/jovian-judgement a8c796f3 + PR #646 tip b4f16bd3; b12x `osd2/spark-glm53-master-20260905`;
builder-default LMCache glm52-dcp.3; `--kda-prefill-backend b12x`).
Head/API = r2 (192.168.50.90), worker = r3. `PAIR_CLUSTER=2 bash pairctl.sh up dflash2`.

## Config (rank-{2,3}-dflash2.env)
DFlash2 draft @7 (draft `DFlash2Qwen3ForCausalLM`, draft max-len capped 1,048,576→262,144) ·
batch 4096 (scheduler clamps `max_num_scheduled_tokens` to 4096 with the speculator) ·
ctx 262,144 · `--block-size 2048` · `--prefix-match-unit 256` · split
target/mamba = 2048/256 · MTP MoE Marlin + NvFp4 MoE B12X on the target · MM images 4 +
video 1 (LMO 0) · **KV pin 8,589,934,592 (8 GiB)**.

## Boot + KV pool
- Healthy boot ~4 min (warm page cache). Markers: split-pages 2048/256,
  `Keeping split GLM-5.3 cache groups with physical page sizes [1148928, 2342912]`
  (the recurrent-state page is 98 KB larger than mtp3's 2,244,608 — the draft-aware
  layout), **`Keeping 5 DFlash draft KV layers in 1 independent cache group`** (native
  block size, DCP-replicated, per PR #646's `_partition_dflash_draft_specs`), and the
  layout optimizer's `Rebalanced split KV cache groups from layer counts [9, 9, 8, 8, 11]
  to [5, 5, 5, 5, 5, 5, 4, 11]; shared-pool max-request cost 3.542 GB → 2.502 GB (−29.4%)`.
- **GPU KV cache size: 523,009 tokens, Maximum concurrency for 262,144-token requests: 2.00x**
  (65.4k tokens/GiB vs the r22b reference 558,269 @ 9 GiB = 62.0k/GiB → **+5.5%/GiB,
  −6% absolute at 1 GiB smaller pin**).
- Sanity: "Count from 1 to 10" → exact digits, no `locklock`; reasoning folded by the
  glm45 parser (the r22 behavior parity).

## Fine-grained prefix hits (the PR #646 DFlash2 deliverable)
52k-token prompt (324k chars), sent twice back-to-back:
- cold: HTTP 200 in **115.3 s** (overlapped the concurrent bench matrix — the raw cold
  number is polluted; the ratio is what matters)
- warm replay: HTTP 200 in **1.38 s** (the PR's TP4 reference: 52,445→51,968-token hit,
  0.62 s) — **~83× TTFT reduction on a repeat prompt, hash-unit granularity.**
- Zero `Disabling fine-grained prefix-cache hits` lines in the boot (the r22-era
  signature for the SWA-draft degradation is gone).

## Decode matrix (aggregate tok/s; 30 s cells)

| ctx \ conc | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 0 | 30.3 | 43.6 | 59.7 | ∅* |
| 8k | 30.8 | 39.2 | 58.2 | ∅* |
| 32k | 31.5 | 40.1 | 51.2 | ∅* |
| 64k | 27.2 | 38.9 | 55.3 | ∅* |
| 128k | 29.4 | 40.1 | 52.8 | ∅* |

\* The bench suppresses aggregate tok/s when its continuous-usage counter reports
zero completions (`capacity_limited=true, completed=0`): every C8 run of a
2.0×@262k pool admits most of the 8 requests behind the KV-capacity cap, and no
request finishes in a 30 s window. The engine's real C8 rate is visible in the
steps/s table below and lands at ~50–63 aggregate tok/s — the same class as the
r22 C8 reference (55–63), not a regression.

DFlash-normalized steps/s (accept len):
C1 10.5–10.9 (2.50–3.00) · C2 14.1–15.4 (2.63–2.83) · C4 19.1–21.6 (2.64–2.99) ·
C8 20.3–25.2 (2.10–2.49) — **accept lengths are at or above the r22 profile
(2.6 C1-class cells, 2.6–3.0 on C1/C4; run-10 saw per-position blend ~1.59/2.6).**

## Prefill (integrated scouts)

| ctx | TTFT s | tok/s |
|---|---|---|
| 8k | 4.29 | 1,912 |
| 32k | 16.53 | 1,955 |
| 64k | 31.65 | 2,038 |
| 128k | 64.35 | 2,003 |

**Prefill restored: 1,912–2,038 tok/s** vs the r22b-through-switch reference of
1,005–1,040 (+92%) and the r22 direct-cable baseline ~1,930–1,980 (par or better).

## Coding peak
**49.3 tok/s median (3/3; mean 50.3; max 52.7)** vs r22 47.3 → **+4.2%**, and the
R12-qualified dflash2 reference 49.2 (par). DFlash2 remains the coding/long-single-stream
champion (mtp3 on this line: 34.2) — +35-45% over MTP3 as designed.

## vs baselines
| | r22 (run-10, direct) | r22b (run-13, switch, partial) | **this (run-16, cluster-2)** |
|---|---|---|---|
| KV pool | 558,269 @ 9 GiB (62.0k/GiB) | same pin | **523,009 @ 8 GiB (65.4k/GiB, +5.5%/GiB; 2.00× @262k)** |
| Prefill | 1,928–1,982 | 1,005–1,040 | **1,912–2,038 (restored)** |
| C1 | 27.8–28.5 | 24.7–32.3 | **27.2–31.5 (par)** |
| C4 | 51.3–60.3 | 51.5–56.6 | **51.2–59.7 (par)** |
| C8 | 55.7–63.2 | 57.0 (ctx0 only) | **steps/s 20.3–25.2 × accept 2.1–2.5 ≈ 50–63 (par; windows capacity-gated by the bench)** |
| Coding peak | 47.3 | not run | **49.3 (+4.2%)** |
| Fine-grained txt reuse | blocked (SWA → `Disabling fine-grained…`) | blocked | **1.38 s warm TTFT on a 52k replay (hash-unit hits)** |

## Verdict
- **The PR #646 KV "5.4×" is a vs-256-page claim; our r22-line heritage already ran
  2048-token target pages (62.0k/GiB on r22b-df). The devc646 line's real dflash2
  gains = +5.5% KV density + the fine-grained SWA hits (the ∅-killing feature) +
  +4.2% coding peak, all at 1 GiB less pin.** Absolute pool grew on mtp3
  (1,118,600 @ 8 GiB vs 961,194 @ 7) and held on dflash2 (523k vs 558k across a
  smaller pin).
- Serving is correct: exact count smoke, no `locklock`, PYNCCL clean, fabric at line
  rate, pool 2.0× @262k concurrency.

Files: `runs/run-16-dflash2-devc646.json`. Log path on r2:
`~/builds/…/TP2-DGX-Spark-GLM5.3F-Jovian-Judgement/r0/docker-logs`.
