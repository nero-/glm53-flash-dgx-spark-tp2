# Run 11 — decode/context optimization iterations (plateau)

Date: 2026-09-04 · Image: `local/vllm:glm53-flash-nvfp4-r22` · TP2 pair.
Goal: maximize PP / decode / context via env-only iterations; stop at plateau.
Baseline = run-10 master (adaptive MTP3, window 32): C1 ctx0 30.0 · C8
104–131 · coding 36.7 · PP ~2.0k.

## Speculation sweep — adaptive wins; fixed-MTP is a dead end

| variant | C1 ctx0 | C1 ctx64k | C1 ctx128k | C8 ctx0 | C8 ctx64k | coding |
|---|---|---|---|---|---|---|
| master (adaptive, w32) | 30.0 | 37.9* | 16.8 | 103.8* | 128.5 | 36.7 |
| **fx3 (fixed@3)** | 30.8 | **16.3** | 13.7 | 94.7 | 94.0 | 39.0 |

\* master cells are single 30-s runs (noisy).

- Fixed@3 accept length **collapses to 1.00 at long ctx** (draft fully
  rejected) — the MTP draft is pure overhead there. Adaptive's depth-throttle
  is what keeps long-ctx decode fast. **Adaptive (master) is the winner.**
- fx2 (fixed@2) skipped — same failure mode as fx3.

## Memory / context bisect — dev win not isolable to a safe setting

The dev profile showed ~1.2 GiB less memory + KV 1.1M vs master's 961k. Tested
each dev-only setting in isolation (single-variable, same 7 GiB pin):

| variant | setting | free mem (boot) | KV tokens | verdict |
|---|---|---|---|---|
| master | — | 113.52 GiB | 961,194 | baseline |
| mm0 | `MM_IMAGES=1` + `--mm-processor-cache-gb 0` | 113.3 | 961,194 | no change |
| mm0-ds | + `VLLM_SSM_CONV_STATE_LAYOUT=DS` | 113.11 | 961,194 | no change |
| master-v2 | + `VLLM_USE_V2_MODEL_RUNNER=1` | 113.14 | 961,194 | no change |

- DS layout and V2 runner were both confirmed active in the boot logs, yet KV
  stayed 961,194. The dev memory/KV gain is tied to the *remaining* dev
  settings (AOT compile / `--gdn-decode-kernel b12x` / encoder-tp-mode /
  disable-custom-allreduce) — the same settings that slow decode. Conclusion:
  **"master + memory" without the decode cost is not achievable** on r22.

## NCCL micro-opt (r22-qualified defaults)

| variant | C1 ctx0 | C8 ctx0 | C8 ctx64k | coding |
|---|---|---|---|---|
| master | 30.0 | 103.8* | 128.5 | 36.7 |
| **match22** (`NCCL_MAX_NCHANNELS=16`, `NCCL_BUFFSIZE=2MiB`, OMP=1) | 32.8 | 130.2 | 128.5 | 35.9 |

- Par-or-slightly-better (C8 ctx0 130 vs 104; C1 ctx0 32.8 vs 30.0). Safe to
  adopt — it matches the published r22 qualified NCCL defaults. No regression.

## Plateau summary

| axis | result | plateau? |
|---|---|---|
| PP | ~1.9–2.0k (batch 8192 + piecewise) | ✅ at ceiling (dual-rail is the only remaining lever, parked) |
| Decode C1 | ~28–33 adaptive MTP | ✅ (fixed-MTP worse; acceptance ~2.5 capped) |
| Decode C8 | ~104–131 adaptive MTP | ✅ |
| Decode coding | **DFlash2 47.3** vs MTP3 36.7 | ✅ (df profile for code) |
| Context/KV | 961k @ 7 GiB pin | ✅ (dev 1.1M not isolable without decode cost) |

**Final config = master (adaptive MTP3)**, unchanged. DFlash2 (`df`) remains
the coding/long-single-stream option (wedge fixed on r22, coding +34%).
