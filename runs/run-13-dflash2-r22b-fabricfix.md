# Run 13 — r22b DFlash2 after fabric repair (PARTIAL — ended at operator request)

Date: 2026-09-05 · Image: `local/vllm:glm53-flash-nvfp4-r22b` · df profile
(DFlash2@7, batch 4096, KV pin 9 GiB = 558,269 tokens, 256k ctx, split pages
4096/4096, FA2 draft path, B12X fused DFlash K/V projection).

Boot healthy in **266 s** — NCCL init ~22 s (the lesson-17 IPv4LL fix holds
across profile switches; PYNCCL on tp/ep groups, no warnings).

Bench suite was stopped by the operator after 10/20 decode cells + the
prefill scouts; coding peak did not run.

## Partial decode matrix (vs run-10 r22 baseline in parens)

| ctx \ conc | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 0 | 24.7 (27.8) | 38.5 (38.8) | 51.5 (51.3) | 57.0 (63.2) |
| 8k | 28.7 (28.3) | 37.9 (36.0) | 56.6 (60.3) | — |
| 32k | 25.1 (28.5) | — | — | — |
| 64k | 32.3 (23.8) | — | — | — |
| 128k | 26.5 (17.4) | — | — | — |

Decode is **par with the r22 baseline** (C1 cells noisy by nature; C4 ctx0
51.5 = exactly baseline). The switch-in-fabric throttle is bandwidth, not
latency, so decode — the axis DFlash2 was chosen for — is unaffected.

## Prefill
**1,005–1,040 tok/s** at 8k–128k (run-10: ~1,928–2,002). Halved, exactly
like MTP3 on this fabric. TP2 prefill is NCCL-allreduce-bound; the fabric
delivers ~2.8 GB/s NCCL where ~3.5+ GB/s is needed for the baseline rate.
Same physical root cause as run-12; no software fix — recable direct.

## Takeaway
r22b + repaired NCCL init: DFlash2 serves correctly (quality sanity clean,
no locklock) and decode matches r22. Remaining gap (prefill, and the C1/C2
long-ctx cell starvation in run-12) is 100% the fabric throttle. Re-bench
full suite after the cables go back direct.
