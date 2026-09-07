# Overnight run summary — r22 build + optimization (2026-09-04 → 09-05)

## What was done
1. **Built r22** (`local/vllm:glm53-flash-nvfp4-r22`): vLLM `70b3c1c7` +
   B12X `1e59a1fd` via dgx-spark-builder (~13 min warm), loaded to both nodes.
2. **DFlash2 (`df`) tested** — see `runs/run-10-r22-dflash2-mtp3.md`.
3. **MTP3 (`master`) tested** — same file.
4. **Optimization iterations to plateau** — `runs/run-11-optimization-iterations.md`.

## Headline results

### DFlash2 (r22) — WEDGE FIXED
- 32k/64k/128k prefills all complete (were hanging on r17/r20/r21).
- **Coding peak 47.3 t/s** = +34% vs MTP3 (35.4). C1 ~28 (par), C8 ~60
  (MTP3 wins at concurrency — deeper draft = 2× verify batch).
- Real acceptance confirmed (8.45% of draft tokens; ~4.4 accept on code).

### MTP3 (r22) — no regression
- C1 ~28–30 · C8 104–131 · coding 36.7 · PP ~2.0k — par-or-better vs jj-master.

### Optimization plateau (run-11)
- **Adaptive MTP wins**; fixed@3 collapses to accept 1.0 at long ctx (C8 ~94 vs ~130). Keep `ADAPTIVE_SPECULATIVE_TOKENS=1`.
- **`match22` NCCL defaults** (16 ch + 2 MiB + OMP1) par/slightly-better — safe optional.
- **Dev memory/KV win NOT isolable**: mm-flags, DS layout, V2 runner all leave KV at 961,194. The dev 1.1M came from the decode-slowing settings (AOT/gdn-decode). "Master + memory without the slowdown" isn't achievable.

### Recommendation
- **Daily driver = master (adaptive MTP3)** — unchanged, now on r22.
- **Coding / long-single-stream = `df` (DFlash2)** — coding +34%.
- PP is at ceiling (~2k); dual-rail RoCE is the only remaining lever (parked).

## Final config
`master` (MTP3 adaptive, 8192 batch, 7 GiB pin, 512k ctx) on r22 image,
serving on `192.168.50.23:8000`. Left running.

## Quality gate — estonia (MTP3-r22, 10-run, C4)
**0/10 PASS — but 0 wrong answers; all 10 hit the 1000 s wall-clock timeout**
with no "Final answer" line. Each run generated ~17.5k tokens of reasoning at
~4.5 tok/s/request (aggregate 18.2 tok/s at C4 × ~166k ctx) before the 1000 s
cap. No `locklock` loop; output is coherent, just extremely verbose and
KV-bandwidth-bound at long context. This is the first estonia measurement on
this hardware (no jj-master baseline exists — run-05 never ran estonia), so it
reads as a *long-context decode speed + reasoning-verbosity* limit, not a
correctness regression. To get a scorable gate: run at C1 or raise
`--completion-stats-request-timeout` well above 1000 s.
