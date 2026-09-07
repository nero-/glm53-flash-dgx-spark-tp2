# Run 03 — dflash2@7 (kyonjova r17-roce): boots, but blocked by pin-level bugs

Date: 2026-09-03 · Image: `local/vllm:glm53-flash-nvfp4-R17-roce` · Profile:
`rank-0-df.env` / `rank-1-df.env` (dflash2, split pages 4096/4096, piecewise).

## Summary
DFlash2 **boots and serves** on this image, but two pin-level problems
blocked a full bench. The working speculator for daily use on r17-roce is
**MTP3 (Run 02: 1.47M KV, PP 1,810, C8 101.1, coding 37.0)**. Her published
dflash2 numbers (CC1 27.3 / c8 81.0 / coding **49.2** / PP ~1,955) were
qualified on the **R12 image**, not this r17 composition.

## Finding 1 — 12 GiB pin OOMs at boot (fixed by dropping to 10 GiB)
First boot used the R12-qualified 12 GiB pin (`12884901888`). The r17-roce
image is ~2 GiB heavier than R12; the vision-encoder warmup spike pushed r0
over the edge and the host OOM-killer SIGKILL'd the worker:
`Out of memory: Killed process (VLLM::Worker_TP)` (dmesg confirmed,
`exit code: None` = SIGKILL).
**Fix applied:** pin 10 GiB → boots clean, **GPU KV cache: 618,951 tokens**
(17.3 KiB/token split-page geometry, 2.36× at 262k). Expected ~731k at
12 GiB on R12; the ~112k difference is exactly the pin reduction.

## Finding 2 — first-request wedge (NOT fixed; pin-level)
With the 10 GiB pin the engine boots healthy (FlashAttention v2 draft
attention, `B12xMxfp8LinearKernel` fused DFlash K/V projection, split pages
4096/4096, 5 draft KV layers in independent groups, graphs captured) — but
the **first request after boot wedges on most boots**: the request sits in
"Waiting: 1" while the engine logs `EngineCore waiting for work` (KV usage
0.0%, host RAM free) — an API→engine shm-broadcast desync, not memory.
Debug-level logs confirmed the request never reaches the engine core.
Verified across ~8 boots with varied knobs (prefix caching off, chunked
prefill off, cache wipe, engine-ready waits, immediate-fire timing): one
boot out of eight passed 8k+32k prefills (then wedged on 64k). Not fixable
by config; a real regression of the r17 pins vs the R12-qualified image.

## What we could capture
- Boot + serve: OK (619k KV).
- One good window: 8k and 32k prefills completed (TTFT in line with R12),
  then the 64k cell wedged (a second, size-dependent wedge — draft-KV group
  allocation for ≥64k contexts is the suspect).
- No clean bench table — the benches aborted on the wedge every time.

## Conclusion / recommendations
1. **Daily driver: MTP3 1M (Run 02).** Stable, full bench clean.
2. dflash2 on r17-roce: boot with the 10 GiB df profile works, but treat it
   as broken-until-fixed (wedge). If dflash2 numbers are needed, build the
   **R12-qualified image** (`build-glm53-r12.env` in the same builder) — her
   R12 table is published in the recipe README (coding peak 49.2).
3. The df env files remain deployed (10 GiB pin) for whoever picks this up.
