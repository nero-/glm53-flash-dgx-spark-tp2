# Run 05 — jj-master validation (the developer's fixes)

Date: 2026-09-04 · Image: `local/vllm:glm53-flash-nvfp4-jj-master` (built
OK+VERIFIED, 1h09m cold, peak 88.1 GiB) · Profile: `rank-{0,1}-master.env`
(MTP3, batch 8192, KV pin 7 GiB, 512k ctx, vision 4/1).

## Pins (upstream master, post-fix)
- vLLM `local-inference-lab/vllm` `dev/jovian-judgement` @ `a50ebee1d2460d22386b54e79f46236376e2b486`
- B12X `local-inference-lab/b12x` master @ `9ae41c5cb9935d740456479954b0089f80bd2ef2`
- Aligned with the r21 runtime contract (rtx6kpro `models/glm-5.3-flash.md`):
  **`MTP_MOE_BACKEND=marlin`** — boot log confirms
  `Using 'MARLIN' NvFp4 MoE backend` for the MTP head (target experts stay B12X).

## Boot
Healthy in ~4.5 min · **GPU KV cache: 961,194 tokens** (same 8192/7 GiB
geometry as run 04) · graphs captured 47 s · PYNCCL all-reduce · B12X
attention/linear · FlashInfer sampling.

## Quality validation (the whole point)
| Test | Result |
|---|---|
| "Count to 100" (fresh boot, first minutes) | ✅ coherent reasoning + counts 1, 2, 3… (no `locklock` loop) |
| "Capital of France?" | ✅ "The capital of France is Paris." |
| Vision (solid red PNG) | ✅ "Red" |

All three on the previously-corrupting configuration class (batch 8192,
MTP3, SM121). The first-token loop corruption is **fixed** on this line.

## Next
- John runs the full benchmark set (estonia 30×C4, throughput suite,
  coding peak) — see PLAN-RUNS.md.
- DFlash2 option (`pairctl.sh up df`) to be validated separately.
