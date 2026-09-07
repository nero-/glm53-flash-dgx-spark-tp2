# Run plan — jj-master (two serving options)

Image: `local/vllm:glm53-flash-nvfp4-jj-master` (local-inference-lab
dev/jovian-judgement master `a50ebee1` + b12x master `9ae41c5c` — the
developer's fixes). Both options: vision 4 img/1 video, fp8 KV, B12X
backends, `MTP_MOE_BACKEND=marlin`, instanttensor, autotune OFF.

| Option | pairctl | Speculator | Batch | KV pin | KV tokens | Max ctx |
|---|---|---|---|---|---|---|
| **MTP3 (daily driver)** | `up master` | mtp@3 adaptive | 8192 | 7 GiB | ~961k | 512k |
| **DFlash2** | `up df` | dflash@7 | 4096 | 9 GiB | ~534k | 256k |

## Validation ladder (each fresh boot)
1. **Chat sanity**: "Count to 100" + a short conversation — must be coherent
   (the SM121 loop fix). First request may be slow (JIT warmup).
2. **Estonia needle** (the corruption discriminator):
   `--test-profile estonia --profile-concurrency 4 --profile-runs 30
   --completion-stats-seed 1000 --completion-stats-request-timeout 1000`
   — target: ≥28/30 PASS on both options.
3. **Throughput suite** on `master`: prefill 8k–128k, C1–C8 decode, coding
   peak — compare against run 04 (PP ~1.85–1.96k, C8 101.1, coding 36.6).
4. **DFlash2**: coding-peak target 40–50; re-test the old 64k+ prefill wedge
   (was broken on r17-roce — master may fix it; stay ≤32k if not).

## What "done" looks like
- Chat coherent on first request after fresh boot (both options).
- Estonia ≥28/30 on `master` (and `df`).
- `master` keeps run-04-class throughput at 8192.
- One MD per run in `runs/`, results published to GitHub.

## History (for reference)
runs/run-00…04 cover the kyonjova r17-roce/r20 era. The corruption
investigation that drove the upstream fixes:
`DIAG-HALLUCINATIONS-2026-09-03.md`.
