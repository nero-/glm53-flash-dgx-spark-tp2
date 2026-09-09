# Run index

Series A — self-composed r17 image (retired after the 900k OOM; kept for reference).
Series B — kyonjova r17-roce recipe (image `local/vllm:glm53-flash-nvfp4-R17-roce`
+ `glm53_pair_serve.sh` + rank env files). All B runs: vision 4/1, fp8 KV,
B12X backends, instanttensor, autotune OFF, prefix caching + chunked prefill.

| Run | File | Series | Speculator | Batch | Pin | KV tokens | PP 8k | PP 32k | TG C1 | TG C8 |
|---|---|---|---|---|---|---|---|---|---|---|
| 00 | [run-00-baseline.md](run-00-baseline.md) | A | mtp3 | 4096 | 4.55 GiB | 510,306 | 1,898 | 1,760 | 36.8 | 94.9 |
| 01 | [run-01-smoke.md](run-01-smoke.md) | B | mtp3 | 4096 | unpinned 0.85 | 503,316 | 1,408* | 1,673* | 30.7 | 100.0 |
| 02 | [run-02-mtp3-1m.md](run-02-mtp3-1m.md) | B | mtp3 | 4096 | **10 GiB** | **1,474,826** | 1,810 | 1,679 | 28.0 | 101.1 |
| 03 | run-03-dflash2.md | B | dflash2@7 | 4096 | 10 GiB | 618,951 | — | — | — | — |
| 04 | [run-04-mtp3-8192.md](run-04-mtp3-8192.md) | B | mtp3 | 8192 | 7 GiB | 961,194 | 1,960 | 1,950 | 27.3 | 101.1 |

\* Run 01 ran `CUDAGRAPH_MODE=FULL`, which auto-downgrades to
`FULL_DECODE_ONLY` (eager prefill) on the GDN backend — prefill cells are
eager-mode; decode is representative.
Series C — fabric changes + the devc646 rebuild (r25-line = dev/jovian-judgement
a8c796f3 + PR #646 b4f16bd3, b12x preview; second cluster: gx10-r2/r3).
C1 = 30 s cell C1, C8 = same at concurrency 8 (ctx0).

| Run | File | Series | Speculator | Batch | Pin | KV tokens | PP 8k | PP 32k | TG C1 | TG C8 |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | [run-10-r22-dflash2-mtp3.md](run-10-r22-dflash2-mtp3.md) | C (r22, direct cable) | dflash2@7 | 4096 | 9 GiB | 558,269 | 1,928 | 1,982 | 27.8 | 63.2 |
| 10b | run-10 (mtp3 requal section) | C (r22, direct cable) | mtp3 | 8192 | 7 GiB | 961,194 | 1,978 | 2,017 | 30.0 | 103.8 |
| 12 | [run-12-mtp3-r22b-fabricfix.md](run-12-mtp3-r22b-fabricfix.md) | C (r22b, switch) | mtp3 | 8192 | 9 GiB | 1,235,254 | 1,037-1,080 PP | — | 25.9 | 96.4 |
| 13 | [run-13-dflash2-r22b-fabricfix.md](run-13-dflash2-r22b-fabricfix.md) | C (r22b, switch, partial) | dflash2@7 | 4096 | 9 GiB | 558,269 | 1,005-1,040 PP | — | 24.7 | 57.0 |
| 15 | [run-15-mtp3-devc646.md](run-15-mtp3-devc646.md) | C (devc646, cluster-2) | mtp3 | 8192 | 8 GiB | **1,118,600** | 1,978 | 2,017 | 31.4 | 101.1 |
| 16 | [run-16-dflash2-devc646.md](run-16-dflash2-devc646.md) | C (devc646, cluster-2) | dflash2@7 | 4096 | 8 GiB | 523,009 | 1,912 | 1,955 | 30.3 | ≈50-63 (steps/s-par; C8 usage-gated) |

devc646 KV density: 139.8k tokens/GiB (mtp3) / 65.4k (dflash2: 5 DFlash draft KV
layers kept in 1 independent native-block group). Fine-grained prefix hits on both
profiles (the r22-era `Disabling fine-grained…` lines are gone); 52k replay TTFT
115 s → 1.38 s on dflash2.

Series D — the CURRENT line (head0906 + `-managed` loader, 2026-09-07): ONE
image on both clusters, the same four profiles (mtp3/df × spark/nvfp4), batch
8192 everywhere, `PREFILL_SCHEDULE_INTERVAL=8`, flashkda KDA prefill, marlin
MTP experts on the spark quant / humming on the non-spark one. Its A/Bs
(reference-env adoption, ASYNC_SCHEDULING drop, the marlin pick, the
`-managed` loader patch) ran as quick matrix passes recorded in `bench/` raw
json/txt artifacts (gitignored by convention — raw stays local), not as
run-N.md reports; RECIPE.md's "Key config" + "Reference-env A/B ledger"
sections carry the decisions. Reference points: KV pool **1,466,929 tokens**
at the 10.5 GiB mtp3-spark pin (2.80× a 512k request), prefill back at
~1.65–2.03k tok/s (the `-managed` patch restored what the b12x pinned_wc
loader cost), page census `physical page sizes [1148928, 2244608]`.
