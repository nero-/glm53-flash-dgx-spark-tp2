# Run 00 — baseline (mtp3, batch 4096, pin 4.55 GiB)

- **Date:** 2026-09-02 · **Image:** glm53-jj-arm64:r17 (sha256:821a1e52…)
- **Config:** TP2/DCP1 · SPECULATOR=mtp · NUM_SPECULATIVE_TOKENS=3 (B12X spec
  backend) · MAX_MODEL_LEN=262144 · KV_CACHE_MEMORY_BYTES=4548000000 (4.55 GiB)
  · MAX_NUM_BATCHED_TOKENS=4096 · MAX_NUM_SEQS=8 · MAX_CUDAGRAPH_CAPTURE_SIZE=96
  · PREFILL_SCHEDULE_INTERVAL=8 · BLOCK_SIZE=256 · GPU_MEMORY_UTILIZATION=0.90
  (inert, pin set) · vision 4 img/1 video · FlashInfer autotune OFF ·
  LINEAR_BACKEND=b12x · OMP_NUM_THREADS=2 · kyonjova NCCL profile.
- **Markers:** GPU KV cache size 510,306 tokens (1.95x @ 262k) · Add 2 padding
  layers (5.88% waste) · Graph capturing finished (26-28 s, 1.3-1.9 GiB) ·
  Application startup complete · /health OK · vision probe answered.
- **Change vs previous:** — (first stable serving config).

## llm-inference-bench v0.4.32 results (verbatim summary)

```text
Engine: vLLM 0.1.dev20051+jj15  Models: ['glm-5.3-flash']
KV cache budget (vLLM metrics): 672,768 tokens (146 blocks × 4608)
Model context length: 262,144 tokens

Prefill Speed (scout requests, client ISL / TTFT)
  Context   Tokens   TTFT (s)   Client tok/s   Server tok/s
  8k         8,197       4.32          1,898              —
  16k       16,224       9.01          1,801      1,820 (1)
  32k       32,313      18.36          1,760      1,772 (1)

Sustained Decode — Aggregate tok/s + TTFT/ITL (ms)
  ctx \ conc │      1 │        2 │      4 │       8
  0          │   36.8 │ ∅ (2/2)* │   70.1 │    94.9
  16k        │   24.8 │ ∅ (2/2)* │   62.9 │    91.6
  32k        │   27.1 │ ∅ (2/2)* │   57.7 │    91.8

Per-Request tok/s:  C1 36.8/24.8/27.1 · C4 17.5/15.7/14.4 · C8 11.9/11.5/11.5

MTP-normalized decode steps/s (accept len)
  ctx \ conc │           1 │           2 │           4 │           8
  0          │ 12.6 (2.92) │ 18.3 (2.58) │ 27.1 (2.59) │ 37.3 (2.54)
  16k        │ 10.4 (2.39) │ 15.2 (2.83) │ 25.0 (2.52) │ 34.7 (2.64)
  32k        │ 10.7 (2.53) │ 15.6 (2.43) │ 26.0 (2.22) │ 35.5 (2.59)
```

## Takeaways
- Prefill ≈1.76–1.90k tok/s; the 2k PP goal needs the batch-8192 lever (run 01).
- Engine rate 12.6 steps/s C1 matches the poster's mtp3 (12.06 steps/s).
- C2 cells were skipped by the tool's budget check — pass `--kv-budget 510000`
  in later runs to fill them in.
