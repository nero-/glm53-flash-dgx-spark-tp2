# Bug report — DFlash2 long-prefill request wedge on 2× DGX Spark (TP2)

> **STATUS: FIXED** in the r22b image (2026-09-04) — 32k/64k/128k prefills
> all complete. Kept for reference.

## Summary
Under DFlash2, a single long-prefill request (32k+ tokens) hangs indefinitely
client-side: no response bytes for 5+ minutes. The engine is idle while the
request sits in the waiting queue — a scheduler/transport wedge, not memory
and not compute. Chat and short prompts work fine. The MTP3 profile on the
same image handles 128k prefills without issue, so this is DFlash2-specific.

## Environment
| | |
|---|---|
| Hardware | 2× NVIDIA DGX Spark GB10 (Grace, SM 12.1, 121.7 GiB unified), TP2 / DCP1, direct ConnectX-7 RoCE link |
| Image | `local/vllm:glm53-flash-nvfp4-jj-master` (arm64, cu132, built from source) |
| vLLM | `local-inference-lab/vllm` `dev/jovian-judgement` @ `a50ebee1d2460d22386b54e79f46236376e2b486` |
| B12X | `local-inference-lab/b12x` master @ `9ae41c5cb9935d740456479954b0089f80bd2ef2` |
| Target model | `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` |
| Draft model | `local-inference-lab/GLM-5.3-Flash-DFlash2` (MXFP8, `model.safetensors` sha256 `c033e03d47c7d5608596c8fc4e9336a1fe086eb781c08fe031be2bdea1614e58`) |

## Serve configuration
```
SPECULATOR=dflash2, NUM_SPECULATIVE_TOKENS=7
MAX_MODEL_LEN=262144, MAX_NUM_SEQS=8
MAX_NUM_BATCHED_TOKENS=4096, MAX_CUDAGRAPH_CAPTURE_SIZE=96
CUDAGRAPH_MODE=FULL_AND_PIECEWISE, PREFILL_SCHEDULE_INTERVAL=8
VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=4096, VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=4096
KV_CACHE_MEMORY_BYTES=9663676416 (9 GiB pin -> 558,269 tokens), KV_CACHE_DTYPE=fp8
attention/moe/linear = B12X, --no-enable-flashinfer-autotune
--enable-prefix-caching --enable-chunked-prefill, --load-format instanttensor
vision on (limit 4 img / 1 video), LD_PRELOAD = CUDA 13.2 compat shim + patched NCCL 2.30.4
```

## Repro (single request, fresh boot)
```bash
# 32k-token prompt, non-streaming, 300s client timeout
curl -m 300 http://<host>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"zai-org/GLM-5.3-Flash","messages":[{"role":"user","content":"<32k tokens of any text>"}],"max_tokens":16}'
```
Observed: client ReadTimeout (no bytes for 300s). Chat request ("Count to 10")
just before it completed normally.

## Engine evidence (docker logs, during the hang)
```
(Worker_TP0) WARNING [jit_monitor.py] Triton kernel JIT compilation during inference:
    _compute_local_logits_stats_kernel ... _rejection_kernel ... _resample_kernel
    (compiled during the request — warmup does not cover these shapes)
(APIServer) Engine 000: Avg prompt throughput: 1.7 tokens/s, Avg generation
    throughput: 1.1 tokens/s, Running: 1 reqs, Waiting: 0 reqs,
    GPU KV cache usage: 14.8%          <-- request starts running
(APIServer) Engine 000: Avg prompt throughput: 0.0, gen 4.8 tok/s,
    Running: 0 reqs, Waiting: 1 reqs, GPU KV cache usage: 0.0%   <-- then...
(APIServer) Engine 000: ... Running: 0 reqs, Waiting: 1 reqs, GPU KV cache usage: 0.0%  <-- stuck forever
```
Worker CPU ~3% (idle), engine core logs `EngineCore waiting for work` at
DEBUG level: the request never reaches the engine core after the wedge;
the API-side queue shows it waiting while the engine is free.

## Prior evidence (same signature, older pins)
- r17-roce image (vLLM `5742ea6a`+PRs, b12x `fb43ea1a`): 8k and 32k prefills
  completed on one boot, the 64k cell wedged identically (`Waiting: 1`,
  `Running: 0`, KV 0.0%). Across ~8 boots the first long request wedged on
  most boots; only one window passed 8k+32k.
- On jj-master the wedge moved DOWN: the 32k probe above wedged on the
  first long request.

## Ruled out (no effect)
- `--enable-prefix-caching` OFF (env `ENABLE_PREFIX_CACHING=0`)
- `--enable-chunked-prefill` OFF (`ENABLE_CHUNKED_PREFILL=0`)
- JIT cache wipe + fresh boots, engine-ready waits (starting requests only
  after "Application startup complete" + settle), multiple restart cycles
- Memory: not OOM (host has free RAM; KV usage 0.0% during the wedge)
- MTP3 on the same image at batch 8192 with 128k prefills: no wedge

## What we'd like to know
1. Is this the known #616/#622-class dflash C4 tail-ring / broadcast
   desync, and is the fix for it in `a50ebee1` or only later/in a PR
   (e.g. #622)? We can build any branch/sha you point at and re-test on
   the pair (TP2/GB10 — the topology your qualification matrix lists as
   "implemented, not independently qualified").
2. Can warmup be extended to cover the dflash verify kernels
   (`_compute_local_logits_stats_kernel`, `_rejection_kernel`,
   `_resample_kernel`, `_expand_c4_block_table_kernel`) so they never JIT
   mid-request?

Happy to run any instrumented build (DEBUG logs, NCCL_DEBUG=INFO, py-spy)
on the pair and post results.
