# GLM-5.3-Flash on 2× DGX Spark (TP2)

Notes, recipes, and run reports for serving **GLM-5.3-Flash** (multimodal,
non-spark NVFP4 quant) across two NVIDIA DGX Spark (Grace + GB10, SM 12.1)
as one TP2/DCP1 pair, using the **Jovian Judgement** arm64 vLLM stack from
[local-inference-lab](https://github.com/local-inference-lab).

| | |
|---|---|
| Engine | `local-inference-lab/vllm` `dev/jovian-judgement` @ `858b4912` (merged #667/#669/#672 + b12x-loader integration) |
| Kernels | B12X master @ `79e228ec` (GB10 `--load-format b12x` O_DIRECT loader plugin) + FlashInfer + patched NCCL 2.30.4 |
| Image | `local/vllm:glm53-flash-nvfp4-devspark`, built with the arm64/SM121 retarget of `blackwell-llm-docker` (CUDA 13.2 / torch 2.13.0+cu132) |
| Weights | [`local-inference-lab/GLM-5.3-Flash-NVFP4`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4) (non-spark, 199.4 GB; MXFP8 MTP experts → humming) |
| Speculators | built-in MTP3 (adaptive 1–3) and DFlash2 draft @ 7 tokens |
| Serving profiles | `master` (MTP3, no video, 6.5 GiB KV) · `multi` (MTP3, video, 5.0 GiB KV) · `df` (DFlash2, no video, 4.5 GiB KV) |
| Fabric | 2× DGX Spark, NVIDIA-Sync-managed ConnectX-7 RoCEv2 link (<c1-fabric-subnet> rail) |

## Files

- [`RECIPE.md`](RECIPE.md) — the current recipe: build, serve profiles, memory envelope, lessons
- [`pairctl.sh`](pairctl.sh) — one-command pair control (`up master|multi|df` / `down` / `status` / `logs` / `check`; `PAIR_CLUSTER=2` targets the r2/r3 pair)
- [`serve/glm53_pair_serve.sh`](serve/glm53_pair_serve.sh) — the host launcher pairctl drives (deploy to both nodes)
- [`builder/dgx-spark-builder/`](builder/dgx-spark-builder/) — the arm64/SM121 image-builder wrapper + build profiles
- [`env/`](env/) — the per-rank env files for the three profiles
- [`runs/`](runs/) — one markdown report per benchmarked run
- [`bench/`](bench/) — `llm_decode_bench.py` (llm-inference-bench)

## Upstream / provenance

The runtime derives from the Apache-2.0-licensed
[`local-inference-lab/vllm`](https://github.com/local-inference-lab/vllm),
[`local-inference-lab/b12x`](https://github.com/local-inference-lab/b12x),
[`local-inference-lab/blackwell-llm-docker`](https://github.com/local-inference-lab/blackwell-llm-docker),
[`voipmonitor/flashinfer`](https://github.com/voipmonitor/flashinfer),
[`local-inference-lab/nccl-canonical`](https://github.com/local-inference-lab/nccl-canonical),
[`scitix/InstantTensor`](https://github.com/scitix/InstantTensor) and
[`brandonmmusic-max/exllamav3`](https://github.com/brandonmmusic-max/exllamav3).
Model weights are governed by their repositories' terms
([GLM-5.3-Flash-NVFP4](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4),
[GLM-5.3-Flash-DFlash2](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-DFlash2)
— the DFlash2 draft is CC BY-NC-ND, non-commercial).

This is a personal knowledge repo, not an official local-inference-lab release.
