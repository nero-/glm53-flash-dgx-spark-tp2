# GLM-5.3-Flash on 2× DGX Spark (TP2)

Notes, recipes, and run reports for serving **GLM-5.3-Flash** (multimodal —
both the NVFP4-Spark and the non-spark NVFP4 quant) across two NVIDIA DGX
Spark (Grace + GB10, SM 12.1) as one TP2/DCP1 pair, using the **Jovian
Judgement** arm64 vLLM stack from
[local-inference-lab](https://github.com/local-inference-lab).

| | |
|---|---|
| Engine | `local-inference-lab/vllm` `dev/jovian-judgement` @ `2a979314` (#667/#669/#672 + loader-owned GLM/KDA weights + MTP `eh_proj` allocation fix + #674/#675/#683) |
| Kernels | B12X master @ `f46fee91` (GB10 `--load-format b12x` O_DIRECT loader, strided TP slices, TP2 peer-push generation-safe) + FlashInfer + patched NCCL 2.30.4 |
| Image | `local/vllm:glm53-flash-nvfp4-devspark2`, built with the arm64/SM121 retarget of `blackwell-llm-docker` (CUDA 13.2 / torch 2.13.0+cu132); previous line `…-devspark` kept as rollback |
| Weights | [`local-inference-lab/GLM-5.3-Flash-NVFP4`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4) (non-spark, 199.4 GB; MXFP8 MTP experts → humming) + `GLM-5.3-Flash-NVFP4-Spark` (spark quant; NVFP4 MTP experts → marlin) |
| Speculators | built-in MTP3 (adaptive 1–3) and DFlash2 draft @ 7 tokens |
| Serving profiles | `mtp3-spark` (MTP3, 9.5 GiB KV) · `mtp3-nvfp4` (MTP3, 6.5 GiB) · `df-spark` (DFlash2@7, 12.5 GiB) · `df-nvfp4` (DFlash2@7 + video, 4.5 GiB) |
| Fabric | 2× DGX Spark, NVIDIA-Sync-managed ConnectX-7 RoCEv2 link (<c1-fabric-subnet> rail) |

## Files

- [`START.md`](START.md) — one-screen current state (image, four profiles, pins)
- [`RECIPE.md`](RECIPE.md) — the current recipe: build, serve profiles, memory envelope, lessons
- [`pairctl.sh`](pairctl.sh) — one-command pair control (`[1|2] up mtp3-spark|df-spark|mtp3-nvfp4|df-nvfp4` / `down` / `status` / `logs` / `check`)
- [`serve/glm53_pair_serve.sh`](serve/glm53_pair_serve.sh) — the host launcher pairctl drives (deploy to both nodes)
- [`builder/dgx-spark-builder/`](builder/dgx-spark-builder/) — the arm64/SM121 image-builder wrapper + build profiles
- [`env/`](env/) — the per-rank env templates for the four cluster-1 profiles (+ cluster-2 devc646)
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
