# Start here

GLM-5.3-Flash on 2× DGX Spark pairs (TP2), serving **both** checkpoints —
the NVFP4-**Spark** quant and the non-spark NVFP4 quant — on the devspark2
build: vLLM `dev/jovian-judgement` `2a979314` + b12x `f46fee91`, the b12x
O_DIRECT checkpoint loader running **with** MTP (upstream `eh_proj`
loader-allocation fix — the devspark era had to fall back to instanttensor).
PR 646 not composed (closed upstream unmerged; see RECIPE lessons 21/22).

Four profiles on cluster 1 (r0/r1), one image:

| command (from repo dir) | model | speculator | KV pin | MM |
|---|---|---|---|---|
| `./pairctl.sh 1 up mtp3-spark` | spark quant | MTP3 adaptive 1/3/32 | 9.5 GiB | 4 img / 0 vid |
| `./pairctl.sh 1 up mtp3-nvfp4` | non-spark | MTP3 adaptive 1/3/32 | 6.5 GiB | 4 img / 0 vid |
| `./pairctl.sh 1 up df-spark` | spark quant | DFlash2@7 | 12.5 GiB | 4 img / 0 vid |
| `./pairctl.sh 1 up df-nvfp4` | non-spark | DFlash2@7 | 4.5 GiB | 4 img / 1 vid |

First boot on a new image (cold JIT): `PAIR_HEALTH_TIMEOUT=1900 ./pairctl.sh 1 up <profile>`.

Cluster 2 (r2/r3) runs the separate **devc646** line — leave it alone during
cluster-1 work.

**Read order:** [`RECIPE.md`](RECIPE.md) (build + serve + memory envelope +
lessons — authoritative) → [`runs/`](runs/) (benchmark history). Live host
env files (`~/builds/…/serve/TP2-DGX-Spark-GLM5.3F-Jovian-Judgement/`) are
ground truth over anything a doc claims.
