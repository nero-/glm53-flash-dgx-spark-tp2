# Start here

GLM-5.3-Flash (non-spark NVFP4 quant) on 2× DGX Spark pairs (TP2), the
devspark build (vLLM dev head + b12x master with the GB10 `--load-format
b12x` loader; PR 646 dropped, #667/#669 merged). Two profiles, same names
on both clusters:

- `./pairctl.sh 1 up mtp3` — cluster 1 (r0/r1): MTP3, images, 6.5 GiB KV
- `./pairctl.sh 2 up mtp3` — cluster 2 (r2/r3): MTP3, video, 6.5 GiB KV
- `./pairctl.sh 1 up df` / `./pairctl.sh 2 up df` — DFlash2@7, 4.5 GiB KV

**Read order:** [`RECIPE.md`](RECIPE.md) (build + serve + memory envelope +
lessons) → [`runs/`](runs/) (benchmark history).
