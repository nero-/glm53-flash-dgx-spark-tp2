# GLM-5.3-Flash TP2 pair — operator guide (no-agent edition)

Two DGX Sparks serving **GLM-5.3-Flash** as one TP2 model, both checkpoints —
the NVFP4-Spark quant and the non-spark NVFP4 quant — on the
`glm53-flash-nvfp4-head0906-managed` image (b12x loader + MTP3/DFlash2).
`gx10-r0` is the API head (rank 0), `gx10-r1` is the headless worker (rank 1).
Vision: 4 images per prompt (+1 video on `df-nvfp4`). Four profiles — pick one
per boot (KV pins 10.5 / 6.5 / 12.5 / 4.5 GiB; see the table below).
A SECOND cluster exists: `gx10-r2` (head) + `gx10-r3` (worker) running the
**same `head0906-managed` image** and the same four profiles — see "Second
cluster" below; don't cycle it from a cluster-1 session.

## Hosts and access

| | IP (LAN) | IP (direct CX7 link) | role |
|---|---|---|---|
| gx10-r0 | <r0-lan-ip> | <r0-fabric-ip> | rank 0, API + web UI |
| gx10-r1 | <r1-lan-ip> | <r1-fabric-ip> | rank 1, headless worker |
| gx10-r2 | <r2-lan-ip> | <c2-r0-fabric-ip> | second cluster: rank 0, API |
| gx10-r3 | <r3-lan-ip> | <c2-r1-fabric-ip> | second cluster: rank 1, worker |

```bash
ssh -4 gx10-r0          # user nero, key auth. -4 is REQUIRED (mDNS/IPv6 quirk)
ssh -4 gx10-r1
# sudo password when needed: REDACTED-SUDO-PASSWORD  ->  echo REDACTED-SUDO-PASSWORD | sudo -S <cmd>
```

## Files on each host

```
~/builds/glm53-flash-dgx-spark-tp2/
  models/          model weights (both nodes, identical)
  cache/           compiled-kernel cache (per node)
  serve/TP2-DGX-Spark-GLM5.3F-Jovian-Judgement/
    glm53_pair_serve.sh     the launcher (--check/--run/--verify/--logs/--status/--down)
    rank-0*.env             on r0 only
    rank-1*.env             on r1 only
  builder/         image builder (r0 only)
  GUIDE.md         this guide (on both nodes)
```
(same tree on r2/r3 for cluster 2: `rank-2*.env` on r2, `rank-3*.env` on r3,
same launcher, no builder)
Env-file profiles on cluster 1 (r0 / r1) — pick ONE matching pair per boot:

| files (r0 / r1) | what it is |
|---|---|
| `rank-{0,1}-mtp3-spark.env` | **MTP3 spark quant — daily driver**: batch 8192, KV pinned 10.5 GiB (11274289152; 512k ctx, 2.80x pool), 512k context, 4 img / 0 vid, marlin MTP experts |
| `rank-{0,1}-mtp3-nvfp4.env` | **MTP3 non-spark quant**: batch 8192, KV pinned 6.5 GiB (6979321856), 512k context, 4 img / 0 vid, humming MTP experts |
| `rank-{0,1}-df-spark.env` | **DFlash2@7 spark**: batch 8192, KV pinned 12.5 GiB (13421772800), 256k context, 4 img / 0 vid |
| `rank-{0,1}-df-nvfp4.env` | **DFlash2@7 non-spark**: batch 8192, KV pinned 4.5 GiB (4831838208), 256k context, 4 img / 1 vid |

(DFlash2 draft is CC BY-NC-ND — non-commercial; MTP3 is the commercial path.)

## Start it — one command (pairctl.sh)

From the machine that hosts the repo clone, in its directory (positional
first arg = cluster, default 1; default profile `mtp3-spark`):

```bash
./pairctl.sh 1 up mtp3-spark   # or mtp3-nvfp4 / df-spark / df-nvfp4
./pairctl.sh 1 down            # stop both ranks (default profile; names share the teardown)
./pairctl.sh 1 status          # container state on both nodes
./pairctl.sh 1 logs 0          # follow head logs (1 = worker); optional [profile]
./pairctl.sh 1 check           # validate env files without starting
# first boot on a NEW image (cold JIT): PAIR_HEALTH_TIMEOUT=1900 ./pairctl.sh 1 up <profile>
```

`up` does everything below for you: fixes swappiness, re-checks the RoCE GID
index (auto-fixes the env files after a reboot), drops page cache, tears down
stale containers, starts **worker first, then head**, and waits for the API to
come healthy. It asks once for the pair's sudo password and caches it in
`~/.pair-sudo` (never in git).

## Second cluster (gx10-r2 / gx10-r3) — same image, same four profiles
Everything above applies to cluster 1. The second pair runs the SAME
`local/vllm:glm53-flash-nvfp4-head0906-managed` image and the same four
profile names, through the same pairctl — positional cluster arg (or the
`PAIR_CLUSTER=2` env form):

```bash
./pairctl.sh 2 up mtp3-spark    # also mtp3-nvfp4 / df-spark / df-nvfp4 (same pins as cluster 1)
./pairctl.sh 2 down             # stop both ranks (status / check / logs 0|1 the same way)
PAIR_CLUSTER=2 PAIR_HEALTH_TIMEOUT=1900 ./pairctl.sh up df-nvfp4   # cold JIT boot
```
- API on r2 (LAN <r2-lan-ip>:8000), model name `zai-org/GLM-5.3-Flash`.
- `KV_CACHE_MEMORY_BYTES` MUST be set on every profile (it already is in all
  templates) — the engine refuses unpinned 512k/262k profiles; while a pin is
  set the engine skips memory profiling and `gpu_memory_utilization` is ignored.
- Do not cycle cluster-2 serving containers from a cluster-1 session. (The
  old "the agent runtime lives inside them" reason is obsolete — agent
  sessions run on the operator's machine since 2026-09-07 — but cluster-2
  lifecycle is still not cluster-1 business. The pre-2026-09-07 devc646
  parking line is retired and its image is DELETED; the legacy
  `rank-{2,3}-{mtp3,dflash2}.env` templates no longer boot — history in
  RECIPE.md.)
- After a reboot: swappiness, GID check, `drop_caches` — same drill as
  cluster 1; pairctl auto-fixes the GID index for the r2/r3 env files when
  the cluster arg is 2.

## Manual start (what `pairctl.sh up` does, step by step)

Do this after any reboot or after the pair was down:

```bash
# 1. After a reboot, check swappiness (must be 0) on both:
ssh -4 gx10-r0 'cat /proc/sys/vm/swappiness'
ssh -4 gx10-r1 'cat /proc/sys/vm/swappiness'
# if it shows 60, fix it:
ssh -4 gx10-r0 'echo REDACTED-SUDO-PASSWORD | sudo -S sh -c "echo 0 > /proc/sys/vm/swappiness"'
ssh -4 gx10-r1 'echo REDACTED-SUDO-PASSWORD | sudo -S sh -c "echo 0 > /proc/sys/vm/swappiness"'

# 2. Check the RoCE GID index (this shifts between reboots sometimes):
ssh -4 gx10-r0 show_gids        # find IPv4 <r0-fabric-ip> -> its INDEX (currently 3)
ssh -4 gx10-r1 show_gids        # find IPv4 <r1-fabric-ip> -> its INDEX (currently 3; was 4 pre-2026-09-04)
# If the index differs, edit it in the env files you plan to use (or just run
# pairctl — it re-checks and auto-fixes every `up`):
#   r0: nano ~/builds/glm53-flash-dgx-spark-tp2/serve/TP2-DGX-Spark-GLM5.3F-Jovian-Judgement/rank-0-mtp3-spark.env
#   r1: same path, rank-1-mtp3-spark.env   -> line NCCL_IB_GID_INDEX=<index>

# 3. Drop page cache on both (vLLM counts it against free memory):
ssh -4 gx10-r0 'echo REDACTED-SUDO-PASSWORD | sudo -S sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"'
ssh -4 gx10-r1 'echo REDACTED-SUDO-PASSWORD | sudo -S sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"'

# 4. Start WORKER first, then HEAD (about 3-5 min to load); here: mtp3-spark.
# The launcher wires the env-file settings itself (KDA prefill = flashkda,
# geometry 2048/256, KV pin, ...); pairctl adds the ONLY extra flag,
# --recurrent-checkpoint-policy request_boundaries:
ssh -4 gx10-r1 'cd ~/builds/glm53-flash-dgx-spark-tp2/serve/TP2-DGX-Spark-GLM5.3F-Jovian-Judgement && bash glm53_pair_serve.sh --run rank-1-mtp3-spark.env --recurrent-checkpoint-policy request_boundaries'
ssh -4 gx10-r0 'cd ~/builds/glm53-flash-dgx-spark-tp2/serve/TP2-DGX-Spark-GLM5.3F-Jovian-Judgement && bash glm53_pair_serve.sh --run rank-0-mtp3-spark.env --recurrent-checkpoint-policy request_boundaries'

# 5. Verify it is healthy (expect "GPU KV cache size", "Application startup complete", health OK):
ssh -4 gx10-r0 'cd ~/builds/glm53-flash-dgx-spark-tp2/serve/TP2-DGX-Spark-GLM5.3F-Jovian-Judgement && bash glm53_pair_serve.sh --verify rank-0-mtp3-spark.env'
```

Everyday verbs (same dir on the node; here `rank-0-mtp3-spark.env`, use the
env file you booted):

```bash
bash glm53_pair_serve.sh --status rank-0-mtp3-spark.env      # is the container up?
bash glm53_pair_serve.sh --logs   rank-0-mtp3-spark.env      # follow logs (Ctrl-C to stop)
bash glm53_pair_serve.sh --down   rank-0-mtp3-spark.env      # stop this rank (do BOTH ranks)
bash glm53_pair_serve.sh --check  rank-0-mtp3-spark.env      # dry-run: validate + print command
```

Switch profiles (mtp3 ↔ dflash2, spark ↔ nvfp4): `--down` both ranks, then
`--run` the other env pair (e.g. `rank-1-df-nvfp4.env` on r1,
`rank-0-df-nvfp4.env` on r0) — or just `./pairctl.sh 1 up <profile>`.

## Using the API once it is up

OpenAI-compatible endpoint (no auth, API key can be anything/blank):

| thing | value |
|---|---|
| Base URL | `http://<r0-lan-ip>:8000/v1` |
| Model name | `zai-org/GLM-5.3-Flash` |
| Health | `http://<r0-lan-ip>:8000/health` · models: `/v1/models` |

```bash
# text
curl http://<r0-lan-ip>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"zai-org/GLM-5.3-Flash","messages":[{"role":"user","content":"Hello!"}]}'

# vision: 1-4 images + up to 1 video per message (data URI or http URL)
curl http://<r0-lan-ip>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"zai-org/GLM-5.3-Flash","messages":[{"role":"user","content":[
       {"type":"text","text":"What color is this?"},
       {"type":"image_url","image_url":{"url":"data:image/png;base64,<BASE64>"}}]}]}'
```

### Pointing your clients at it

- **Open WebUI (OWUI)** — Admin Panel → Settings → Connections → OpenAI API:
  URL `http://<r0-lan-ip>:8000/v1`, Key: anything (e.g. `none`), then the
  model picker shows `zai-org/GLM-5.3-Flash`.
- **Hermes** — add an OpenAI-compatible provider with the same base URL and
  model name above.
- **DSH (this GUI)** — Models/Providers settings → add a **custom provider**:
  base URL `http://<r0-lan-ip>:8000/v1`, model `zai-org/GLM-5.3-Flash`
  (model discovery reads `/v1/models`), API key can be blank.

## Re-running the benchmark

On the operator machine (uses the venv in this repo; on Linux rebuild it
once: `python3 -m venv bench/.venv && bench/.venv/bin/pip install httpx rich`):

```bash
cd /Users/jg/Agent/Builds/glm53-flash-dgx-spark-tp2/bench
./.venv/bin/python llm_decode_bench.py \
  --host <r0-lan-ip> --port 8000 --model zai-org/GLM-5.3-Flash \
  --concurrency 1,2,3,4,8 --contexts 0,8192,32768,65536 \
  --prefill-contexts 8k,32k,64k,128k --max-tokens 2048 --duration 30 \
  --coding-peak --coding-peak-runs 3 --coding-peak-max-tokens 65536 \
  --display-mode plain --no-hw-monitor \
  --kv-budget <pool_tokens_from_boot_log> --output <name>.json
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Free memory … less than desired GPU memory utilization` | run step 3 (`drop_caches`), or `--down` a stale container |
| Both ranks sit waiting forever at boot | typo in `MASTER_ADDR`, or wrong GID index — redo step 2 |
| NCCL errors right after a reboot | the GID table shifted — redo step 2 and edit the env files |
| `port 8000 already in use` | `--down` the old container first |
| A node OOM'd and SSH answers but hangs | power-cycle that Spark, then redo steps 1-3 |
| Wipe compiled caches (after image change) | `bash glm53_pair_serve.sh --clear <env>` (next boot recompiles, slower) |

Rule of thumb: **worker (r1) up first, head (r0) second; bring both down
before any reboot, and always re-check swappiness + GIDs after one.**
