# Agent brief — starting a NEW agent session for GLM-5.3-Flash pair work

Everything a fresh agent needs to know about the GLM-5.3-Flash repo and the
two DGX Spark pairs behind it — so you (John) don't have to re-explain.
The CROSS-project brief (access from any machine, first-time setup, both
clusters + other projects) lives in the synced workspace:
`~/Agent/Builds/AGENT-BRIEF.md`. This file tracks what this repo serves.

## Paste-in prompt template (for a new agent session)

Copy the block below into a new session. Replace `<TASK>` with the repo/link
or task description; delete sections that don't apply.

---

Read `~/Agent/Builds/AGENT-BRIEF.md` first — it covers the pairs' access,
hardware, folder layout, and conventions. Also inspect `~/Agent/Builds/`
for project folders and read the relevant project's RECIPE/OPS-GUIDE before
changing anything (here: `glm53-flash-dgx-spark-tp2`). If this machine has
never connected to the pairs before, start with the cross-project brief's
"First-time setup on a new machine" section.

You are helping me with my GLM-5.3-Flash serving on NVIDIA DGX Spark pairs
(GB10, arm64/SM12.1, 121.7 GiB unified memory each) — one TP2/DCP1 pair per
cluster over a direct ConnectX-7 RoCE link. Cluster 1 (`gx10-r0` head +
`gx10-r1` worker) is the daily-driver pair; cluster 2 (`gx10-r2`+`gx10-r3`)
runs the SAME image and the same four profiles.

Context and conventions (read these first — they are kept current):
- `~/Agent/Builds/glm53-flash-dgx-spark-tp2/RECIPE.md` — current recipe, pins,
  memory envelope, bug log (authoritative for lessons)
- `~/Agent/Builds/glm53-flash-dgx-spark-tp2/OPS-GUIDE.md` — operator guide
- `~/Agent/Builds/glm53-flash-dgx-spark-tp2/START.md` — one-screen current state
- `~/Agent/Builds/glm53-flash-dgx-spark-tp2/runs/` — per-run reports
  (INDEX.md lists the series; raw bench json/txt artifacts stay LOCAL/ignored)

Hardware/access (sudo password in the SSH info file):
- `ssh -4 gx10-r0` (API head, rank 0) / `ssh -4 gx10-r1` (headless worker,
  rank 1), user `nero`; cluster 2 = `ssh -4 gx10-r2` / `ssh -4 gx10-r3`.
- `-4` forces IPv4 (mDNS/IPv6 quirk). Under the DSH/agent sandbox ssh may
  fail with "Bad owner … /etc/ssh/ssh_config.d/…" — use
  `ssh -F ~/.ssh/config` (pairctl already does).
- sudo: `echo <password> | sudo -S <cmd>` — password in
  `~/Agent/Builds/Cluster gx10-r0-r1 SSH info.txt` (cluster 1 only; the
  cluster-2 sudo password the operator supplies separately).
- Fabric: BOTH nodes of each pair use their f1 ports (`rocep1s0f1` /
  `enp1s0f1np1`), RoCEv2 GID index 3 — cluster 1
  `<c1-r0-fabric-ip>`/`<c1-r1-fabric-ip>`, cluster 2
  `<c2-r0-fabric-ip>`/`<c2-r1-fabric-ip>`. The GID table SHIFTS after
  reboots/recables — pairctl re-reads `show_gids` and auto-fixes the env
  files on every `up`. Swappiness must be 0.
- r0/r1 container lifecycle: the agent runtime used to ride INSIDE the
  r0-r1 serving container pair (2026-09-05 incident); as of 2026-09-07 agent
  sessions run on the operator's machine and MAY cycle cluster 1 when the
  task calls for it (builds, qualification) — confirm with the operator
  first, because API consumers (OWUI/Hermes/DSH) lose their model while the
  pair is down. Cluster-2 containers (`glm53-flash-r0/-r1` on r2/r3) are
  NEVER to be cycled from cluster-1 sessions.
- Host layout (both nodes unless noted):
  `~/builds/glm53-flash-dgx-spark-tp2/{models, cache,
  serve/TP2-DGX-Spark-GLM5.3F-Jovian-Judgement, builder (r0 only)}`.
- Current line: `local/vllm:glm53-flash-nvfp4-head0906-managed`
  (2026-09-07: vLLM `dev/jovian-judgement` @ `2a979314` + b12x @ `3eb57add`
  + the one-line `-managed` loader patch; b12x loader boots WITH MTP — the
  eh_proj allocation fix is in). BOTH clusters run this one image;
  superseded images (devspark, devspark2, devc646) are deleted from all
  four nodes.
- Four profiles on BOTH clusters (`./pairctl.sh [1|2] up <p>`):
  `mtp3-spark` (MTP3 adaptive 1/3/32, batch 8192, 512k ctx, KV pin 10.5 GiB,
  marlin MTP) · `mtp3-nvfp4` (6.5 GiB, humming) · `df-spark` (DFlash2@7,
  8192 batch, 256k ctx, 12.5 GiB) · `df-nvfp4` (4.5 GiB, +video).
  `KV_CACHE_MEMORY_BYTES` is MANDATORY on this engine — unpinned ≥262k
  profiles refuse to start; while pinned, `gpu_memory_utilization` is ignored.
- Env values uniform on all four profiles: `LOAD_FORMAT=b12x` +
  `VLLM_PLUGINS=b12x_loader`, `KDA_PREFILL_BACKEND=flashkda`, geometry 2048
  target/256 split + `PREFIX_MATCH_UNIT=256` (PR #646 census on the boot
  log is the arbiter), `PREFILL_SCHEDULE_INTERVAL=8`,
  `MM_PROCESSOR_CACHE_GB=0`, `MM_ENCODER_TP_MODE=data`,
  `B12X_POLICY_MODE=auto`, `COMPILATION_LEVEL=2` (launcher passes `-O2` but
  the engine resolves `CompilationMode.NONE` — inert on this line; kept in
  the envs), MM images 4 everywhere, videos 0 except `df-nvfp4` (1).
- Bench: `bench/.venv/bin/python llm_decode_bench.py`. On a Linux machine
  rebuild the venv once: `python3 -m venv bench/.venv && bench/.venv/bin/pip
  install httpx rich`.
- One MD per run in `runs/`, update RECIPE.md when pins/settings change,
  commit + push the GitHub snapshot (`public-main`; NEVER push `main`
  directly — private history).
- Key lessons (full list in RECIPE bug log): RoCE GID shifts after reboots
  (pairctl auto-fixes) · memory binds on the HEAD node (~2.5 GiB heavier;
  OOM cliff ~121.5 GiB; a wedged host needs a physical power-cycle) ·
  adaptive MTP > fixed MTP · marlin on the spark quant's NVFP4 experts,
  humming on the MXFP8 ones · `KDA_PREFILL_BACKEND=flashkda` (in-image since
  head0906) · estonia times out at C4 (run C1 or raise the request timeout)
  · C1 single-run is noisy (judge C8).

My task for you: <TASK>
---

## The condensed brief (the facts the template references)

### Access
| Host | Role | LAN IP | Fabric IP (f1, serving) | RoCE device | GID (v2) |
|---|---|---|---|---|---|
| gx10-r0 | API head, rank 0 (cluster 1) | <r0-lan-ip> | <c1-r0-fabric-ip> | rocep1s0f1 / enp1s0f1np1 | 3 |
| gx10-r1 | worker, rank 1 (cluster 1) | <r1-lan-ip> | <c1-r1-fabric-ip> | rocep1s0f1 / enp1s0f1np1 | 3 |
| gx10-r2 | API head, rank 0 (cluster 2) | <r2-lan-ip> | <c2-r0-fabric-ip> | rocep1s0f1 / enp1s0f1np1 | 3 |
| gx10-r3 | worker, rank 1 (cluster 2) | <r3-lan-ip> | <c2-r1-fabric-ip> | rocep1s0f1 / enp1s0f1np1 | 3 |

`ssh -4` + key auth (above). The second f1-port rail (`roceP2p1s0f1`) is
unused — single-rail is the qualified profile. Node-to-node SSH works over
the fabric. Sudo password is NOT in git: cluster 1's is in the operator's
workspace SSH info file, cluster 2's the operator supplies. Unified memory:
page cache counts against free CUDA memory — `drop_caches` before big boots
(pairctl does it).

### Host layout (both nodes unless noted)
```
~/builds/glm53-flash-dgx-spark-tp2/
  models/glm53-flash-nvfp4-spark   spark-quant target weights (byte-identical both nodes)
  models/glm53-flash-nvfp4         non-spark NVFP4 target weights (199.4 GB, both nodes)
  models/glm53-flash-dflash2       DFlash2 draft (MXFP8, sha c033e03d…)
  cache/                           per-node JIT/kernel cache (container-owned)
  serve/TP2-DGX-Spark-GLM5.3F-Jovian-Judgement/
    glm53_pair_serve.sh            pair launcher (--check/--run/--verify/--down/--status/--logs)
    rank-{0,1}-{mtp3,df}-{spark,nvfp4}.env   cluster 1 (rank-0 files on r0, rank-1 on r1)
    rank-{2,3}-{mtp3,df}-{spark,nvfp4}.env   cluster 2 (rank-2 files on r2, rank-3 on r3)
  builder/                         image build system (r0 only)
```
(The retired devc646-era `rank-{2,3}-{mtp3,dflash2}.env` files still exist in
the repo's `env/` as history; the image behind them was deleted, they will
not boot.) The repo's `env/` copies are IP-scrubbed templates — **live host
files are ground truth**.

### Operations (from any machine with the SSH aliases)
```bash
./pairctl.sh 1 up mtp3-spark|mtp3-nvfp4|df-spark|df-nvfp4
./pairctl.sh 1 down / status / check / logs 0|1 [profile]
./pairctl.sh 2 …               # cluster 2, same verbs, same four profiles
PAIR_TS=1 ./pairctl.sh …       # remote mode: both ranks via the Tailscale aliases
PAIR_HEALTH_TIMEOUT=1900 ./pairctl.sh 1 up <p>   # first boot on a NEW image (cold JIT)
```
pairctl handles swappiness, GID auto-fix, drop_caches, stale-container
teardown, worker-first start order, and the health wait; its only launcher
passthrough is `--recurrent-checkpoint-policy request_boundaries` (KDA
pre-fill backend comes from the env file). It caches the sudo password in
`~/.pair-sudo`. Custom env variants (not one of the four profile names) are
booted via the launcher directly, worker first:
```bash
ssh -4 gx10-r1 'cd ~/builds/glm53-flash-dgx-spark-tp2/serve/TP2-DGX-Spark-GLM5.3F-Jovian-Judgement && bash glm53_pair_serve.sh --run rank-1-<variant>.env'
sleep 6
ssh -4 gx10-r0 'cd ~/builds/glm53-flash-dgx-spark-tp2/serve/TP2-DGX-Spark-GLM5.3F-Jovian-Judgement && bash glm53_pair_serve.sh --run rank-0-<variant>.env'
# then health-poll the LAN head:8000/health
```
Daily driver is MTP3 (`mtp3-spark` / `mtp3-nvfp4`); df profiles are the
coding option (draft CC BY-NC-ND, non-commercial).

### Building a new image (r0, cluster-1 lines)
```bash
cd ~/builds/glm53-flash-dgx-spark-tp2/builder/blackwell-llm-docker/dgx-spark-builder
bash build-spark-cu132.sh --dry-run build-glm53-head0906.env    # validates the 6 arm64 patches + pins
bash build-spark-cu132.sh build-glm53-head0906.env              # ~15 min warm / ~70 min cold (detached: setsid nohup)
# derive the -managed variant via Dockerfile.managed (one-line loader sed), then:
docker save local/vllm:glm53-flash-nvfp4-head0906-managed | ssh -4 <c1-r1-fabric-ip> docker load
```
Build profiles on r0: `build-glm53-head0906.env` (current line),
`build-glm53-devspark.env` + `build-glm53-devspark2.env` (previous lines),
`build-glm53-jovian.env` (the reference profile this line was A/B'd
against), `build-example.env` (documented template). Build with the cluster-1
pair DOWN (peak ~103 GiB). Never `docker builder prune` — that forces the
multi-hour toolchain recompile. Pins come from `git ls-remote <repo>
<branch>`; the current pair is vLLM `2a979314` + b12x `3eb57add`
(deliberately the matched cut — the later b12x line regressed decode in our
A/B). The retired devc646 line built on r2 (git mirrors + http.server 4443);
r0 owns all current lines.

### Testing
- Chat sanity first ("Count to 10") after every fresh boot — the SM121 loop bug's tell.
- Quick throughput: `--concurrency 1,2,4,8 --contexts 0,8192,32768,65536,131072 --max-tokens 2048 --duration 30 --coding-peak --coding-peak-runs 1 --no-hw-monitor --display-mode plain --output <name>.json`.
- **Judge C8 for throughput; C1 single-run is noisy (adaptive MTP).** Estonia LAST, and at `--profile-concurrency 1` (C4 times out on this hardware).

### Conventions
- One MD per run in `runs/` (config, numbers vs previous, takeaways). Raw
  bench json/txt artifacts stay local (gitignored) — the mirror carries the
  .md reports only.
- Update RECIPE.md when pins/settings change.
- Commit to `public-main` and push THAT branch (the GitHub mirror); never
  push `main` directly (private history).
- Don't touch NVIDIA Sync fabric state or netplan. No host reboots mid-build.
- Unified-memory rule: page cache counts against CUDA memory; pairctl drops
  page cache on every `up`. Streaming per-run results to a file (`-u` +
  stdout redirect) for live progress.
