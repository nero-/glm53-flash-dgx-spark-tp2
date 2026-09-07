# Agent brief — starting a NEW agent session for GLM-5.3-Flash pair work

Everything a fresh agent needs to know about the DGX Spark pair, the hosts,
the tooling, and the conventions — so you (John) don't have to re-explain.

## Paste-in prompt template (for a new agent session)

Copy the block below into a new session. Replace `<LINK>` with the repo/link
or task description; delete sections that don't apply.

---

You are helping me with my GLM-5.3-Flash-NVFP4-Spark setup on two NVIDIA
DGX Spark (GB10, arm64/SM12.1, 121.7 GiB unified memory each) running as one
TP2/DCP1 pair over a direct ConnectX-7 RoCE link.

Context and conventions (read these first — they are kept current):
- `~/Agent/Builds/glm53-flash-dgx-spark-tp2/AGENT-BRIEF.md` — this brief
- `~/Agent/Builds/glm53-flash-dgx-spark-tp2/RECIPE.md` — current recipe, pins, and the bug log (authoritative for lessons)
- `~/Agent/Builds/glm53-flash-dgx-spark-tp2/OPS-GUIDE.md` — operator guide
- `~/Agent/Builds/glm53-flash-dgx-spark-tp2/PLAN-RUNS.md`, `PLAN-OPT-MTP3.md` — validation + optimization plans
- `~/Agent/Builds/glm53-flash-dgx-spark-tp2/runs/` — per-run reports (INDEX.md lists the series; run-15/16 = devc646 cluster-2 baselines)

Hardware/access (sudo password in the SSH info file):
- `ssh gx10-r0` (API head, rank 0) / `ssh gx10-r1` (headless worker, rank 1), user `nero`.
- `AddressFamily inet` is baked into `~/.ssh/config` (the `-4` mDNS/IPv6 quirk).
  Under the DSH/agent sandbox ssh may fail with "Bad owner … /etc/ssh/ssh_config.d/…" — use `ssh -F ~/.ssh/config` (pairctl already does).
- sudo: `echo <password> | sudo -S <cmd>` (password in `~/Agent/Builds/Cluster gx10-r0-r1 SSH info.txt`).
- Fabric: BOTH nodes use their f1 ports — r0 `rocep1s0f1` GID 3 @ 10.100.80.2, r1 `rocep1s0f1` GID 3 @ 10.100.80.1 (normalized 2026-09-05 recable/sync). GID table SHIFTS after reboots — re-read `show_gids`, let pairctl auto-fix. Swappiness must be 0.
- SECOND CLUSTER: `ssh gx10-r2` (head/rank0, LAN 192.168.50.90, fabric 10.100.120.2, GID v2 idx 3 on rocep1s0f1) / `ssh gx10-r3` (worker/rank1, LAN 192.168.50.5, fabric 10.100.120.1, GID 3). Runs image `local/vllm:glm53-flash-nvfp4-devc646` (r25-line + PR #646 = dev/jovian-judgement a8c796f3 + b4f16bd3; pins frozen on nero-/vllm `osd2/spark-glm53-dev-c646` + nero-/b12x `osd2/spark-glm53-master-20260905`). Control: `PAIR_CLUSTER=2 bash pairctl.sh up mtp3|dflash2` (hosts/`rank-2-`/`rank-3-` remapped). **KV pin is MANDATORY on this line** (`KV_CACHE_MEMORY_BYTES`; engine refuses unpinned ≥262k profiles; pin mode ignores `gpu_memory_utilization`). Measured pins: mtp3 8,589,934,592 (→1,118,600 tok; steady ~119/2 GiB free on r2), dflash2 8,589,934,592 (→523,009 tok). KDA prefill = explicit `--kda-prefill-backend b12x`. Builds happen on r2 only (github flaky from containers → `~/builds/.git-mirrors` + `python3 -m http.server 4443` on r2; B12X_REF as a BRANCH name).
- r0/r1 container lifecycle: the DSH/agent runtime used to ride INSIDE the r0-r1 container pair (2026-09-05 incident); as of 2026-09-07 agent sessions run on the operator laptop (p1) and MAY cycle cluster 1 when the task calls for it (builds, qualification) — confirm with the operator first, because API consumers (OWUI/Hermes/DSH custom provider) lose their model while the pair is down. Cluster 2 (`glm53-flash-r0/-r1` containers on r2/r3) is NOT to be cycled from cluster-1 sessions.
- r22's Lesson run-12/13 = the degraded-era references; run-15/16 = the devc646 cluster-2 baselines (prefill ~2.0k restored, mtp3 C8 101/82.7-96.7, dflash2 coding 49.3, fine-grained 52k-replay 115 s → 1.38 s).
- Host layout: `~/builds/glm53-flash-dgx-spark-tp2/{models, cache, serve/TP2-DGX-Spark-GLM5.3F-Jovian-Judgement, builder (r0 only)}`.
- Current image: `local/vllm:glm53-flash-nvfp4-devspark2` (2026-09-07: vLLM `dev/jovian-judgement` @ `2a979314` + b12x @ `f46fee91`, b12x loader WITH MTP; previous line `-devspark` on both nodes as rollback until devspark2 qualifies). Profiles (all cluster 1, `./pairctl.sh 1 up <p>`): `mtp3-spark` (MTP3 adaptive 8192/11.0GiB/512k, marlin MTP) · `mtp3-nvfp4` (8192/6.5GiB, humming) · `df-spark` (DFlash2@7 4096/12.5GiB/256k) · `df-nvfp4` (4096/4.5GiB + video).
- Bench: `bench/.venv/bin/python llm_decode_bench.py`. On Linux rebuild the venv once: `python3 -m venv bench/.venv && bench/.venv/bin/pip install httpx rich`.
- One MD per run in `runs/`, update RECIPE/PLAN, commit (never push `main` directly — `public-main` branch).
- Key lessons (full list in RECIPE bug log): SM121 NVFP4 MoE loops (fixed; `MTP_MOE_BACKEND=marlin`), dflash2 wedge FIXED on r22, **adaptive MTP > fixed MTP**, **RoCEnante hurts TP2 C1 (reverted)**, dev memory win not isolable, **estonia times out at C4 (run C1 or raise timeout)**, **C1 single-run is noisy (judge C8)**.

My task for you: <LINK / description>
---

## The condensed brief (the facts the template references)

### Access
| Host | Role | LAN IP | Fabric IP | RDMA (both PCIe funcs) | GID (v2) |
|---|---|---|---|---|---|
| gx10-r0 | API head, rank 0 | 192.168.50.23 | 10.100.80.2 | rocep1s0f1 (serving) + roceP2p1s0f1 (idle) | 3 |
| gx10-r1 | worker, rank 1 | 192.168.50.192 | 10.100.80.1 | rocep1s0f1 + roceP2p1s0f1 | 3 |
| gx10-r2 | second cluster: API head, rank 0 | 192.168.50.90 | 10.100.120.2 | rocep1s0f1 + roceP2p1s0f1 | 3 |
| gx10-r3 | second cluster: worker, rank 1 | 192.168.50.5 | 10.100.120.1 | rocep1s0f1 + roceP2p1s0f1 | 3 |
`ssh gx10-r0` / `ssh gx10-r1` (user nero, key auth). Sudo password NOT stored
here — see `~/Agent/Builds/Cluster gx10-r0-r1 SSH info.txt`; use `echo <pw> | sudo -S <cmd>`.
Node-to-node SSH works over the fabric (`ssh 10.100.80.1` from r0).

### Host layout (both nodes unless noted)
```
~/builds/glm53-flash-dgx-spark-tp2/
  models/glm53-flash-nvfp4-spark   spark-quant target weights (byte-identical both nodes)
  models/glm53-flash-nvfp4         non-spark NVFP4 target weights (199.4 GB, both nodes)
  models/glm53-flash-dflash2       DFlash2 draft (MXFP8, sha c033e03d…)
  cache/                           per-node JIT/kernel cache (container-owned)
  serve/TP2-DGX-Spark-GLM5.3F-Jovian-Judgement/
    glm53_pair_serve.sh            pair launcher (--check/--run/--verify/--down/--status/--logs)
    rank-0-mtp3-spark.env / rank-1-mtp3-spark.env   MTP3 spark (8192/11.0GiB/512k, marlin MTP)
    rank-0-mtp3-nvfp4.env / rank-1-mtp3-nvfp4.env   MTP3 non-spark (8192/6.5GiB/512k, humming)
    rank-0-df-spark.env   / rank-1-df-spark.env     DFlash2@7 spark (4096/12.5GiB/256k)
    rank-0-df-nvfp4.env   / rank-1-df-nvfp4.env     DFlash2@7 non-spark (4096/4.5GiB/256k, video)
  builder/                         image build system (r0 only)
```
rank-0-* files live on r0, rank-1-* on r1 (same directory). The repo's `env/`
copies are IP-scrubbed templates — **live host files are ground truth**.

### Operations (from any machine with the SSH aliases)
```bash
./pairctl.sh 1 up mtp3-spark|mtp3-nvfp4|df-spark|df-nvfp4   # preflight -> GID fix -> caches -> worker -> head -> health wait
./pairctl.sh 1 down [profile] / 1 status / 1 check / 1 logs 0|1 [profile]
```
pairctl caches the sudo password in `~/.pair-sudo`. Custom env variants (not
one of the four profile names) are booted via the launcher directly:
```bash
ssh gx10-r1 'cd ~/builds/glm53-flash-dgx-spark-tp2/serve/TP2-DGX-Spark-GLM5.3F-Jovian-Judgement && bash glm53_pair_serve.sh --run rank-1-<variant>.env'
sleep 6
ssh gx10-r0 'cd ~/builds/glm53-flash-dgx-spark-tp2/serve/TP2-DGX-Spark-GLM5.3F-Jovian-Judgement && bash glm53_pair_serve.sh --run rank-0-<variant>.env'
# then health-poll http://192.168.50.23:8000/health (worker first, then head)
```

### Building a new image (r0, cluster-1 lines)
```bash
cd ~/builds/glm53-flash-dgx-spark-tp2/builder/blackwell-llm-docker/dgx-spark-builder
bash build-spark-cu132.sh --dry-run build-glm53-devspark2.env   # validates the 6 arm64 patches + pins resolve
bash build-spark-cu132.sh build-glm53-devspark2.env             # ~15 min warm / ~70 min cold (detach: setsid nohup)
docker save local/vllm:glm53-flash-nvfp4-devspark2 | ssh -4 10.100.80.1 docker load
```
Pins from `git ls-remote <repo> <branch>`. Current (devspark2, 2026-09-07):
vLLM `dev/jovian-judgement` @ `2a979314`, b12x master @ `f46fee91`; toolchain
unchanged from the r22b set. Build with the cluster-1 pair DOWN (peak
~103 GiB; the pair is idle anyway while serving devspark). Never
`docker builder prune` — that forces the multi-hour toolchain recompile.
The devc646 line builds on **r2** instead (github flaky from r2 containers →
`~/builds/.git-mirrors` + `python3 -m http.server 4443` on r2; B12X_REF as a
BRANCH name) — that line is parked; leave cluster 2 alone.

### Testing
- Chat sanity first ("Count to 10") after every fresh boot — the SM121 loop bug's tell.
- Quick throughput: `--concurrency 1,2,4,8 --contexts 0,8192,32768,65536,131072 --max-tokens 2048 --duration 30 --coding-peak --coding-peak-runs 1 --no-hw-monitor --display-mode plain --output <name>.json`.
- **Judge C8 for throughput; C1 single-run is noisy (adaptive MTP).** Estonia LAST, and at `--profile-concurrency 1` (C4 times out on this hardware).

### Conventions
- One MD per run in `runs/` (config, numbers vs previous, takeaways).
- Daily driver is MTP3 (`mtp3-spark` / `mtp3-nvfp4`); df profiles are the coding option (draft CC BY-NC-ND).
- Update RECIPE.md when pins/settings change; commit + push the GitHub snapshot (`public-main` → origin main).
- Don't touch NVIDIA Sync fabric state or netplan. No host reboots mid-build.
