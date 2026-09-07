# Recipe — GLM-5.3-Flash on 2× DGX Spark (TP2)

**Current line: devspark2** (`local/vllm:glm53-flash-nvfp4-devspark2`, built
2026-09-07 on r0; qualification in progress — the previous line `devspark`
stays loaded on both nodes as rollback until devspark2 passes). One image
serves **both** GLM-5.3-Flash checkpoints on cluster 1; four profiles =
2 speculators × 2 quants:
- `mtp3-spark` / `mtp3-nvfp4` — MTP3 adaptive (1/3/32), daily driver, ctx
  512k, batch 8192. Spark quant: KV pin 9.5 GiB (10200547328; 1.5 GiB below the 11.0 that ran 120.6 used on r0) + marlin MTP
  experts; non-spark quant: pin 6.5 GiB (6979321856), MXFP8 MTP experts →
  humming (no override).
- `df-spark` / `df-nvfp4` — DFlash2@7 (draft
  `local-inference-lab/GLM-5.3-Flash-DFlash2`, MXFP8, CC BY-NC-ND), ctx 256k,
  batch 4096, split pages 4096/4096. Spark pin 12.5 GiB (13421772800);
  non-spark pin 4.5 GiB (4831838208) + video on.
  Model dirs: `models/glm53-flash-nvfp4-spark` (spark quant) and
  `models/glm53-flash-nvfp4` (non-spark, 199.4 GB on disk).
- MM numbers (keep across rebuilds): images 4 everywhere; videos 0 except
  `df-nvfp4` (1).
- pairctl (cluster 1): `./pairctl.sh 1 up mtp3-spark` (also
  `mtp3-nvfp4` / `df-spark` / `df-nvfp4`). Cluster 2 runs the separate
  **devc646** line with its own `mtp3` / `dflash2` profiles (see its section
  below) — leave cluster 2 alone during cluster-1 work.

## Hardware / topology
2× DGX Spark GB10 (arm64, SM 12.1, 121.7 GiB unified mem), TP2/DCP1. The
fabric is **NVIDIA-Sync-managed** (`/etc/netplan/99-nvidia-sync-cluster.yaml`
on both nodes — sync-owned, don't edit): both nodes use their f1 ports
(`rocep1s0f1`/`enp1s0f1np1`, IPs <c1-r0-fabric-ip> / <c1-r1-fabric-ip>), RoCEv2 GID
index 3 on both. pairctl re-checks the GID table on every `up` and
auto-fixes the env files after reboots. The second rail (<c1-fabric-subnet-2>,
`roceP2p1s0f1`) is unused — single-rail is the qualified profile.

## Image build (r0, ~15m warm / ~1h10m cold)
```bash
cd ~/builds/glm53-flash-dgx-spark-tp2/builder/blackwell-llm-docker/dgx-spark-builder
bash build-spark-cu132.sh --dry-run build-glm53-devspark2.env   # validates rewrites + pins
bash build-spark-cu132.sh build-glm53-devspark2.env             # build (detached: setsid nohup)
docker save local/vllm:glm53-flash-nvfp4-devspark2 | ssh -4 <c1-r1-fabric-ip> docker load
```
- Pins (devspark2, heads taken 2026-09-07): vLLM `local-inference-lab/vllm`
  `dev/jovian-judgement` @ `2a979314` — on top of the devspark pin `858b4912`
  (#667 gdn spec-decode fast path, #669 aligned hybrid cache reuse, #672
  Mamba-state recurrent-block resume, b12x-loader integration) it adds loader-
  owned GLM/KDA checkpoint weights (7d66922a), b12x loading for full GLM
  (6daf0036), instruction checkpoints for user-required chat templates
  (23771aab), #674 specialized scalar indices in checkpoint restore, #675
  leading instruction-checkpoint retain, #683 admit exact checkpoint hits
  during active decode, and **2a979314 "Fix loader-owned allocation for the
  GLM Flash MTP projection"** (see lesson 22 — this is what unblocks
  LOAD_FORMAT=b12x WITH MTP). b12x master @ `f46fee91` — on top of the devspark
  pin `79e228ec` (GB10 O_DIRECT loader as a vllm plugin: reads into pinned
  write-combined CUDA storage) it adds strided TP checkpoint slice loading
  (eb246263, the vLLM companion), the GLM/DeepSeek loader-fix merge
  (7e9361e4), MTP-projection allocation regression test (3eb57add), GLM-5.3
  MoE M1 FC2 row-pair overlap, MXFP8 GEMM fill-eliding, #302 W4A8 profiling
  oracle fix, and #246 "make TP2 graph peer-push generation-safe".
  PR 646 (fine-grained prefix hits + decoupled 2048/256 split blocks) was
  closed upstream UNMERGED and is deliberately not composed here (measured
  +1.8% KV/GiB on master only; the coupled large-page default already had the
  pool efficiency). The rank envs still carry `VLLM_GLM53_SPLIT_*` /
  `BLOCK_SIZE` 2048/256 — harmless-if-inert on this line, live with the
  devc646 build; the boot log `physical page sizes` census is the arbiter.
- Loader contract at these pins (b12x docs): `VLLM_PLUGINS=b12x_loader` +
  `--load-format b12x`; write-combined pinned storage is unconditional (no
  `allocation` loader option is accepted — vLLM 1259c74d).
- Toolchain unchanged from the r22b qualification (devspark → devspark2 diff
  is ONLY the two source pins): patched NCCL `dfab7c1a`
  (canonical/cu132-nccl2304), FlashInfer `1ac6942`, InstantTensor `49b4010`
  (dormant when booting the b12x loader; still the fallback path), torch
  2.13.0+cu132, cutlass `e6233cba`, humming-kernels 0.1.12 (MTP MXFP8 experts).
- The `dgx-spark-builder` wrapper is the arm64/SM121 retarget of upstream
  `blackwell-llm-docker` (6 patches). NOTE: the Dockerfile now uses FULL
  clones for vLLM (main + launcher) — GitHub 408s on `--filter=blob:none`
  lazy blob fetches killed several builds.
- Re-pin: `git ls-remote` the two repos, update `VLLM_PIN`/`B12X_PIN` in a new
  profile, rebuild. Profiles on r0: build-glm53-devspark.env (previous line),
  build-glm53-devspark2.env (current).

## Serve (from any machine with the SSH aliases; repo `glm53-flash-dgx-spark-tp2/`)
```bash
./pairctl.sh 1 up mtp3-spark   # cluster 1 (r0/r1); also: mtp3-nvfp4 / df-spark / df-nvfp4
./pairctl.sh 1 down            # status / check / logs 0|1 [profile] the same way
```
(pairctl's positional first arg picks the cluster; cluster 2 = `2 up mtp3` /
`2 up dflash2` on the devc646 line — leave it alone during cluster-1 work.
First-ever boot on a NEW image: `PAIR_HEALTH_TIMEOUT=1900 ./pairctl.sh 1 up …`
— cold JIT can exceed the default 720 s wait.)
pairctl handles swappiness, GID auto-fix, drop_caches, teardown, worker-first
start order, health wait, and passes `--kda-prefill-backend b12x
--recurrent-checkpoint-policy request_boundaries` as launcher passthrough
EXTRA on every boot. Env files live on hosts in
`~/builds/glm53-flash-dgx-spark-tp2/serve/TP2-DGX-Spark-GLM5.3F-Jovian-Judgement/`
(repo copies in `env/` are IP-scrubbed templates; hosts hold real IPs — the
**live host files are ground truth**). The host launcher that pairctl drives,
`glm53_pair_serve.sh`, ships in this repo at `serve/` — deploy it to the hosts'
serve dir.

## Key config (cluster 1, all profiles)
`LOAD_FORMAT=b12x` + `VLLM_PLUGINS=b12x_loader` (the b12x GB10 O_DIRECT
loader; the devspark era had to run `LOAD_FORMAT=instanttensor` because of
lesson 22 — reverted to b12x at devspark2) · mtp3 adaptive (1/3/32) or
dflash@7 · MTP experts: non-spark quant **MXFP8 → humming** (no
`MTP_MOE_BACKEND` override); spark quant **NVFP4 → `MTP_MOE_BACKEND=marlin`** ·
fp8 KV · B12X attention/MoE/linear · `KDA_PREFILL_BACKEND=flashkda` (the
`vllm/_flashkda_C.abi3.so` extension IS in the devspark2 image — RECIPE's old
"flashkda not present in the public builder" note is stale; `b12x` KDA prefill
fell back to a heuristic policy at this pin) · `--recurrent-checkpoint-policy
request_boundaries` · `--mamba-cache-mode align` · piecewise graphs · prefix
caching + chunked prefill ON · `VLLM_ENABLE_PCIE_ALLREDUCE=0` · LD_PRELOAD =
CUDA 13.2 compat shim + patched NCCL 2.30.4 ·
`VLLM_B12X_MOE_FP4_FORCE_A16=0`.

Reference-env settings adopted 2026-09-07 (from Luke's
testing_glm53rankenv.txt + glm53-jovian.env, A/B'd): `KDA_PREFILL_BACKEND=flashkda` (prefill 1.65–2.03k tok/s vs 1.3k with the b12x KDA heuristic),
`MM_PROCESSOR_CACHE_GB=0` (vLLM default 4 GiB per process of HOST unified
memory otherwise; frees ~8 GiB/node), `MM_ENCODER_TP_MODE=data` (each rank runs
the full encoder, no encoder collectives over RoCE), `PREFIX_MATCH_UNIT=256`,
`PREFILL_SCHEDULE_INTERVAL=1` (their C=1 profile), `ASYNC_SCHEDULING=1`
(CLI flag, store_true — `--async-scheduling` alone; passing a value errors),
`B12X_POLICY_MODE=auto` (b12x env var — no CLI flag), `COMPILATION_LEVEL=2`
(torch.compile `-O2`), and `MTP_MOE_BACKEND=humming` on the Spark quant too
(their accept ~2.5 vs marlin's ~2.3; the r21-era marlin note on spark is
superseded — lesson 6). **Reverted**: `BLOCK_SIZE=256` → 2048 (El8's
block-table unit 256 measured smaller pool: 1,326,879 vs 1,466,929 tokens
at the same pin), `VLLM_GDN_DECODE_KERNEL` (unset — engine default `cuda`),
MTP5 → **MTP3** (acceptance 0.853/0.581/0/0/0 — depth 2–3 real; MTP5 a
wash + 2 GiB graph memory), **AOT** (not in image — no mega artifact),
`--chat-template` high-reasoning (NOT copied — it changes model behavior;
stock template defaults reasoning_effort to `max`).
KV pins and MM caps keep the project's own numbers; mtp3-spark now 10.5 GiB
(11274289152 — managed loader boots at 118 GiB used, +1 GiB affordable).
Boot log: `physical page sizes [1148928, 2244608]`,
`GPU KV cache size: 1,466,929 tokens` (2.80× a 512k request).
Loader: the `-managed` derivative image (one-line sed in
b12x/integration/vllm/loader.py `pinned_wc` → `managed`; see
builder/dgx-spark-builder/ Dockerfile.managed note) — without it b12x
loading costs ~36% of prefill.
Template sync 2026-09-07: Spark snapshot carried an older chat_template.jinja
than current HF main (16ffa9ab); synced from the NVFP4 copy on both nodes.

Memory envelope (cluster 1, non-spark quant): worker ≈ 117 GiB used at a
6.5 GiB KV pin; **the head runs ~2.5 GiB heavier** (API + engine + MM
encoder reservation) ≈ 119 GiB. The OOM cliff is ~121.5 GiB — a 9 GiB pin
OOM'd the head on the NON-spark weights. Current pins: mtp3-nvfp4 6.5 /
df-nvfp4 4.5 (draft weights + draft KV groups + video reservation);
the spark quant leaves more room, so mtp3-spark 11.0 / df-spark 12.5.
Move up only in small steps and watch `free -g` on r0. The real KV pool is the
boot log line `GPU KV cache size: N tokens`.

## Benchmarks (llm-inference-bench, from `bench/`)
Estonia needle (30 runs, C4, seed 1000, 1000s request timeout):
```bash
cd bench
./.venv/bin/python llm_decode_bench.py --host [IP] --port 8000 \
  --model zai-org/GLM-5.3-Flash --test-profile estonia \
  --profile-concurrency 4 --profile-runs 30 --completion-stats-seed 1000 \
  --completion-stats-request-timeout 1000 --no-hw-monitor \
  --output <name>.json > <name>.txt 2>&1
```
Full throughput suite: `--concurrency 1,2,3,4,8 --contexts 0,8192,32768,65536
--prefill-contexts 8k,32k,64k,128k --max-tokens 2048 --duration 30
--coding-peak --coding-peak-runs 3 --coding-peak-max-tokens 65536
--kv-budget <pool>`.

Bench caveats (learned the hard way):
- **Linux venv**: the checked-in `bench/.venv` is macOS-only. Rebuild once:
  `python3 -m venv bench/.venv && bench/.venv/bin/pip install httpx rich`.
- **C1 single-run is noisy** with adaptive MTP — a 30 s cell is ~one adaptive
  window (32). Judge C8 (aggregate), not C1. The bench's "(accept len)" column
  is also unreliable (server metrics under-report draft acceptance as 0.0%).
- **Estonia times out at C4 on this hardware**: long-ctx decode is KV-bound
  (~18 tok/s aggregate at C4×166k) and the model reasons ~17.5k tokens, so the
  1000 s wall-clock expires with no answer (0/10 PASS, all TIMEOUT, 0 wrong —
  not corruption). Run estonia at `--profile-concurrency 1` or raise
  `--completion-stats-request-timeout`.
- **The bench's "KV cache budget" over-reports capacity**: it shows
  `num_gpu_blocks × block_size` (242 × 4608 ≈ 1.1M) which includes
  mamba-alignment padding. The real usable KV is the boot log / metrics
  `GPU KV cache size` (e.g. **1,235,254** at the 9 GiB pin). Don't chase
  "1.1M KV" — it's a display artifact, not a real gain.

## Lessons learned (condensed, current)
1. **RoCE GID indices shift after reboots/recables** → pairctl re-checks +
   auto-fixes on every `up`.
2. **Memory: the head node binds** (~2.5 GiB heavier than the worker). The
   9 GiB KV pin OOM'd the head on the non-spark quant; current pins: mtp3-nvfp4
   6.5 / df-nvfp4 4.5 / mtp3-spark 11.0 / df-spark 12.5 GiB. A
   wedged host after an OOM needs a physical power-cycle.
3. **After cable swaps / switch experiments: reboot both nodes if fabric
   throughput looks capped** (a warm reboot cleared a stuck CX7 state that
   limited a 200G link to ~26 Gbps with zero errors).
4. **GitHub 408s on partial-clone blob fetches** are recurring on the
   builder — the Dockerfile now uses full clones for vLLM. A build dying
   with `could not fetch … from promisor remote` = another
   `--filter=blob:none` line to remove.
5. **The DFlash2 draft is CC BY-NC-ND (non-commercial)** — MTP3 is the
   commercial path.
6. **Locklock lineage**: fixed by b12x `9ae41c5c` ("wait for persistent
   epilogue stores"); every later b12x pin includes it. If loops return,
   suspect a stale b12x pin first. (The r21-era marlin-MTP workaround is
   obsolete — MTP experts are MXFP8 → humming on this line.)
7. **Adaptive MTP > fixed MTP** at long context — keep adaptive (window 32).
8. **First request after boot is JIT-slow** — not a hang.
9. **The bench's "KV cache budget" over-reports** (`num_gpu_blocks ×
   block_size` display artifact) — the boot log `GPU KV cache size` is the
   real pool.
10. **lil** (`local-inference-lab/lil`) is built and installed on r0
    (`~/.local/bin/lil`) as a typed launcher alternative; the HF catalog
    carries `GLM-5.3-Flash-NVFP4`. pairctl remains the operator default
    until a lil topology is qualified.
18. **The r25-line engine refuses to start unpinned on a 512k profile**
    (`_check_enough_kv_cache_memory`): at util 0.85 the profiling run left
    ~0.2–0.8 GiB for KV and the new guard demands ≥3.75 GiB to host one
    512k-token request → `ValueError … (3.75 GiB KV cache)`. **Always set
    `KV_CACHE_MEMORY_BYTES` on every profile** (byte pin skips memory
    profiling and makes `gpu_memory_utilization` irrelevant); 8 GiB measured
    stable with vision on the devc646 line (r2 steady 119/2 GiB, r3 114/7;
    survived the full 20-cell matrix at 120 used / 1 avail). Never exceed
    the r22 9-GiB-with-vision cliff.
19. **`resolve_kda_prefill_backend` never auto-selects b12x** — on the
    r25/devc646 line pass `--kda-prefill-backend b12x` explicitly. R25's
    default (flashkda) is not present in the public Spark builder (its
    extension is a private R-artifact); b12x is built by the wrapper and
    works (mtp3 + dflash2 boots verified, no `locklock`).
20. **r2 (and likely any node) container builds flake on github.com**
    (135 s connect timeouts twice today). Mitigation on r2: bare mirrors of
    b12x/lmcache under `~/builds/.git-mirrors/` served by
    `python3 -m http.server 4443 --bind 0.0.0.0` (nohup), reachable as
    `http://172.17.0.1:4443/<name>.git` from the BuildKit sandbox. Dumb-http
    mirrors cannot serve `git fetch origin <40hex>` — pass upstream-pin
    **branches** (`B12X_REF`) and keep the exact commit as `B12X_PIN` (it is
    still checked). vLLM stayed on the nero- fork URL because its wheel
    stage was already cached (changing B12X/LMCache args DOES invalidate
    downstream stages — observed).
21. **PR #646 geometry is picked up by plain env, no patching**: the
    launcher's `--block-size 2048` + the split env pair
    `VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=256`) produce the
    `Using split GLM-5.3 cache pages … / physical page sizes [1148928,
    2244608]` at boot and the pool math follows (139.8k tokens/GiB at the
    8 GiB pin, 2.13× concurrency @512k). `--prefix-match-unit 256` rides
    along for the one-exchange top-k; no `Disabling fine-grained…` lines
    in a healthy boot. (Measured on the devc646 build, which composed the
     unmerged #646 tip; on the plain-jj devspark line the split env vars are
     inert unless the census appears in the boot log.)
 22. **b12x loader × MTP `eh_proj` allocation bug (devspark era)**: the
     devspark line was *composed* for `LOAD_FORMAT=b12x` but could not
     actually boot that way with MTP — the GLM Flash MTP projection's plain
     `nn.Linear` weight was unmarked as loader-owned (ordinary CUDA storage)
     and **b12x rejected the eh_proj checkpoint copy**. Every devspark boot
     therefore ran `LOAD_FORMAT=instanttensor` (what the live envs showed)
     while this doc's Key config still said b12x — docs were aspirational,
     envs honest. Fixed upstream by vLLM `2a979314` + b12x `3eb57add`
     (regression test) / `eb246263` (strided TP slices); **devspark2 boots
     the b12x loader with MTP for real** (`VLLM_PLUGINS=b12x_loader`).
     Rule: the live host env files are ground truth over this file's
     config claims; if they disagree, check the pins before believing
     either.

## Second cluster + the `devc646` line (2026-09-05)
The second pair — **gx10-r2** (head/rank0, LAN <r2-lan-ip>, fabric <c2-r0-fabric-ip>) and
**gx10-r3** (worker/rank1, fabric <c2-r1-fabric-ip>) — serves the
`glm53-flash-nvfp4-devc646` image. It is used as the redeploy target for the newest
software so cluster 1 (r0/r1, r22b master) can keep daily driving without
software churn. Build on **r2 only**; state on r0/r1 stays read-only (the agent's
own runtime rides inside those containers — never cycle them from tooling).

- **Line**: vLLM = `dev/jovian-judgement` @ `a8c796f3` (morning perf work: #649
  one-exchange top-k, GLM5Next LM-head/metadata reuse, B12X sparse-MLA cache fix,
  b12x dense-activation envs) + **PR #646 tip `b4f16bd3`** (fine-grained SWA/EAGLE
  prefix hits + decoupled GLM-5.3 target/recurrent blocks). Frozen as
  nero-/vllm branch `osd2/spark-glm53-dev-c646` (= `b4f16bd3…`) and
  nero-/b12x branch `osd2/spark-glm53-master-20260905` (= `b58f34ea…`). LMCache =
  builder-default `release/v0.5.2-glm52-dcp-base` (the upstream main builder's
  pytest gate needs its own test-file set; the R25 artifact branch lacks one).
  Version string: `0.26.1rc0+glm53.flash.nvfp4.devc646`.
- **Build profile** `builder/dgx-spark-builder/build-glm53-devc646.env`; wrapper
  `build-spark-cu132.sh` (build on r2: `~/builds/.git-mirrors` + http.server
  4443 feed B12X; vLLM from the nero- fork URL; image 24.9 GB, ~4.5 min warm /
  ~1h cold, peak 75.5 GiB).
- **Deployment**: `python3 -m http.server 4443` must be up on r2 for repeat builds;
  image moves to r3 via `docker save | ssh r3 docker load` over the fabric
  (~61 s at 108.9 Gb/s RDMA). SERVE_DIR = `~/builds/glm53-flash-dgx-spark-tp2/
  serve/TP2-DGX-Spark-GLM5.3F-Jovian-Judgement/` on both; container names
  `glm53-flash-r0` (r2) / `glm53-flash-r1` (r3).
- **Profiles**: only two, both multimodal (images 4 + video 1, LMO 0):
  `mtp3` (MTP adaptive 1/3/32, batch 8192, ctx 524,288) and `dflash2` (DFlash2@7,
  batch 4096, ctx 262,144). `pairctl.sh` grew a `PAIR_CLUSTER=2` flag that remaps
  hosts/files (`gx10-r2`/`gx10-r3`, `rank-2-`/`rank-3-`) and honors
  `PAIR_HEALTH_TIMEOUT` (first-ever boot on a new image = raise to 1900).
- **Geometry** (runs/run-15/16 for measurements): `--block-size 2048`,
  `--prefix-match-unit 256`, `VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=2048`,
  `VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=256` → the PR #646 page census
  (mtp3 pages [1,148,928, 2,244,608]; dflash2 [1,148,928, 2,342,912] + 5 DFlash
  draft KV layers in 1 independent native-block group). KDA prefill =
  `--kda-prefill-backend b12x` (explicit; flashkda is not buildable here).
- **KV pins measured**: mtp3 = 8,589,934,592 (8 GiB) → pool **1,118,600 tokens**
  (2.13× @512k; steady 119/2 GiB used on r2, 114/7 on r3; full matrix peak 120
  used / 1 avail — do not exceed). dflash2 = same 8 GiB pin → **523,009 tokens**
  (2.00× @262k). UNPINNED boots on this line crash with
  `_check_enough_kv_cache_memory` (`To serve at least one request with the model's
  max seq len … 3.75 GiB KV cache`) — the pin is mandatory, and while pinned
  `gpu_memory_utilization` is effectively ignored.
- **Results vs r22/r22b** (runs/run-15, run-16): prefill restored to
  ~1.9-2.05k tok/s (fabric: r2↔r3 = 108.9 Gb/s measured); mtp3: C1 31.3-33.5,
  C8-ctx0 101.1, coding 34.2 (−5.5% vs r22); dflash2: C1 ~30.3, coding peak
  **49.3** (+4.2% vs r22), fine-grained replay 52k cold→warm 115 s → 1.38 s.

## Fabric re-cluster handover (2026-09-05 — DONE)
The sync redo completed: the fabric is now sync-managed on both nodes
(see topology above). The old runtime-`ip addr add` scheme is gone; no
operator action needed on fabric unless recabling again.


## Second cluster — the devc646 quick start
```bash
# build (r2 only; needs the git mirrors + http.server 4443 running there)
cd ~/builds/glm53-flash-dgx-spark-tp2/builder/blackwell-llm-docker/dgx-spark-builder
bash build-spark-cu132.sh build-glm53-devc646.env
docker save local/vllm:glm53-flash-nvfp4-devc646 | ssh -4 gx10-r3 docker load
# serve
PAIR_CLUSTER=2 PAIR_HEALTH_TIMEOUT=1900 bash pairctl.sh up mtp3     # or dflash2
PAIR_CLUSTER=2 bash pairctl.sh --verify   # then: glm53_pair_serve.sh --verify rank-2-mtp3.env on r2
```
- Pins: vLLM pin `b4f16bd3…` (nero-/vllm branch `osd2/spark-glm53-dev-c646`),
  b12x `b58f34ea…` (nero-/b12x branch `osd2/spark-glm53-master-20260905`,
  fetched as branch — dumb-http mirrors can't serve a sha fetch), LMCache
  builder-default glm52-dcp.3. Toolchain = r22b set. Image 24.9 GB,
  `0.26.1rc0+glm53.flash.nvfp4.devc646`.

## Files

- `pairctl.sh` — one-command pair control (`[1|2] up/down/status/logs/check`;
  profiles mtp3-spark/df-spark/mtp3-nvfp4/df-nvfp4 on cluster 1).
- `serve/glm53_pair_serve.sh` — the host launcher (deploy to both nodes).
- `builder/dgx-spark-builder/` — the arm64/SM121 image-builder wrapper + build
  profiles (`build-glm53-devspark2.env` current, `build-glm53-devspark.env`
  previous line; `build-example.env` is the documented template).
- `env/rank-{0,1}-{mtp3,df}-{spark,nvfp4}.env` — the four cluster-1 profile
  templates (live host files are ground truth).
- `env/rank-{2,3}-{mtp3,dflash2}.env` — the cluster-2 devc646 templates (see
  its section).
- `OPS-GUIDE.md` — operator guide (start, API, OWUI/Hermes/DSH, bench).
- `runs/` — per-run reports; `bench/` — llm_decode_bench.py + results.
