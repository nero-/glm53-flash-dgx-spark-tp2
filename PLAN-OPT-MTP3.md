# Plan — MTP3 (`master`) decode-throughput optimizations

Written 2026-09-04 after the jj-master validation (run 05/06). Goal: raise
single-stream decode (28.8–29.2 t/s) without giving back the ~2k PP, until the
upstream update that carries the DFlash2 fix lands. Companion to
`PLAN-RUNS.md` (validation ladder) — this file is the *experiment menu*.

## Baseline (live, jj-master, `bench/glm53fd-jj-master-mtp3-.json`)

| metric | value | note |
|---|---|---|
| C1 decode @ ctx 0 / 8k / 32k | **28.8 / 29.2 / 28.4** | the target to move |
| C1 @ 64k / 128k | 23.5 / 26.8 | dips at long ctx |
| C8 aggregate | 104.5 (ctx0) · 88–95 (long ctx) | healthy |
| MTP steps/s C1 | ~11.8–13.2 | the real bottleneck |
| Accept/step | 2.2–2.45 | ≈ r22 contract (2.46–2.53) |
| PP 8k/32k/128k | 1,960 / 1,950 / 1,851 | ≈ qualified ceiling (~2k) |
| Coding peak | 35.4 | DFlash2 ref: 49.2 (R12) |

**Reading:** acceptance is already at the MTP3 structural ceiling (~2.5), so
C1 t/s ≈ steps/s × 2.5. Every lever is step-time (CPU overhead per step, GDN
decode kernel, TP comm latency) or planned/accepted length. Batch/KV/pin
won't move C1. PP is already at contract level — treat as done.

## Tier A — no rebuild (same image, env/CLI only) — TONIGHT

| # | What | Expected | Status |
|---|---|---|---|
| A1 | **Run 07 — dev profile** (`pairctl up dev`): `VLLM_USE_AOT_COMPILE=1`, `VLLM_USE_MEGA_AOT_ARTIFACT=1`, `VLLM_USE_V2_MODEL_RUNNER=1`, `--gdn-decode-kernel b12x` (GDN decode: B12X vs Triton fallback is direct step-time), `SSM_CONV_STATE_LAYOUT=DS`, OMP 16, `--mm-processor-cache-gb 0`, image limit 1. Same image as master; glm parsers (block-16 stays dropped). | C1 **+5–15%** → 30–33; PP neutral/positive | **GO** — sanity: chat + estonia ≥28/30 first |
| A2 | **Run 08 — speculation sweep** (master geometry): fx3 (adaptive off, fixed 3) vs fx2 (fixed 2) vs aw64 (adaptive window 64). Depth >3 impossible (3 MTP heads). Variant envs deployed on both nodes. | ±5–12%, winner per workload mix | **GO** — chat sanity + C1/coding slice per variant |
| A3 | **Run 09 — r22-qualified comm defaults** (master + `NCCL_MAX_NCHANNELS=16`, `NCCL_BUFFSIZE=2 MiB`, OMP=1; variant `match22` deployed). Instrument first if numbers move (NCCL_DEBUG=INFO window). | +0–5% C1 | **GO** — last; keep-or-revert on the slice bench |
| A4 | FlashInfer autotune ON | — | **DROPPED**: `--no-enable-flashinfer-autotune` is hard-coded in the launcher (line 810) AND is the developer's recommended set. Parked until upstream moves it |
| A5 | KV pin 7→7.5 GiB | — | **OUT** (John): live serving ~120.25 GiB/node → only ~1.2 GiB below the OOM edge; the +5% capacity route later is r22's memory-weighted allocation, not a bigger pin |

## Tier B — the local-inference-lab update (next build)

**Watch** (`git ls-remote`, on-demand):
- vLLM `dev/jovian-judgement`: still `a50ebee1` == our pin (2026-09-04)
- b12x master: still `9ae41c5c` == our pin
- Fix traffic currently on `agent/ii-b12x-*` / `agent/ii-allreduce-*` branches

Optional early build today: published R22 artifact line
(`artifact/jovian-judgement-community-20260904-r22-source` @ `70b3c1c7…`,
B12X `1e59a1fd…`) — FlashKDA (stable GDN prefill), memory-weighted KV
allocation (+5% capacity), 2,048-token pages, `FAIRNESS_ENGINE=compute_share`
w/ `PREFILL_SCHEDULE_INTERVAL=1`, qualified NCCL defaults. Decode-neutral per
upstream control (-0.3/-1.3%). Build only if we want stability/capacity pieces
early.

When the dev branch moves:
| # | Test | Expected |
|---|---|---|
| B1 | master requal on new pins | guard run-06 numbers; drift = pin regression |
| B2 | **DFlash2@7** (wedge fix): batch 4096, pin 9–10 GiB, 256k ctx | coding **+27–40% → 43–50**; C1 ≈/below MTP3 — the coding/long-gen profile |
| B3 | **MTP experts b12x vs marlin** A/B (if SM121 NVFP4 MTP path fixed upstream) | unknown; direct C1 lever if it wins |
| B4 | adopt upstream vitals (v2 runner/AOT defaults) into master env | fold A1 winners |

## Tier C — infra (parked)

Second CX7 cable between the nodes (each node's second port routes
10.100.81.x today): dual-rail RoCE for lower TP allreduce latency. Off the
qualified single-rail profile + fabric-state risk → **park until profiling
shows comm is a meaningful slice of step time.** NIC IRQ / engine-core
pinning: 0–3%. Both deferred.

## Explicit non-levers (don't spend bench time)

Bigger batch · bf16 KV · `CUDAGRAPH_MODE=FULL` (downgrades on GDN) ·
PCIE allreduce (intra-host only) · MTP depth 4+ (no 4th head) · single-node
replicas.

## Runbook — tonight (execute from `~/Agent/Builds/glm53-flash-dgx-spark-tp2/`)

Variant envs already on the hosts: `rank-{0,1}-fx3|fx2|aw64|match22.env`
(single-line deltas of today's master envs, current GIDs baked in).
Conventions: worker (r1) first, then head; preflight = swappiness + GID +
drop caches; bench = `bench/.venv` (rebuilt for Linux, smoke-tested).
Baseline to beat: C1 28.8/23.5 (ctx0/64k), coding 35.4, C8 104.5, PP ~1.96k.

```bash
cd ~/Agent/Builds/glm53-flash-dgx-spark-tp2

# ---- A1 / run-07: dev profile (full gate) ----
./pairctl.sh down
PAIR_SUDO_PASSWORD=$(cat ~/.pair-sudo) ./pairctl.sh up dev
# sanity FIRST: {"prompt":"Count to 100"} must be coherent (loop gate)
cd bench && ./.venv/bin/python -u llm_decode_bench.py --host 192.168.50.23 --port 8000 \
  --model zai-org/GLM-5.3-Flash --test-profile estonia --profile-concurrency 4 \
  --profile-runs 30 --completion-stats-seed 1000 --completion-stats-request-timeout 1000 \
  --no-hw-monitor --display-mode plain --output run-07-dev > run-07-dev.txt 2>&1
# then the slice:
./.venv/bin/python -u llm_decode_bench.py --host 192.168.50.23 --port 8000 \
  --model zai-org/GLM-5.3-Flash --concurrency 1,8 --contexts 0,65536 \
  --max-tokens 2048 --duration 30 --coding-peak --coding-peak-runs 3 \
  --coding-peak-max-tokens 65536 --no-hw-monitor --display-mode plain \
  --output run-07-dev-slice > run-07-dev-slice.txt 2>&1
cd ..
# write runs/run-07-*.md, then either keep dev env + extras or roll back:
./pairctl.sh down
```

```bash
# ---- A2 / run-08 + A3 / run-09: variant boots (custom env files) ----
# boot variant $V on both nodes (worker first); containers are port-based, so
# --down with any rank-N env filename of this family tears down the live one.
boot() { V=$1
  ssh -4 gx10-r1 "cd ~/builds/glm53-flash-dgx-spark-tp2/serve/TP2-DGX-Spark-GLM5.3F-Jovian-Judgement && \
    bash glm53_pair_serve.sh --down rank-1-master.env 2>/dev/null; \
    bash glm53_pair_serve.sh --run rank-1-$V.env"
  sleep 5
  ssh -4 gx10-r0 "cd ~/builds/glm53-flash-dgx-spark-tp2/serve/TP2-DGX-Spark-GLM5.3F-Jovian-Judgement && \
    bash glm53_pair_serve.sh --down rank-0-master.env 2>/dev/null; \
    bash glm53_pair_serve.sh --run rank-0-$V.env"
  for i in $(seq 1 36); do curl -fsS -m 3 http://192.168.50.23:8000/health && break; sleep 20; done
}
boot fx3    # fixed@3  → bench slice → runs/run-08-fx3.md
boot fx2    # fixed@2  → bench slice → runs/run-08-fx2.md
boot aw64   # adaptive w64 → bench slice → runs/run-08-aw64.md
boot match22 # NCCL 16ch + 2MiB + OMP1 → bench slice → runs/run-09-match.md
# slice once up:
bench/.venv/bin/python -u bench/llm_decode_bench.py --host 192.168.50.23 --port 8000 \
  --model zai-org/GLM-5.3-Flash --concurrency 1,8 --contexts 0,65536 \
  --max-tokens 2048 --duration 30 --coding-peak --coding-peak-runs 1 \
  --no-hw-monitor --display-mode plain --output <out> > <out>.txt 2>&1
# rollback at any time: ./pairctl.sh down && ./pairctl.sh up master
```

Notes:
- After a **reboot**, clone the variants again from the freshly GID-fixed
  master env (`pairctl up dev` or master auto-fixes) — the same `sed` lines
  as recorded in the boot log of each run report.
- Dev-profile vision limit is 1 image (`--limit-mm-per-prompt`); master stays
  4/1. For tonight's text decode/coding benches that's fine; requal vision on
  master after any promotion.
- If `up dev` fails to boot (AOT memory pressure or runner interactions),
  revert to master and log the failure in runs/ A1 slot.

**Realistic envelope:** C1 28.8 → **~31–34** env-only (+8–17%); DFlash2 adds a
**40–50** coding profile once the fix lands; C8 ~90–105; PP stays ~1.9–2k.
Single-stream MTP3 beyond ~3.5k tok/min is unlikely without upstream engine
work — acceptance is capped, so this plan captures the remaining step-time
headroom, then charters it explicitly.
