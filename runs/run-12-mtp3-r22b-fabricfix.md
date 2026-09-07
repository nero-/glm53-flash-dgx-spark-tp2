# Run 12 — r22b MTP3 after fabric repair (IPv4LL fix; switch still in path)

Date: 2026-09-05 · Image: `local/vllm:glm53-flash-nvfp4-r22b` · master profile
(MTP3 adaptive 1/3/32, batch 8192, KV pin 9 GiB = 1,235,254 tokens, no video).

Context: after the 2026-09-05 recable (r0 moved to f1 ports, new NADDOD
QSFP56 DACs, management ethernets onto a switch), the pair hung at NCCL init.
Two distinct problems found and diagnosed:

1. **NCCL init hang — FIXED (software).** r0's new fabric port carried a
   stray link-local IPv4 (169.254.78.62/16) listed BEFORE the real
   10.100.80.2/24. NCCL advertises the interface's first IPv4; r1 tried to
   reach the link-local via the default gateway → 80 min timeout
   (`ncclOsSocketPollConnect … Connection timed out`; r0 waits forever).
   Fix: `ip -4 addr flush` + re-add only 10.100.80.2/24 (both f1 ports on
   r0). GID index shifted 5→3 (pairctl auto-fixed). Boot now healthy in
   205 s; NCCL init ~22 s. See RECIPE lesson 17.
2. **Fabric bandwidth collapse — NOT fixed (physical).** RDMA write 1.0–1.6
   GB/s single-QP, NCCL 64 MiB allreduce **2.78 GB/s**, iperf3 TCP **15.8
   Gbps** on a 200G link (≈ 1/13 line rate), identical on both rails, zero
   NIC drops/CRC/FEC errors, clean cable EEPROMs. The new **ethernet switch
   in the fabric path** cannot forward 200G (its switching fabric is the
   bottleneck; ~16 Gbps ≈ 10–25G-class silicon). RoCE survives (latency OK,
   no loss) but bandwidth is crippled. Fix = physical: direct QSFP56
   machine-to-machine (also required for NVIDIA Sync cluster), or a real
   200G switch with ≥200G switching fabric (+PFC/ECN for RoCE).

## Results (degraded-fabric r22b MTP3, master)

| ctx \ conc | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| 0 | 25.9 | 42.8 | 48.8 | **96.4** |
| 8k | 30.0 | 48.8 | 61.1 | 54.7* |
| 32k | ∅* | 47.7 | 65.6 | 94.1 |
| 64k | ∅* | ∅* | 57.6 | 91.9 |
| 128k | ∅* | ∅* | 50.5 | 88.9 |

\* ∅ = cell produced no measurable output: with prefill at ~1075 tok/s
(below), a 32k–128k prefill (30–120 s) exceeds the 30 s cell window at
C1/C2 (prefix cache cold; every cell re-prefills). Not a serving bug —
a direct consequence of the fabric throttle.

- **Coding peak: 36.1 tok/s** (3/3, mean; max 36.5) — **par with r22
  baseline 36.7** (jj-master 35.4). Decode is latency-bound, so the broken
  fabric barely touches it.
- **Prefill: 1,037–1,080 tok/s** at 8k–128k vs **~1,950** pre-recable.
  Prefill is TP2-allreduce-bound (needs ~3+ GB/s NCCL; fabric delivers
  2.78). This is the measurable cost of the switch.
- C8 88.9–96.4 vs run-10 r22 104–131. Part of that gap is the fabric;
  part is r22b's corrected b12x kernels (run-10's numbers were measured on
  the b12x contract pin that lacks the locklock fix — possibly inflated,
  see run-10 footnote).
- Acceptance lengths 2.06–2.76 (steps/s table in JSON) — adaptive MTP3
  behaving normally.

## Verdict
Serving is **up and correct** (no locklock, quality fine, coding peak par).
Prefill and long-ctx C1/C2 throughput are capped by the switch in the fabric
path. To restore: remove the switch (direct QSFP56 link) and optionally redo
the NVIDIA Sync cluster — that also replaces the runtime-IP hacks with a
persistent config (kills the lesson-17 IPv4LL hazard class permanently).

Numbers here are the honest degraded-fabric reference; re-bench after
recable.
