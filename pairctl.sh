#!/usr/bin/env bash
# pairctl.sh -- one-command control of the GLM-5.3-Flash TP2 pairs.
#
#   ./pairctl.sh [1|2] up mtp3|df     bring a pair up (worker first, then head)
#   ./pairctl.sh [1|2] down          stop both ranks
#   ./pairctl.sh [1|2] status|check  show container state / validate env files
#   ./pairctl.sh [1|2] logs 0|1      follow logs
#
# Clusters (default 1): 1 = gx10-r0/r1, 2 = gx10-r2/r3.
# Profiles (same names on both clusters):
#   mtp3 = rank-*-mtp3.env  MTP3 adaptive (1/3/32), humming MTP experts
#   df   = rank-*-df.env    DFlash2@7 (draft CC BY-NC-ND, non-commercial)
#
# Handles for you: swappiness fix, RoCE GID re-check (+auto-fix of the env
# files after reboots), page-cache drop, stale-container teardown, start
# order (worker before head), and a health wait with the qualified markers.
#
# sudo password: read once from $PAIR_SUDO_PASSWORD, else cached in
# ~/.pair-sudo (chmod 600, created on first prompt). Never stored in git.

set -euo pipefail

# Cluster selection: positional first arg (./pairctl.sh 2 up mtp3) or
# PAIR_CLUSTER env (default 1) -> gx10-r0/r1 (files rank-0/1-);
# cluster 2 -> gx10-r2 (head, files rank-2-)/gx10-r3 (worker, rank-3-).
if [ "${1:-}" = 1 ] || [ "${1:-}" = 2 ]; then
  PAIR_CLUSTER=$1
  shift
fi
PAIR_NUM="${PAIR_CLUSTER:-1}"
# Remote mode: PAIR_TS=1 routes both ranks through the Tailscale aliases
# (gx10-r0-ts / gx10-r1-ts via ProxyJump over the fabric, same for r2/r3).
[ "${PAIR_TS:-0}" = 1 ] && { R0=${R0}-ts; R1=${R1}-ts; }
case "$PAIR_NUM" in
  1) R0=gx10-r0; R1=gx10-r1; RF0=0; RF1=1 ;;
  2) R0=gx10-r2; R1=gx10-r3; RF0=2; RF1=3 ;;
  *) echo "unknown PAIR_CLUSTER: $PAIR_NUM (use 1|2)"; exit 64 ;;
esac
SERVE_DIR='~/builds/glm53-flash-dgx-spark-tp2/serve/TP2-DGX-Spark-GLM5.3F-Jovian-Judgement'

HEALTH_TIMEOUT="${PAIR_HEALTH_TIMEOUT:-720}"   # seconds to wait for the API after start; first boot on a new image = JIT cold, raise it (e.g. 1800)
cmd=${1:-}
rank_arg=${2:-}
[ "$cmd" = logs ] && { profile=${3:-mtp3-spark}; rank_arg=$2; } || { profile=${2:-mtp3-spark}; rank_arg=; }
case "$profile" in
  mtp3-spark) sfx="-mtp3-spark" ;;  # spark quant, MTP3 adaptive (1/3/32), marlin MTP
  df-spark)   sfx="-df-spark" ;;    # spark quant, DFlash2@7
  mtp3-nvfp4) sfx="-mtp3-nvfp4" ;;  # non-spark quant, MTP3 adaptive, humming MTP
  df-nvfp4)   sfx="-df-nvfp4" ;;    # non-spark quant, DFlash2@7
  *) echo "unknown profile: $profile (use mtp3-spark|df-spark|mtp3-nvfp4|df-nvfp4)"; exit 64 ;;
esac
env_file() { printf 'rank-%s%s.env' "$1" "$sfx"; }

say()  { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
# -F keeps ssh off /etc/ssh/ssh_config.d snippets (portable across machines
# whose system snippets are unreadable, e.g. sandboxed or remapped views).
rssh() { ssh -4 -F "$HOME/.ssh/config" -o BatchMode=yes -o ConnectTimeout=20 "$1" "${2:-true}"; }

get_pass() {
  local pass="${PAIR_SUDO_PASSWORD:-}"
  if [ -z "$pass" ] && [ -f "$HOME/.pair-sudo" ]; then pass=$(cat "$HOME/.pair-sudo"); fi
  if [ -z "$pass" ]; then
    read -rsp "sudo password for the pair (cached in ~/.pair-sudo): " pass; echo
    printf '%s' "$pass" > "$HOME/.pair-sudo"; chmod 600 "$HOME/.pair-sudo"
  fi
  printf '%s' "$pass"
}

# --- per-host preflight: swappiness=0 + drop caches (single sudo call) ----
preflight() {
  local host=$1
  local sw
  sw=$(rssh "$host" 'cat /proc/sys/vm/swappiness')
  if [ "$sw" != 0 ]; then
    say "fixing swappiness on $host ($sw -> 0)"
    P="$P" rssh "$host" "echo '$P' | sudo -S sh -c 'echo 0 > /proc/sys/vm/swappiness'" || \
      P="$P" rssh "$host" "echo '$P' | sudo -S sysctl vm.swappiness=0"
  else
    echo "swappiness already 0 on $host"
  fi
  P="$P" rssh "$host" "echo '$P' | sudo -S sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'" \
    && echo "dropped page cache on $host" || echo "WARN: drop_caches failed on $host"
}

# --- GID re-check: compare env file against live show_gids, auto-fix ------
fix_gid() {
  local host=$1 rank=$2 file
  file=$(env_file "$rank")
  rssh "$host" "
    cd $SERVE_DIR || exit 1
    hca=\$(grep '^NCCL_IB_HCA=' $file | cut -d= -f2)
    ip=\$(grep '^VLLM_HOST_IP=' $file | cut -d= -f2)
    set_idx=\$(grep '^NCCL_IB_GID_INDEX=' $file | cut -d= -f2)
    live_idx=\$(show_gids | awk -v h=\"\$hca\" -v i=\"\$ip\" '\$1==h && \$5==i && \$6==\"v2\" {print \$3}' | head -1)
    if [ -z \"\$live_idx\" ]; then
      echo \"ERROR: no RoCEv2 GID row for HCA=\$hca IP=\$ip on $host -- fabric down or IP changed?\"
      exit 1
    fi
    if [ \"\$live_idx\" != \"\$set_idx\" ]; then
      sed -i \"s/^NCCL_IB_GID_INDEX=.*/NCCL_IB_GID_INDEX=\$live_idx/\" $file
      echo \"GID index changed on $host: \$set_idx -> \$live_idx (fixed $file)\"
    else
      echo \"GID index OK on $host: \$live_idx\"
    fi"
}

case "$cmd" in
  up)
    P=$(get_pass)
    say "preflight $R0"; preflight "$R0"
    say "preflight $R1"; preflight "$R1"
    say "GID check"; fix_gid "$R0" "$RF0"; fix_gid "$R1" "$RF1"
    say "stale containers down (if any)"
    rssh "$R0" "cd $SERVE_DIR && bash glm53_pair_serve.sh --down $(env_file "$RF0") 2>/dev/null || true"
    rssh "$R1" "cd $SERVE_DIR && bash glm53_pair_serve.sh --down $(env_file "$RF1") 2>/dev/null || true"
    EXTRA="--kda-prefill-backend b12x --recurrent-checkpoint-policy request_boundaries"
    say "starting worker ($R1)"; rssh "$R1" "cd $SERVE_DIR && bash glm53_pair_serve.sh --run $(env_file "$RF1") $EXTRA"
    sleep 5
    say "starting head ($R0)";   rssh "$R0" "cd $SERVE_DIR && bash glm53_pair_serve.sh --run $(env_file "$RF0") $EXTRA"
    say "waiting for API on r0 (up to $HEALTH_TIMEOUT s)"
    t0=$SECONDS
    while ! rssh "$R0" "curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1"; do
      [ $((SECONDS - t0)) -lt "$HEALTH_TIMEOUT" ] || { echo "TIMEOUT: not healthy after ${HEALTH_TIMEOUT}s -- check logs: $0 logs 0 $profile"; exit 1; }
      echo "... not ready yet ($((SECONDS - t0))s)"
      sleep 20
    done
    echo "API healthy after $((SECONDS - t0))s"
    say "startup markers"
    rssh "$R0" "cd $SERVE_DIR && bash glm53_pair_serve.sh --verify $(env_file "$RF0")"
    echo
    echo "Done. API: http://<r0-lan-ip>:8000/v1  model: zai-org/GLM-5.3-Flash"
    ;;
  down)
    say "stopping $R0"; rssh "$R0" "cd $SERVE_DIR && bash glm53_pair_serve.sh --down $(env_file "$RF0")"
    say "stopping $R1"; rssh "$R1" "cd $SERVE_DIR && bash glm53_pair_serve.sh --down $(env_file "$RF1")"
    ;;
  status)
    rssh "$R0" "cd $SERVE_DIR && bash glm53_pair_serve.sh --status $(env_file "$RF0")"
    rssh "$R1" "cd $SERVE_DIR && bash glm53_pair_serve.sh --status $(env_file "$RF1")"
    ;;
  check)
    say "check r0"; rssh "$R0" "cd $SERVE_DIR && bash glm53_pair_serve.sh --check $(env_file "$RF0")"
    say "check r1"; rssh "$R1" "cd $SERVE_DIR && bash glm53_pair_serve.sh --check $(env_file "$RF1")"
    ;;
  logs)
    case "$rank_arg" in
      0) host=$R0; file=$(env_file "$RF0") ;;
      1) host=$R1; file=$(env_file "$RF1") ;;
      *) echo "usage: $0 logs 0|1 [profile]"; exit 64 ;;
    esac
    exec ssh -4 -F "$HOME/.ssh/config" -o BatchMode=yes "$host" "cd $SERVE_DIR && bash glm53_pair_serve.sh --logs $file"
    ;;
  *)
    sed -n '2,9p' "$0" | sed 's/^# //'
    exit 64
    ;;
esac
