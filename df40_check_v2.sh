#!/bin/bash
#SBATCH --job-name=df40_check_v2
#SBATCH --output=df40_check_v2_%j.out
#SBATCH --error=df40_check_v2_%j.err
#SBATCH --partition=long
#SBATCH --nodelist=gpu22
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00

set -euo pipefail

BASE="/tmp/f2256768/df40/deepfakes_detection_datasets/DF40"

DIRS=(
  blendface e4s hyperreenact one_shot_free SiT tpsm
  CollabDiff facedancer inswap pirender stargan uniface
  danet faceswap lia pixart starganv2 VQGAN
  ddim facevid2vid mcnet RDDM styleclip wav2lip
  deepfacelab fomm MidJourney sadtalker StyleGAN2 whichfaceisreal
  DiT fsgan mobileswap sd2.1 StyleGAN3
  e4e heygen MRAA simswap StyleGANXL
)

# timeout 可能没有；给个兼容兜底
have_timeout=1
command -v timeout >/dev/null 2>&1 || have_timeout=0
tcmd() {
  # 用法: tcmd 30s <command...>
  if [ "$have_timeout" -eq 1 ]; then
    timeout "$@"
  else
    # 没 timeout 就直接跑
    shift
    "$@"
  fi
}

echo "BASE = $BASE"
echo "Host = $(hostname)"
echo "Time = $(date)"
echo

if [ ! -d "$BASE" ]; then
  echo "ERROR: BASE path not found: $BASE"
  exit 1
fi

echo "==== Top-level listing (du -sh, largest 60) ===="
tcmd 60s du -sh "$BASE"/* 2>/dev/null | sort -h | tail -n 60 || true
echo

SUSPICIOUS=()

echo "==== Per-folder checks (per-kind sample 20) ===="
for d in "${DIRS[@]}"; do
  P="$BASE/$d"
  echo "------------------------------------------------------------"
  echo "[$d]  $(date '+%F %T')"

  if [ ! -d "$P" ]; then
    echo "MISSING: $P"
    SUSPICIOUS+=("$d:MISSING")
    continue
  fi

  # 1) Size (lightweight)
  SIZE="$(tcmd 60s du -sh "$P" 2>/dev/null | awk '{print $1}' || echo "UNKNOWN")"
  echo "Size: $SIZE"

  # 2) Quick non-empty check (early stop)
  if tcmd 20s find "$P" -type f -print -quit 2>/dev/null | grep -q .; then
    echo "Non-empty: yes"
  else
    echo "Non-empty: NO (suspicious)"
    SUSPICIOUS+=("$d:EMPTY")
    echo "ls -lah (top 100):"
    ls -lah "$P" | head -n 100 || true
    # 空目录没必要继续列结构
    continue
  fi

  # 3) Per-kind sample: each top-level kind print up to 20 subdirs then move on
  echo "Structure sample (per kind, show <=20 children each):"

  # 列出一级子目录（cdf/ff/...），只遍历一层，不会炸
  kinds="$(tcmd 30s find "$P" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" 2>/dev/null | sort || true)"

  if [ -z "$kinds" ]; then
    echo "  (No subdirectories at depth=1?)"
    SUSPICIOUS+=("$d:NO_KIND_DIR")
    continue
  fi

  while IFS= read -r kind; do
    [ -z "$kind" ] && continue
    echo "  - $kind:"

    # 只列该 kind 下一级（第二层）目录的前 20 个
    tcmd 30s find "$P/$kind" -mindepth 1 -maxdepth 1 -type d -printf "    %p\n" 2>/dev/null | head -n 20 || true
    echo "    ... (showing up to 20)"
  done <<< "$kinds"

done

echo
echo "==== Summary ===="
if [ "${#SUSPICIOUS[@]}" -eq 0 ]; then
  echo "No obvious issues detected (missing/empty/no-kind)."
else
  echo "Suspicious entries:"
  for x in "${SUSPICIOUS[@]}"; do
    echo "  - $x"
  done
fi

echo
echo "Done. Time = $(date)"