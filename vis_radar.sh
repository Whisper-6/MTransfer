#!/usr/bin/env bash
set -e

# ----------------------
# 参数解析
# ----------------------
if [ $# -lt 1 ]; then
  echo "Usage: $0 <result-dir>"
  exit 1
fi

RESULT_DIR="$1"
MIN_VAL=${2:-0.0}
MAX_VAL=${3:-1.0}
SCALE_VAL=${4:-0.1}

# ----------------------
# 打印信息
# ----------------------
echo "Result directory: ${RESULT_DIR}"
echo "Running radar visualizations..."

# ----------------------
# 调用脚本
# ----------------------
python vis_each_radar.py --result-dir "${RESULT_DIR}" --min ${MIN_VAL} --max ${MAX_VAL} --scale ${SCALE_VAL}
python vis_group_radar.py --result-dir "${RESULT_DIR}" --min ${MIN_VAL} --max ${MAX_VAL} --scale ${SCALE_VAL}

echo "All radar plots generated successfully."
