#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

usage() {
  cat << 'EOF'
Usage:
  # 1) 你已经有 frozen .pth（你当前情况：exports_cuda_opt/model_cuda_*.pth）
  ./freeze_compress_test_deepmd_pt_cuda.sh -m exports_cuda_opt/model_cuda_20251228-121805.pth -s /path/to/data -n 200

  # 2) 你只有训练目录（里面有 checkpoint .pt），需要先 freeze 再 compress 再 test
  ./freeze_compress_test_deepmd_pt_cuda.sh -t /path/to/train_dir -s /path/to/data -n 200

Options:
  -e <env>        conda 环境名（默认：deepmd）
  -m <model.pth>  已冻结的 PyTorch 模型（.pth）。提供则跳过 freeze
  -t <train_dir>  训练目录（包含 checkpoint），用于 dp --pt freeze（当未提供 -m 时必需）
  -s <system_dir> 测试数据目录（dp test 的 -s）
  -n <N>          测试帧数（dp test 的 -n，默认：100）
  --head <name>   multi-task 模型冻结指定 head（可选）
  -o <out_dir>    输出目录（默认：exports_freeze_compress）
EOF
}

# ---------------- defaults ----------------
CONDA_ENV="deepmd"
FROZEN_IN=""
TRAIN_DIR=""
SYSTEM_DIR=""
NUMB_TEST="100"
HEAD=""
OUT_DIR="exports_freeze_compress"

# ---------------- args parse ----------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    -e) CONDA_ENV="$2"; shift 2;;
    -m) FROZEN_IN="$2"; shift 2;;
    -t) TRAIN_DIR="$2"; shift 2;;
    -s) SYSTEM_DIR="$2"; shift 2;;
    -n) NUMB_TEST="$2"; shift 2;;
    --head) HEAD="$2"; shift 2;;
    -o) OUT_DIR="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "[ERROR] Unknown argument: $1"; usage; exit 2;;
  esac
done

if [[ -z "${SYSTEM_DIR}" ]]; then
  echo "[ERROR] 必须提供测试数据目录：-s /path/to/system"
  usage
  exit 2
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
mkdir -p "${OUT_DIR}" "${OUT_DIR}/logs"

# ---------------- conda activate ----------------
if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda 未在 PATH 中。请在同一 shell 里先 source 你的 conda 初始化脚本。"
  exit 1
fi

# 记录当前是否启用了 nounset（set -u）
_nounset_was_on=0
case "$-" in
  *u*) _nounset_was_on=1; set +u ;;  # 临时关闭 nounset，避免 conda activate.d 脚本引用未定义变量时报错
esac

# 兼容非交互 shell：显式加载 conda.sh
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "[INFO] Activating conda env: ${CONDA_ENV}"
conda activate "${CONDA_ENV}"

# conda 激活完成后，恢复 nounset
if [[ "${_nounset_was_on}" -eq 1 ]]; then
  set -u
fi

if ! command -v dp >/dev/null 2>&1; then
  echo "[ERROR] dp 命令不存在。请确认 deepmd-kit 已正确安装在环境 ${CONDA_ENV} 中。"
  exit 1
fi

echo "[INFO] dp version:"
dp --version || true

# ---------------- step 1: freeze (optional) ----------------
FROZEN_OUT="${OUT_DIR}/frozen_${timestamp}.pth"

if [[ -n "${FROZEN_IN}" ]]; then
  if [[ ! -f "${FROZEN_IN}" ]]; then
    echo "[ERROR] 找不到你提供的 .pth：${FROZEN_IN}"
    exit 1
  fi
  # 统一拷贝到 out_dir，便于后续压缩/测试结果可追溯
  cp -f "${FROZEN_IN}" "${FROZEN_OUT}"
  echo "[INFO] Using existing frozen model: ${FROZEN_IN}"
  echo "[INFO] Copied to: ${FROZEN_OUT}"
else
  if [[ -z "${TRAIN_DIR}" ]]; then
    echo "[ERROR] 未提供 -m 时，必须提供训练目录 -t /path/to/train_dir（用于 dp --pt freeze）。"
    usage
    exit 2
  fi
  if [[ ! -d "${TRAIN_DIR}" ]]; then
    echo "[ERROR] 训练目录不存在：${TRAIN_DIR}"
    exit 1
  fi

  echo "[INFO] Freezing from train dir: ${TRAIN_DIR}"
  pushd "${TRAIN_DIR}" >/dev/null

  FREEZE_LOG="../${OUT_DIR}/logs/freeze_${timestamp}.log"
  if [[ -n "${HEAD}" ]]; then
    # multi-task: 需要 --head :contentReference[oaicite:3]{index=3}
    dp --pt freeze -o "../${FROZEN_OUT}" --head "${HEAD}" 2>&1 | tee "${FREEZE_LOG}"
  else
    dp --pt freeze -o "../${FROZEN_OUT}" 2>&1 | tee "${FREEZE_LOG}"
  fi

  popd >/dev/null

  if [[ ! -f "${FROZEN_OUT}" ]]; then
    echo "[ERROR] freeze 失败：未生成 ${FROZEN_OUT}"
    exit 1
  fi
  echo "[INFO] Frozen model written to: ${FROZEN_OUT}"
fi

# ---------------- step 2: compress ----------------
COMPRESSED_OUT="${OUT_DIR}/compressed_${timestamp}.pth"
COMPRESS_LOG="${OUT_DIR}/logs/compress_${timestamp}.log"

echo "[INFO] Compressing model..."
# PyTorch 压缩命令：dp --pt compress -i ... -o ... :contentReference[oaicite:4]{index=4}
dp --pt compress -i "${FROZEN_OUT}" -o "${COMPRESSED_OUT}" 2>&1 | tee "${COMPRESS_LOG}"

if [[ ! -f "${COMPRESSED_OUT}" ]]; then
  echo "[ERROR] compress 失败：未生成 ${COMPRESSED_OUT}"
  exit 1
fi
echo "[INFO] Compressed model written to: ${COMPRESSED_OUT}"

# ---------------- step 3: test (frozen + compressed) ----------------
TEST_LOG_FROZEN="${OUT_DIR}/logs/test_frozen_${timestamp}.log"
TEST_LOG_COMP="${OUT_DIR}/logs/test_compressed_${timestamp}.log"

echo "[INFO] Testing frozen model..."
# dp test 基本用法：dp test -m model -s system -n N :contentReference[oaicite:5]{index=5}
dp test -m "${FROZEN_OUT}" -s "${SYSTEM_DIR}" -n "${NUMB_TEST}" 2>&1 | tee "${TEST_LOG_FROZEN}"

echo "[INFO] Testing compressed model..."
dp test -m "${COMPRESSED_OUT}" -s "${SYSTEM_DIR}" -n "${NUMB_TEST}" 2>&1 | tee "${TEST_LOG_COMP}"

echo
echo "[DONE] Outputs:"
echo "  frozen     : ${FROZEN_OUT}"
echo "  compressed : ${COMPRESSED_OUT}"
echo "  logs       : ${OUT_DIR}/logs/"
echo
