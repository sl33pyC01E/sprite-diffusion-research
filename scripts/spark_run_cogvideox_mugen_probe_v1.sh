#!/usr/bin/env bash
set -euo pipefail

root="/home/sleepy/sprite-lab-cogvideox"
python="/home/sleepy/ComfyUI/comfyui-env-310/bin/python"
diffusers="${root}/diffusers-2da7040be1a2"
packages="${root}/python-packages"
probe="${root}/spark_probe_cogvideox_mugen_v1.py"
output="${root}/probe-orange-fighter-normal-attack-v2"
log="${root}/probe-orange-fighter-normal-attack-v2.log"

if [[ -e "${output}" || -e "${log}" ]]; then
  echo "refusing to replace CogVideoX probe output or log" >&2
  exit 1
fi
if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
  echo "refusing to launch while a GPU compute process is active" >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader >&2
  exit 1
fi

export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONPATH="${diffusers}/src:${packages}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

exec "${python}" "${probe}" >"${log}" 2>&1
