#!/usr/bin/env bash
set -euo pipefail

root="/home/sleepy/sprite-lab-cogvideox"
model="${root}/CogVideoX-5b-I2V-a6f0f4858a83"
dataset="${root}/mugen-cogvideox-orange-fighter-i2v-native-caption-v3"
diffusers="${root}/diffusers-2da7040be1a2"
trainer="${diffusers}/examples/cogvideo/train_cogvideox_image_to_video_lora.py"
python="/home/sleepy/ComfyUI/comfyui-env-310/bin/python"
packages="${root}/python-packages"
output="${root}/lora-orange-fighter-native-caption-r128-step250-v2"
log="${root}/lora-orange-fighter-native-caption-r128-step250-v2.log"

test "$(sha256sum "${model}/source-index.json" | cut -d' ' -f1)" = "98fbc592f23269a38d039d16f969844a9da073b56b24567772433d4b02e2f831"
test "$(sha256sum "${dataset}/manifest.json" | cut -d' ' -f1)" = "524a387ef02ce3ef42ac711e80f476d992f28e515edec37196822124821658aa"
test "$(sha256sum "${trainer}" | cut -d' ' -f1)" = "78430e548e68f7f30e434abc877439f560f428686a00cf5e04ae471b3e6249e8"
if [[ -e "${output}" || -e "${log}" ]]; then
  echo "refusing to replace CogVideoX LoRA output or log" >&2
  exit 1
fi
mkdir -p "${output}"

export PYTHONPATH="${diffusers}/src:${packages}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

"${python}" "${trainer}" \
  --pretrained_model_name_or_path "${model}" \
  --instance_data_root "${dataset}" \
  --video_column videos.txt \
  --caption_column prompts.txt \
  --height 480 \
  --width 720 \
  --fps 8 \
  --max_num_frames 9 \
  --rank 128 \
  --lora_alpha 128 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --max_train_steps 250 \
  --checkpointing_steps 50 \
  --checkpoints_total_limit 2 \
  --gradient_checkpointing \
  --learning_rate 1e-4 \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --mixed_precision bf16 \
  --optimizer adamw \
  --enable_slicing \
  --enable_tiling \
  --noised_image_dropout 0 \
  --seed 20260832 \
  --allow_tf32 \
  --dataloader_num_workers 0 \
  --output_dir "${output}" \
  2>&1 | tee "${log}"

sums="${output}.sha256sums.txt"
test ! -e "${sums}"
find "${output}" -type f -print0 | sort -z | xargs -0 sha256sum > "${sums}"
mv "${sums}" "${output}/sha256sums.txt"
