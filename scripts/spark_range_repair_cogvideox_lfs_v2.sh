#!/usr/bin/env bash
set -euo pipefail

revision="a6f0f4858a8395e7429d82493864ce92bf73af11"
root="/home/sleepy/sprite-lab-cogvideox/CogVideoX-5b-I2V-a6f0f4858a83"
base="https://huggingface.co/THUDM/CogVideoX-5b-I2V/resolve/${revision}"
work="${root}/.cache/huggingface/fixed-ranges-repair-v2"
quarantine="${root}/.cache/huggingface/quarantine-v2"
chunk_bytes=300000000

range_repair() {
  local relative="$1"
  local expected_bytes="$2"
  local expected_sha256="$3"
  local corrupt_sha256="$4"
  local label="$5"
  local destination="${root}/${relative}"
  local quarantined="${quarantine}/${label}.corrupt-${corrupt_sha256:0:8}"
  local parts="${work}/${label}"
  local assembled="${parts}/assembled.incomplete"
  local url="${base}/${relative}?download=true"

  if [[ ! -f "${destination}" || -e "${quarantined}" || -e "${parts}" ]]; then
    echo "repair preconditions differ for ${relative}" >&2
    return 1
  fi
  test "$(stat --format=%s "${destination}")" = "${expected_bytes}"
  test "$(sha256sum "${destination}" | cut -d' ' -f1)" = "${corrupt_sha256}"
  mkdir -p "${quarantine}" "${work}"
  mv --no-clobber "${destination}" "${quarantined}"
  mkdir "${parts}"

  local pids=()
  local index=0
  local start=0
  while (( start < expected_bytes )); do
    local end=$((start + chunk_bytes - 1))
    if (( end >= expected_bytes )); then
      end=$((expected_bytes - 1))
    fi
    local part="${parts}/part-$(printf '%02d' "${index}")"
    local expected_part_bytes=$((end - start + 1))
    (
      curl --fail --location --retry 20 --retry-all-errors \
        --range "${start}-${end}" --output "${part}" "${url}"
      test "$(stat --format=%s "${part}")" = "${expected_part_bytes}"
    ) &
    pids+=("$!")
    index=$((index + 1))
    start=$((end + 1))
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done

  cat "${parts}"/part-* >"${assembled}"
  test "$(stat --format=%s "${assembled}")" = "${expected_bytes}"
  echo "${expected_sha256}  ${assembled}" | sha256sum --check --strict
  mkdir -p "$(dirname "${destination}")"
  mv --no-clobber "${assembled}" "${destination}"
  echo "repaired ${relative}"
}

range_repair \
  "text_encoder/model-00001-of-00002.safetensors" \
  4994582224 \
  9162b8ae9152e7a8e3bbebc535c8692783f50aec8cd3bb8ef6a751c432dd6392 \
  dabd814e520f66caedf4fdef6b740c840aa05bd943261d1647c1eb37eaaba697 \
  text-encoder-00001 &
pid_text=$!

range_repair \
  "transformer/diffusion_pytorch_model-00003-of-00003.safetensors" \
  1272025856 \
  da91a0051da3f39caf10944b7c9aa66b14ddeffb37a25b087c49fc1692c1a361 \
  bd6a0f0360c9e10febba10452d55a95048a1538e923ab8087dd7c4f7b739e437 \
  transformer-00003 &
pid_transformer=$!

range_repair \
  "vae/diffusion_pytorch_model.safetensors" \
  862388596 \
  a410e48d988c8224cef392b68db0654485cfd41f345f4a3a81d3e6b765bb995e \
  d978a6d8a242fac730e98479ddafafa3d6f2b87970c982c69d103cb8e0a9fbb6 \
  vae &
pid_vae=$!

wait "${pid_text}"
wait "${pid_transformer}"
wait "${pid_vae}"
echo "all previously complete CogVideoX LFS payloads repaired and verified"
