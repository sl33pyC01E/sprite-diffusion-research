#!/usr/bin/env bash
set -euo pipefail

revision="a6f0f4858a8395e7429d82493864ce92bf73af11"
root="/home/sleepy/sprite-lab-cogvideox/CogVideoX-5b-I2V-a6f0f4858a83"
base="https://huggingface.co/THUDM/CogVideoX-5b-I2V/resolve/${revision}"

fresh_download() {
  local relative="$1"
  local partial="$2"
  local quarantine="$3"
  local expected_bytes="$4"
  local expected_sha256="$5"
  local destination="${root}/${relative}"

  if [[ -e "${destination}" || -e "${quarantine}" ]]; then
    echo "refusing to replace CogVideoX destination or quarantine" >&2
    return 1
  fi
  mv --no-clobber "${partial}" "${quarantine}"
  curl --fail --location --retry 20 --retry-all-errors \
    --output "${partial}" "${base}/${relative}?download=true"
  test "$(stat --format=%s "${partial}")" = "${expected_bytes}"
  echo "${expected_sha256}  ${partial}" | sha256sum --check --strict
  mv --no-clobber "${partial}" "${destination}"
}

fresh_download \
  "transformer/diffusion_pytorch_model-00001-of-00003.safetensors" \
  "${root}/.cache/huggingface/download/transformer/5n3ByLHcDVGl-4O5UeVdJRvodxk=.f2e3060199c34a0d18892a19d687455f938b0ac3d2ea7d48f37cb4090e141965.incomplete" \
  "${root}/.cache/huggingface/download/transformer/shard1.resume-corrupt.incomplete" \
  4992465072 \
  f2e3060199c34a0d18892a19d687455f938b0ac3d2ea7d48f37cb4090e141965 &
pid_one=$!

fresh_download \
  "text_encoder/model-00002-of-00002.safetensors" \
  "${root}/.cache/huggingface/download/text_encoder/Dr_lZJDwE1cnGAQMwA77jJEQIk8=.d3edef29693d52402b1cc7c362f031e052f2e9482ed0c765c6351950434349b0.incomplete" \
  "${root}/.cache/huggingface/download/text_encoder/shard2.resume-corrupt.incomplete" \
  4530066360 \
  d3edef29693d52402b1cc7c362f031e052f2e9482ed0c765c6351950434349b0 &
pid_two=$!

wait "${pid_one}"
wait "${pid_two}"
echo "remaining CogVideoX shards freshly downloaded and SHA-256 verified"
