#!/usr/bin/env bash
set -euo pipefail

revision="a6f0f4858a8395e7429d82493864ce92bf73af11"
root="/home/sleepy/sprite-lab-cogvideox/CogVideoX-5b-I2V-a6f0f4858a83"
base="https://huggingface.co/THUDM/CogVideoX-5b-I2V/resolve/${revision}"

continue_download() {
  local relative="$1"
  local partial="$2"
  local expected_bytes="$3"
  local expected_sha256="$4"
  local destination="${root}/${relative}"

  if [[ -e "${destination}" ]]; then
    echo "refusing to replace CogVideoX destination" >&2
    return 1
  fi
  curl --fail --location --retry 20 --retry-all-errors --continue-at - \
    --output "${partial}" "${base}/${relative}?download=true"
  test "$(stat --format=%s "${partial}")" = "${expected_bytes}"
  echo "${expected_sha256}  ${partial}" | sha256sum --check --strict
  mv --no-clobber "${partial}" "${destination}"
}

continue_download \
  "transformer/diffusion_pytorch_model-00001-of-00003.safetensors" \
  "${root}/.cache/huggingface/download/transformer/5n3ByLHcDVGl-4O5UeVdJRvodxk=.f2e3060199c34a0d18892a19d687455f938b0ac3d2ea7d48f37cb4090e141965.incomplete" \
  4992465072 \
  f2e3060199c34a0d18892a19d687455f938b0ac3d2ea7d48f37cb4090e141965 &
pid_one=$!

continue_download \
  "transformer/diffusion_pytorch_model-00002-of-00003.safetensors" \
  "${root}/.cache/huggingface/download/transformer/xIb00LLWe-iHth2j35r6TGIYKx8=.1e8d0c62d366b0d9cc3476d2b21ca54afbecea154d54d923da120b2ec174c7e7.incomplete" \
  4985800640 \
  1e8d0c62d366b0d9cc3476d2b21ca54afbecea154d54d923da120b2ec174c7e7 &
pid_two=$!

continue_download \
  "text_encoder/model-00002-of-00002.safetensors" \
  "${root}/.cache/huggingface/download/text_encoder/Dr_lZJDwE1cnGAQMwA77jJEQIk8=.d3edef29693d52402b1cc7c362f031e052f2e9482ed0c765c6351950434349b0.incomplete" \
  4530066360 \
  d3edef29693d52402b1cc7c362f031e052f2e9482ed0c765c6351950434349b0 &
pid_three=$!

wait "${pid_one}"
wait "${pid_two}"
wait "${pid_three}"
echo "fresh CogVideoX shards resumed and SHA-256 verified"
