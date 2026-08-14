#!/usr/bin/env bash
set -euo pipefail

revision="a6f0f4858a8395e7429d82493864ce92bf73af11"
root="/home/sleepy/sprite-lab-cogvideox/CogVideoX-5b-I2V-a6f0f4858a83"
relative="text_encoder/model-00002-of-00002.safetensors"
url="https://huggingface.co/THUDM/CogVideoX-5b-I2V/resolve/${revision}/${relative}?download=true"
partial="${root}/.cache/huggingface/download/text_encoder/Dr_lZJDwE1cnGAQMwA77jJEQIk8=.d3edef29693d52402b1cc7c362f031e052f2e9482ed0c765c6351950434349b0.incomplete"
quarantine="${root}/.cache/huggingface/download/text_encoder/text-fresh-retry-unstable.incomplete"
parts="${root}/.cache/huggingface/download/text_encoder/fixed-ranges-v1"
assembled="${root}/.cache/huggingface/download/text_encoder/model-00002-assembled-v1.incomplete"
destination="${root}/${relative}"
expected_bytes=4530066360
expected_sha256="d3edef29693d52402b1cc7c362f031e052f2e9482ed0c765c6351950434349b0"
chunk_bytes=300000000

if [[ -e "${destination}" || -e "${quarantine}" || -e "${parts}" || -e "${assembled}" ]]; then
  echo "refusing to replace CogVideoX text destination, quarantine, ranges, or assembly" >&2
  exit 1
fi
mv --no-clobber "${partial}" "${quarantine}"
mkdir "${parts}"

pids=()
index=0
start=0
while (( start < expected_bytes )); do
  end=$((start + chunk_bytes - 1))
  if (( end >= expected_bytes )); then
    end=$((expected_bytes - 1))
  fi
  part="${parts}/part-$(printf '%02d' "${index}")"
  expected_part_bytes=$((end - start + 1))
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
mv --no-clobber "${assembled}" "${destination}"
echo "CogVideoX text shard range-downloaded and SHA-256 verified"
