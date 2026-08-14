#!/usr/bin/env bash
set -euo pipefail

revision="a6f0f4858a8395e7429d82493864ce92bf73af11"
root="/home/sleepy/sprite-lab-cogvideox/CogVideoX-5b-I2V-a6f0f4858a83"
relative="transformer/diffusion_pytorch_model-00002-of-00003.safetensors"
partial="${root}/.cache/huggingface/download/transformer/xIb00LLWe-iHth2j35r6TGIYKx8=.1e8d0c62d366b0d9cc3476d2b21ca54afbecea154d54d923da120b2ec174c7e7.incomplete"
destination="${root}/${relative}"
quarantine="${partial}.corrupt-57f2a089"
expected_bytes=4985800640
expected_sha256="1e8d0c62d366b0d9cc3476d2b21ca54afbecea154d54d923da120b2ec174c7e7"

if [[ -e "${destination}" || -e "${quarantine}" ]]; then
  echo "refusing to replace CogVideoX destination or collision quarantine" >&2
  exit 1
fi
mv --no-clobber "${partial}" "${quarantine}"
curl --fail --location --retry 20 --retry-all-errors \
  --output "${partial}" \
  "https://huggingface.co/THUDM/CogVideoX-5b-I2V/resolve/${revision}/${relative}?download=true"
test "$(stat --format=%s "${partial}")" = "${expected_bytes}"
echo "${expected_sha256}  ${partial}" | sha256sum --check --strict
mv --no-clobber "${partial}" "${destination}"
echo "CogVideoX transformer shard two redownloaded and SHA-256 verified"
