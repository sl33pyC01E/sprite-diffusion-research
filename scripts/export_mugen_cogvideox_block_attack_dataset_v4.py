"""Create a balanced block-versus-attack CogVideoX causal gate."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/mugen-cogvideox-orange-fighter-i2v-native-caption-v3"
OUTPUT = ROOT / "data/processed/mugen-cogvideox-orange-fighter-block-attack-v4"
SOURCE_MANIFEST_SHA256 = "524a387ef02ce3ef42ac711e80f476d992f28e515edec37196822124821658aa"
VERBS = ("block", "normal_attack")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace CogVideoX dataset: {OUTPUT}")
    source_path = SOURCE / "manifest.json"
    source_bytes = source_path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != SOURCE_MANIFEST_SHA256:
        raise RuntimeError("caption-corrected source dataset differs")
    source = json.loads(source_bytes)
    records = [record for record in source["records"] if record["verb"] in VERBS]
    records.sort(key=lambda record: record["verb"].encode())
    if [record["verb"] for record in records] != list(VERBS):
        raise RuntimeError("causal-gate verb closure differs")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{OUTPUT.name}.stage-", dir=OUTPUT.parent))
    try:
        (stage / "videos").mkdir()
        prompts = []
        videos = []
        for record in records:
            relative = Path(record["video"]["path"])
            source_video = SOURCE / relative
            if file_sha256(source_video) != record["video"]["file_sha256"]:
                raise RuntimeError(f"video hash differs: {relative}")
            shutil.copyfile(source_video, stage / relative)
            if file_sha256(stage / relative) != record["video"]["file_sha256"]:
                raise RuntimeError(f"copied video hash differs: {relative}")
            prompts.append(record["prompt"])
            videos.append(relative.as_posix())
        (stage / "prompts.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8")
        (stage / "videos.txt").write_text("\n".join(videos) + "\n", encoding="utf-8")
        manifest = {
            "artifact_kind": "mugen_cogvideox_i2v_block_attack_causal_gate_dataset",
            "claim": "one-identity two-action capacity gate; not a generalization set",
            "counts": {"identities": 1, "sequences": 2, "verbs": 2},
            "projection": source["projection"],
            "records": records,
            "schema_version": 4,
            "source": {
                "dataset_manifest_path": str(source_path),
                "dataset_manifest_sha256": SOURCE_MANIFEST_SHA256,
            },
        }
        payload = canonical_json(manifest)
        (stage / "manifest.json").write_bytes(payload)
        os.rename(stage, OUTPUT)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "manifest_sha256": hashlib.sha256(payload).hexdigest(),
                "output": str(OUTPUT),
                "sequences": len(records),
            },
            sort_keys=True,
        )
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


if __name__ == "__main__":
    main()
