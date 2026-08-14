"""Correct alpha-incompatible captions in the native MUGEN CogVideoX gate."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/processed/mugen-cogvideox-orange-fighter-i2v-native-v2"
OUTPUT = ROOT / "data/processed/mugen-cogvideox-orange-fighter-i2v-native-caption-v3"
OLD_CUE = "transparent background"
NEW_CUE = "plain neutral gray background"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to replace CogVideoX dataset: {OUTPUT}")
    source_path = SOURCE / "manifest.json"
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    if source.get("schema_version") != 2 or len(source.get("records", [])) != 10:
        raise RuntimeError("native source dataset differs")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{OUTPUT.name}.stage-", dir=OUTPUT.parent))
    try:
        (stage / "videos").mkdir()
        records = []
        prompts = []
        videos = []
        for source_record in source["records"]:
            record = json.loads(json.dumps(source_record))
            prompt = record["prompt"]
            if prompt.count(OLD_CUE) != 1 or NEW_CUE in prompt:
                raise RuntimeError(f"caption background cue differs: {record['sequence_id']}")
            record["prompt"] = prompt.replace(OLD_CUE, NEW_CUE)
            relative = Path(record["video"]["path"])
            source_video = SOURCE / relative
            if file_sha256(source_video) != record["video"]["file_sha256"]:
                raise RuntimeError(f"video hash differs: {relative}")
            shutil.copyfile(source_video, stage / relative)
            if file_sha256(stage / relative) != record["video"]["file_sha256"]:
                raise RuntimeError(f"copied video hash differs: {relative}")
            records.append(record)
            prompts.append(record["prompt"])
            videos.append(relative.as_posix())

        (stage / "prompts.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8")
        (stage / "videos.txt").write_text("\n".join(videos) + "\n", encoding="utf-8")
        manifest = {
            "artifact_kind": "mugen_cogvideox_i2v_native_caption_corrected_overfit_dataset",
            "caption_transform": {
                "new_cue": NEW_CUE,
                "old_cue": OLD_CUE,
                "reason": "CogVideoX trains RGB video and cannot supervise alpha transparency",
            },
            "claim": source["claim"],
            "counts": source["counts"],
            "projection": source["projection"],
            "records": records,
            "schema_version": 3,
            "source": {
                "dataset_manifest_path": str(source_path),
                "dataset_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
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
