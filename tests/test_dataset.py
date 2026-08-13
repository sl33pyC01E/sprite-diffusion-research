from dataclasses import replace

from spritelab.dataset import (
    SequenceSample,
    SplitPolicy,
    SplitRatios,
    build_dataset_manifest,
    canonical_frame_phases,
    coverage_report,
    multi_action_identity_groups,
)


def _sample(sequence: str, identity: str, action: str, digest: str) -> SequenceSample:
    blob_sha256 = digest if len(digest) == 64 else digest * 64
    return SequenceSample(
        sequence_id=sequence,
        identity_id=identity,
        source_id="fixture",
        source_pack_id=f"pack-{identity}",
        entity_class="animal" if identity == "wolf" else "humanoid",
        action=action,
        view="side",
        direction="right",
        loop_mode="loop",
        frame_count=4,
        source_blob_sha256=(blob_sha256,),
    )


def test_split_keeps_identity_pack_and_duplicate_edges_together() -> None:
    samples = (
        _sample("wolf-idle", "wolf", "idle", "a"),
        _sample("wolf-run", "wolf", "run", "b"),
        replace(
            _sample("knight-idle", "knight", "idle", "c"),
            duplicate_group_ids=("near-dup-1",),
        ),
        replace(
            _sample("mage-cast", "mage", "cast", "d"),
            duplicate_group_ids=("near-dup-1",),
        ),
    )
    manifest = build_dataset_manifest(samples, SplitPolicy(seed="test-v1"))
    assignments = {row.sequence_id: row for row in manifest.assignments}

    assert assignments["wolf-idle"].split == assignments["wolf-run"].split
    assert assignments["knight-idle"].split == assignments["mage-cast"].split
    assert assignments["knight-idle"].component_id == assignments["mage-cast"].component_id


def test_manifest_is_order_independent_and_hash_stable() -> None:
    samples = (
        _sample("wolf-idle", "wolf", "idle", "a"),
        _sample("wolf-run", "wolf", "run", "b"),
        _sample("mage-cast", "mage", "cast", "c"),
    )
    policy = SplitPolicy(seed="stable", group_source_pack=False)

    forward = build_dataset_manifest(samples, policy)
    reverse = build_dataset_manifest(reversed(samples), policy)

    assert forward == reverse
    assert forward.sha256 == reverse.sha256


def test_balanced_strategy_populates_small_evaluation_splits_deterministically() -> None:
    samples = tuple(
        _sample(f"sequence-{index:02d}", f"identity-{index:02d}", "idle", f"{index:064x}")
        for index in range(30)
    )
    policy = SplitPolicy(
        seed="balanced-v1",
        ratios=SplitRatios(train=0.8, validation=0.1, test=0.1),
        assignment_strategy="balanced",
        group_source_pack=False,
    )

    forward = build_dataset_manifest(samples, policy)
    reverse = build_dataset_manifest(reversed(samples), policy)
    counts = coverage_report(forward).split_counts

    assert forward == reverse
    assert counts == {"test": 3, "train": 24, "validation": 3}


def test_balanced_strategy_never_splits_large_leakage_components() -> None:
    samples = tuple(
        _sample(f"wolf-{index}", "wolf", "run", f"{index:064x}") for index in range(12)
    ) + tuple(
        _sample(f"other-{index}", f"other-{index}", "idle", f"{index + 20:064x}")
        for index in range(8)
    )
    manifest = build_dataset_manifest(
        samples,
        SplitPolicy(
            seed="large-component",
            assignment_strategy="balanced",
            group_source_pack=False,
        ),
    )
    wolf_splits = {
        assignment.split
        for assignment in manifest.assignments
        if assignment.sequence_id.startswith("wolf-")
    }

    assert len(wolf_splits) == 1


def test_multi_action_and_coverage_reports() -> None:
    samples = (
        _sample("wolf-idle", "wolf", "idle", "a"),
        _sample("wolf-run", "wolf", "run", "b"),
        _sample("mage-cast", "mage", "cast", "c"),
    )
    manifest = build_dataset_manifest(
        samples,
        SplitPolicy(seed="coverage", group_source_pack=False),
    )

    assert set(multi_action_identity_groups(samples)) == {"wolf"}
    report = coverage_report(manifest)
    assert report.sample_count == 3
    assert report.identity_count == 2
    assert report.multi_action_identity_count == 1
    assert report.entity_counts == {"animal": 2, "humanoid": 1}
    assert report.temporal_sequence_count == 3


def test_canonical_frame_phases_preserve_loop_seam_contract() -> None:
    assert canonical_frame_phases(4, "loop") == (0.0, 0.25, 0.5, 0.75)
    assert canonical_frame_phases(3, "one_shot") == (0.0, 0.5, 1.0)
