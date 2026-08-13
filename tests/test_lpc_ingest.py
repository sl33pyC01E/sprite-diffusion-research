from __future__ import annotations

from spritelab.adapters.lpc import LpcCredit, parse_sheet_definition
from spritelab.ingest.lpc import (
    LpcArchiveMemberFact,
    LpcManifestBuilder,
    iter_lpc_manifest_records,
)

ARCHIVE_SHA256 = "a" * 64
ROOT = "Universal-LPC-Spritesheet-Character-Generator-deadbeef"


def _builder(
    *,
    credits: tuple[LpcCredit, ...] = (),
    definitions: tuple = (),
) -> LpcManifestBuilder:
    return LpcManifestBuilder(
        archive_blob_sha256=ARCHIVE_SHA256,
        credits=credits,
        definitions=definitions,
    )


def test_builds_modular_layer_record_with_lossless_direction_rectangles() -> None:
    member = LpcArchiveMemberFact(
        ordinal=41,
        member_path=f"{ROOT}/spritesheets/body/bodies/male/run/blue.png",
        width=512,
        height=256,
        extracted_blob_sha256="b" * 64,
        pixel_sha256="c" * 64,
        inspection_status="media_inspected",
    )
    credit = LpcCredit(
        filename="body/bodies/male/run.png",
        notes="generated recolor source",
        authors=("Artist A",),
        licenses=("CC-BY-SA 3.0",),
        urls=("https://example.test/art",),
    )

    record = _builder(credits=(credit,)).build_record(member)

    assert record is not None
    assert record.record_kind == "modular_compositing_layer_sheet"
    assert not record.is_complete_entity
    assert record.composition_required
    assert record.layer_identity == "ulpc:body/bodies/male"
    assert record.source_action == "run"
    assert record.normalized_action == "run"
    assert record.palette == "blue"
    assert record.geometry.status == "canonical"
    assert record.geometry.actual_frame_size == 64
    assert record.geometry.actual_frame_count == 8
    assert [slice_.direction for slice_ in record.slices] == [
        "north",
        "west",
        "south",
        "east",
    ]
    north, west, _south, east = record.slices
    assert north.frame_indices == tuple(range(8))
    assert north.cells[0].source_grid_index == 0
    assert north.cells[0].x == 0 and north.cells[0].y == 0
    assert west.cells[0].source_grid_index == 8
    assert west.cells[0].x == 0 and west.cells[0].y == 64
    assert east.cells[-1].source_grid_index == 31
    assert (east.cells[-1].x, east.cells[-1].y) == (448, 192)
    assert record.credit.status == "resolved"
    assert record.credit.match_method == "credits_csv_deterministic_path_candidate"
    assert record.credit.confidence == 0.97
    assert record.credit.matched_reference == "body/bodies/male/run.png"
    assert record.credit.license_tokens == ("CC-BY-SA 3.0",)


def test_retains_duplicate_credit_claims_and_exact_member_match() -> None:
    member = LpcArchiveMemberFact(
        ordinal=1,
        member_path="spritesheets/body/bodies/male/walk.png",
        width=576,
        height=256,
    )
    credits = (
        LpcCredit(
            filename="body/bodies/male/walk.png",
            notes="original",
            authors=("One",),
            licenses=("OGA-BY 3.0", "GPL 3.0"),
            urls=("https://one.test",),
        ),
        LpcCredit(
            filename="body/bodies/male/walk.png",
            notes="later contribution",
            authors=("Two",),
            licenses=("CC0",),
            urls=("https://two.test",),
        ),
    )

    record = _builder(credits=credits).build_record(member)

    assert record is not None
    assert record.credit.match_method == "credits_csv_exact_filename"
    assert record.credit.confidence == 1.0
    assert len(record.credit.claims) == 2
    assert record.credit.authors == ("One", "Two")
    assert record.credit.license_tokens == ("OGA-BY 3.0", "GPL 3.0", "CC0")
    assert record.credit.source_urls == ("https://one.test", "https://two.test")


def test_definition_layer_prefix_is_explicit_lower_confidence_fallback() -> None:
    definition = parse_sheet_definition(
        {
            "name": "Wheelchair",
            "type_name": "wheelchair",
            "layer_1": {
                "zPos": 1,
                "male": "body/wheelchair/adult/background/",
                "custom_animation": "wheelchair",
            },
            "credits": [
                {
                    "file": "body/wheelchair",
                    "notes": "definition-level evidence",
                    "authors": ["Definition Artist"],
                    "licenses": ["CC-BY 4.0"],
                    "urls": ["https://definition.test"],
                }
            ],
        },
        source_path="sheet_definitions/body/wheelchair.json",
    )
    member = LpcArchiveMemberFact(
        ordinal=9,
        member_path="spritesheets/body/wheelchair/adult/background/black.png",
        width=128,
        height=256,
    )

    record = _builder(definitions=(definition,)).build_record(member)

    assert record is not None
    assert record.geometry.status == "layout_join_required"
    assert record.slices == ()
    assert record.credit.status == "resolved"
    assert record.credit.match_method == "sheet_definition_layer_prefix"
    assert record.credit.confidence == 0.85
    assert "body/wheelchair/adult/background/wheelchair.png" in (record.credit.candidate_filenames)
    assert record.credit.matched_reference == "body/wheelchair/adult/background"
    assert record.credit.definition_sources_considered == (
        "sheet_definitions/body/wheelchair.json",
    )
    assert record.credit.license_tokens == ("CC-BY 4.0",)


def test_unresolved_credit_keeps_attempted_candidates_and_status() -> None:
    member = LpcArchiveMemberFact(
        ordinal=11,
        member_path="spritesheets/head/heads/mystery/adult/idle/purple.png",
        width=128,
        height=256,
    )

    record = _builder().build_record(member)

    assert record is not None
    assert record.credit.status == "unresolved"
    assert record.credit.match_method == "none"
    assert record.credit.confidence == 0.0
    assert record.credit.matched_reference is None
    assert record.credit.claims == ()
    assert record.credit.candidate_filenames == (
        "head/heads/mystery/adult/idle/purple.png",
        "head/heads/mystery/adult/idle.png",
    )


def test_preserves_oversize_geometry_and_native_cell_rectangles() -> None:
    member = LpcArchiveMemberFact(
        ordinal=22,
        member_path="spritesheets/weapon/polearm/dragonspear/background/walk/brass.png",
        width=1664,
        height=512,
    )

    record = _builder().build_record(member)

    assert record is not None
    assert record.geometry.status == "oversize"
    assert record.geometry.actual_frame_size == 128
    assert record.geometry.expected_frame_size == 64
    assert record.geometry.actual_frame_count == 13
    assert record.geometry.expected_frame_count == 9
    assert not record.geometry.frame_size_matches_canonical
    assert not record.geometry.frame_count_matches_canonical
    assert len(record.slices) == 4
    assert all(slice_.frame_size == 128 and slice_.frame_count == 13 for slice_ in record.slices)
    assert (record.slices[-1].cells[-1].x, record.slices[-1].cells[-1].y) == (
        1536,
        384,
    )


def test_preserves_malformed_geometry_as_quarantined_record() -> None:
    member = LpcArchiveMemberFact(
        ordinal=23,
        member_path="spritesheets/head/heads/skeleton/adult/halfslash.png",
        width=384,
        height=254,
        inspection_status="media_inspected",
    )

    record = _builder().build_record(member)

    assert record is not None
    assert record.geometry.status == "malformed"
    assert record.geometry.source_width == 384
    assert record.geometry.source_height == 254
    assert "not divisible" in record.geometry.detail
    assert record.slices == ()


def test_valid_noncanonical_grid_is_sliceable_but_flagged() -> None:
    member = LpcArchiveMemberFact(
        ordinal=231,
        member_path="spritesheets/feet/accessory/plate_toe/male/shoot.png",
        width=512,
        height=256,
    )

    record = _builder().build_record(member)

    assert record is not None
    assert record.geometry.status == "noncanonical"
    assert record.geometry.actual_frame_count == 8
    assert record.geometry.expected_frame_count == 13
    assert len(record.slices) == 4
    assert all(len(slice_.cells) == 8 for slice_ in record.slices)


def test_preserves_uninspected_action_member_and_inspection_error() -> None:
    member = LpcArchiveMemberFact(
        ordinal=24,
        member_path="spritesheets/body/bodies/male/idle.png",
        inspection_status="media_invalid",
        inspection_error="decoder rejected payload",
    )

    record = _builder().build_record(member)

    assert record is not None
    assert record.geometry.status == "uninspected"
    assert "decoder rejected payload" in record.geometry.detail
    assert record.inspection_error == "decoder rejected payload"


def test_stable_sheet_id_ignores_archive_root_but_occurrence_id_is_exact() -> None:
    rooted = LpcArchiveMemberFact(
        ordinal=7,
        member_path=f"{ROOT}/spritesheets/body/bodies/male/hurt.png",
        width=384,
        height=64,
    )
    rootless = LpcArchiveMemberFact(
        ordinal=8,
        member_path="spritesheets/body/bodies/male/hurt.png",
        width=384,
        height=64,
    )
    builder = _builder()

    rooted_record = builder.build_record(rooted)
    rootless_record = builder.build_record(rootless)

    assert rooted_record is not None and rootless_record is not None
    assert rooted_record.stable_sheet_id == rootless_record.stable_sheet_id
    assert rooted_record.archive_occurrence_id != rootless_record.archive_occurrence_id
    assert len(rooted_record.slices) == 1
    assert rooted_record.slices[0].direction == "south"


def test_streaming_entry_point_filters_non_sheet_members_without_reordering() -> None:
    members = (
        LpcArchiveMemberFact(0, f"{ROOT}/README.md"),
        LpcArchiveMemberFact(
            1,
            f"{ROOT}/spritesheets/body/bodies/male/idle.png",
            width=128,
            height=256,
        ),
        LpcArchiveMemberFact(
            2,
            f"{ROOT}/spritesheets/body/bodies/male/run.png",
            width=512,
            height=256,
        ),
    )

    records = tuple(
        iter_lpc_manifest_records(
            archive_blob_sha256=ARCHIVE_SHA256,
            members=members,
        )
    )

    assert [record.archive_member_ordinal for record in records] == [1, 2]
    assert [record.source_action for record in records] == ["idle", "run"]
    assert records[0].as_dict()["record_kind"] == "modular_compositing_layer_sheet"
