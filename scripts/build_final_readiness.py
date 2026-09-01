#!/usr/bin/env python3
"""Build or verify the deterministic local final-readiness snapshot.

This report deliberately separates evidence that can be completed locally from
identity, publication, and submission actions that require action-time facts.
It performs no network or external mutation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.build_video_assets import render as render_video_assets
    from scripts.build_video_captions import render as render_video_captions
    from scripts.verify_champion_manifest import validate_manifest
    from scripts.verify_devpost_draft import validate_draft
    from scripts.verify_video_storyboard import validate_storyboard
except ModuleNotFoundError:  # Direct execution places scripts/ on sys.path.
    from build_video_assets import render as render_video_assets
    from build_video_captions import render as render_video_captions
    from verify_champion_manifest import validate_manifest
    from verify_devpost_draft import validate_draft
    from verify_video_storyboard import validate_storyboard


ROOT = Path(__file__).resolve().parents[1]
CHAMPION_PATH = ROOT / "docs" / "CHAMPION_MANIFEST.json"
DRAFT_PATH = ROOT / "docs" / "DEVPOST_DRAFT.json"
DRAFT_MARKDOWN_PATH = ROOT / "docs" / "DEVPOST_DRAFT.md"
STORYBOARD_PATH = ROOT / "docs" / "VIDEO_STORYBOARD.json"
STORYBOARD_MARKDOWN_PATH = ROOT / "docs" / "VIDEO_STORYBOARD.md"
OUTPUT_PATH = ROOT / "docs" / "FINAL_READINESS.json"
ACTION_CHECKLIST_PATH = ROOT / "docs" / "FINAL_ACTION_CHECKLIST.md"
VIDEO_DRAFT_HUMAN_APPROVED = False
VIDEO_GENERATION_PAUSED_BY_USER = True
EXPECTED_BLOCKERS = (
    "project_name",
    "entrant_team_attribution",
    "video_draft_human_approved",
    "public_repository_url",
    "public_video_url",
    "repository_pushed",
    "repository_public",
    "video_published",
    "devpost_submitted",
)


def validate_action_checklist(checklist: str, blockers: list[str]) -> None:
    if tuple(blockers) != EXPECTED_BLOCKERS:
        raise AssertionError("final action blockers differ from the operator handoff")
    for gate in EXPECTED_BLOCKERS:
        if checklist.count(f"`{gate}`") != 1:
            raise AssertionError(
                f"operator handoff must identify gate exactly once: {gate}"
            )
    normalized = " ".join(checklist.split())
    required = (
        "verified local pre-action checklist",
        "no external action authorized",
        "The current validators are **pre-action guards**",
        "signed sanitized tree is the only publication source",
        "Video generation remains paused",
        "**newly rotated** runtime-only credential",
        "separate action-time authorization",
        "Official MFU or combined organizer score: unavailable and not inferred",
        "2026-09-01 12:00 SGT",
    )
    for fragment in required:
        if fragment not in normalized:
            raise AssertionError(f"operator handoff is missing safeguard: {fragment}")
    forbidden = (
        "./scripts/verify_official_artifacts.sh",
        "scripts/build_video_assets.py",
        "scripts/build_video_captions.py",
        "scripts/build_local_video_draft.py",
    )
    for fragment in forbidden:
        if fragment in checklist:
            raise AssertionError(f"operator handoff contains unsafe command: {fragment}")


def build_snapshot(
    champion: dict,
    draft: dict,
    draft_markdown: str,
    storyboard: dict,
    storyboard_markdown: str,
    root: Path = ROOT,
    supplied_video_assets: dict[str, str] | None = None,
    supplied_captions: str | None = None,
) -> dict[str, object]:
    """Return readiness derived only from already-validated local sources."""

    validate_manifest(champion, root)
    validate_draft(draft, champion, draft_markdown)
    validate_storyboard(storyboard, champion, storyboard_markdown)

    expected_assets = render_video_assets(champion)
    observed_assets = supplied_video_assets
    if observed_assets is None:
        observed_assets = {}
        for relative in expected_assets:
            path = root / "docs" / "video_assets" / relative
            observed_assets[relative] = path.read_text() if path.is_file() else ""
    if observed_assets != expected_assets:
        raise AssertionError("video evidence asset drift")
    expected_captions = render_video_captions(storyboard)
    if supplied_captions is None:
        captions_path = root / "docs" / "video_assets" / "narration.srt"
        supplied_captions = captions_path.read_text() if captions_path.is_file() else ""
    if supplied_captions != expected_captions:
        raise AssertionError("video caption drift")

    row_14_complete = (
        champion["row_14"]["current_route_exact_b32_status"]
        == "complete_contention_controlled_protocol"
    )
    identity_fields = {
        "project_name": draft["project_name"] is not None,
        "entrant_team_attribution": draft["entrant_team_attribution"] is not None,
    }
    publication_fields = {
        "public_repository_url": draft["public_repository_url"] is not None,
        "public_video_url": draft["public_video_url"] is not None,
        "repository_pushed": bool(champion["publication_state"]["pushed"]),
        "repository_public": bool(
            champion["publication_state"]["public_repository"]
        ),
        "video_published": bool(champion["publication_state"]["video_published"]),
        "devpost_submitted": bool(champion["publication_state"]["submitted"]),
    }
    local_human_review_fields = {
        "video_draft_human_approved": VIDEO_DRAFT_HUMAN_APPROVED,
    }
    blockers = []
    if not row_14_complete:
        blockers.append("final_contention_controlled_row_14_measurement")
    blockers.extend(key for key, complete in identity_fields.items() if not complete)
    blockers.extend(
        key for key, complete in local_human_review_fields.items() if not complete
    )
    blockers.extend(key for key, complete in publication_fields.items() if not complete)

    if row_14_complete and not any(
        not complete for complete in identity_fields.values()
    ) and not any(not complete for complete in local_human_review_fields.values()):
        local_status = "local_claims_complete_external_actions_pending"
    else:
        local_status = "local_finalization_incomplete"

    return {
        "schema_version": 1,
        "status": local_status,
        "sources": {
            "champion_manifest": "docs/CHAMPION_MANIFEST.json",
            "devpost_draft": "docs/DEVPOST_DRAFT.json",
            "video_storyboard": "docs/VIDEO_STORYBOARD.json",
            "submission_contract": "docs/SUBMISSION_CONTRACT.md",
        },
        "verified_local_surfaces": {
            "champion_source_hashes": True,
            "devpost_claims_match_champion": True,
            "video_storyboard_matches_champion": True,
            "video_assets_and_captions_current": True,
        },
        "local_human_review_gates": local_human_review_fields,
        "local_completion_gates": {
            "rows_1_13_evidence_complete": (
                champion["rows_1_13"]["fresh_float32_passed"]
                == champion["rows_1_13"]["fresh_float32_total"]
            ),
            "final_contention_controlled_row_14_measurement": row_14_complete,
            "official_mfu_or_combined_score_available": (
                champion["verification"]["official_mfu_or_combined_score"]
                is not None
            ),
        },
        "action_time_identity_gates": identity_fields,
        "external_action_gates": publication_fields,
        "blocking_gate_ids": blockers,
        "boundaries": {
            "ready_for_external_submission": not blockers,
            "external_action_performed_by_this_script": False,
            "video_generation_paused_by_user": VIDEO_GENERATION_PAUSED_BY_USER,
            "organizer_attachment_redistribution_authorized": False,
            "official_mfu_or_combined_score_claimed": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    champion = json.loads(CHAMPION_PATH.read_text())
    draft = json.loads(DRAFT_PATH.read_text())
    storyboard = json.loads(STORYBOARD_PATH.read_text())
    expected = json.dumps(
        build_snapshot(
            champion,
            draft,
            DRAFT_MARKDOWN_PATH.read_text(),
            storyboard,
            STORYBOARD_MARKDOWN_PATH.read_text(),
        ),
        indent=2,
        sort_keys=True,
    ) + "\n"
    if args.write:
        OUTPUT_PATH.write_text(expected)
    elif not OUTPUT_PATH.is_file() or OUTPUT_PATH.read_text() != expected:
        raise AssertionError("final-readiness snapshot drift")

    snapshot = json.loads(expected)
    validate_action_checklist(
        ACTION_CHECKLIST_PATH.read_text(), snapshot["blocking_gate_ids"]
    )
    print(
        "final readiness: OK "
        f"({len(snapshot['blocking_gate_ids'])} explicit blockers, "
        f"status={snapshot['status']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
