#!/usr/bin/env python3
"""Verify that the local Devpost draft matches current evidence and state."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRAFT_PATH = ROOT / "docs" / "DEVPOST_DRAFT.json"
MARKDOWN_PATH = ROOT / "docs" / "DEVPOST_DRAFT.md"
MANIFEST_PATH = ROOT / "docs" / "CHAMPION_MANIFEST.json"


def validate_draft(draft: dict, champion: dict, markdown: str) -> None:
    if draft["schema_version"] != 1:
        raise AssertionError("unsupported Devpost draft schema")
    if draft["status"] != "local_draft_not_submitted":
        raise AssertionError("draft overstates submission status")
    if draft["source_manifest"] != "docs/CHAMPION_MANIFEST.json":
        raise AssertionError("draft source manifest drift")

    claims = draft["claims"]
    rows = champion["rows_1_13"]
    row_14 = champion["row_14"]
    if claims["declared_machine"] != champion["declared_machine"]:
        raise AssertionError("draft machine claim drift")
    if claims["rows_1_13_arithmetic_mean_speedup"] != rows[
        "arithmetic_mean_speedup"
    ]:
        raise AssertionError("draft ordinary-row speed claim drift")
    if claims["rows_1_13_fresh_float32_passed"] != rows["fresh_float32_passed"]:
        raise AssertionError("draft ordinary-row pass count drift")
    if claims["rows_1_13_fresh_float32_total"] != rows["fresh_float32_total"]:
        raise AssertionError("draft ordinary-row total count drift")
    if claims["row_14_qkv_incremental_geometric_mean"] != row_14[
        "qkv_balanced_geometric_mean"
    ]:
        raise AssertionError("draft row-14 QKV claim drift")
    if claims["row_14_preceding_route_high_seconds"] != row_14[
        "preceding_route_organizer_default_seconds"
    ]:
        raise AssertionError("draft preceding-route exact claim drift")
    if claims["row_14_current_route_exact_b32_status"] != row_14[
        "current_route_exact_b32_status"
    ]:
        raise AssertionError("draft current-route status drift")
    if claims["official_mfu_or_combined_score"] is not None:
        raise AssertionError("draft invents an official score")
    if claims["official_mfu_or_combined_score"] != champion["verification"][
        "official_mfu_or_combined_score"
    ]:
        raise AssertionError("draft official-score boundary drift")

    if draft["publication_state"] != champion["publication_state"]:
        raise AssertionError("draft publication state drift")
    if any(draft["publication_state"].values()):
        raise AssertionError("local draft overstates external action")
    for field in (
        "project_name",
        "entrant_team_attribution",
        "public_repository_url",
        "public_video_url",
    ):
        if draft[field] is not None:
            raise AssertionError(f"unverified action-time field is populated: {field}")

    required_placeholders = (
        "[[FINAL PROJECT NAME REQUIRED]]",
        "[[VERIFIED ATTRIBUTION REQUIRED]]",
        "[[PUBLIC SANITIZED REPOSITORY URL REQUIRED]]",
        "[[PUBLIC YOUTUBE URL REQUIRED]]",
    )
    for placeholder in required_placeholders:
        if placeholder not in markdown:
            raise AssertionError(f"draft is missing placeholder: {placeholder}")

    normalized_markdown = " ".join(markdown.split())
    required_judging_boundaries = (
        "## Why it matters and why it is practical",
        "physically unrunnable into a complete 100,000-token execution",
        (
            "broader model or production-inference impact is a potential "
            "application, not a measured claim"
        ),
    )
    for boundary in required_judging_boundaries:
        if boundary not in normalized_markdown:
            raise AssertionError(
                f"draft is missing judge-facing evidence boundary: {boundary}"
            )

    if row_14["current_route_exact_b32_status"] == (
        "pending_contention_controlled_protocol"
    ):
        if claims["row_14_current_route_exact_b32_seconds"] is not None:
            raise AssertionError("pending draft invents current-route exact seconds")
        if "PENDING_CONTENTION_CONTROLLED_PROTOCOL" not in markdown:
            raise AssertionError("draft omits pending row-14 status")
        if "belongs to the preceding route" not in normalized_markdown:
            raise AssertionError("draft omits preceding-route label")
    elif row_14["current_route_exact_b32_status"] == (
        "complete_contention_controlled_protocol"
    ):
        if claims["row_14_current_route_exact_b32_seconds"] != row_14[
            "current_route_exact_b32_seconds"
        ]:
            raise AssertionError("draft exact-row run list drift")
        if "PENDING_CONTENTION_CONTROLLED_PROTOCOL" in markdown:
            raise AssertionError("completed draft retains pending exact-row status")
        median_text = f"{row_14['current_route_exact_b32_median_seconds']:.3f}"
        if median_text not in markdown:
            raise AssertionError("draft omits exact-row median")
    else:
        raise AssertionError("draft has an unsupported exact-row status")

    required_tools = {
        "Codex AI coding agent (exact host-visible model label unavailable)",
        "Python 3.9",
        "PyTorch 2.8",
        "Apple MPS",
        "Metal 4",
        "Apple MLX NAX",
        "Objective-C++",
        "Ninja",
    }
    if set(draft["built_with"]) != required_tools:
        raise AssertionError("draft technology list drift")
    if (
        "Codex AI coding agent (exact host-visible model label unavailable)"
        not in markdown
    ):
        raise AssertionError("draft omits the truthful AI model-label boundary")
    if len(draft["required_action_time_fields"]) < 7:
        raise AssertionError("draft action-time checklist is incomplete")


def main() -> int:
    draft = json.loads(DRAFT_PATH.read_text())
    champion = json.loads(MANIFEST_PATH.read_text())
    markdown = MARKDOWN_PATH.read_text()
    validate_draft(draft, champion, markdown)
    print(
        "Devpost draft: OK (local-only, four action-time identity/URL fields "
        f"unset, row-14 exact status "
        f"{champion['row_14']['current_route_exact_b32_status']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
