#!/usr/bin/env python3
"""Verify that the provisional video plan matches the champion evidence."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STORYBOARD_PATH = ROOT / "docs" / "VIDEO_STORYBOARD.json"
MANIFEST_PATH = ROOT / "docs" / "CHAMPION_MANIFEST.json"
MARKDOWN_PATH = ROOT / "docs" / "VIDEO_STORYBOARD.md"
PUBLIC_SAFE_TESTS_PASSED = 33
PUBLIC_SAFE_TESTS_SKIPPED = 1
AUTHORIZED_INPUT_TESTS_PASSED = 101


def validate_storyboard(storyboard: dict, champion: dict, markdown: str) -> None:
    normalized_markdown = " ".join(markdown.split())
    if storyboard["schema_version"] != 1:
        raise AssertionError("unsupported storyboard schema")
    if storyboard["target_max_seconds"] > 180:
        raise AssertionError("video target exceeds three minutes")
    if storyboard["source_manifest"] != "docs/CHAMPION_MANIFEST.json":
        raise AssertionError("storyboard source manifest drift")

    sections = storyboard["sections"]
    if not sections or sections[0]["start_second"] != 0:
        raise AssertionError("storyboard must begin at second zero")
    previous_end = 0
    narration_words = 0
    for section in sections:
        if section["start_second"] != previous_end:
            raise AssertionError(f"storyboard timeline gap at {section['id']}")
        if section["end_second"] <= section["start_second"]:
            raise AssertionError(f"non-positive section duration: {section['id']}")
        previous_end = section["end_second"]
        narration_words += len(re.findall(r"\b[\w.-]+\b", section["narration"]))
    if previous_end > storyboard["target_max_seconds"]:
        raise AssertionError("storyboard sections exceed target duration")
    if narration_words > 450:
        raise AssertionError(f"narration is too long for three minutes: {narration_words}")

    claims = storyboard["claims"]
    rows = champion["rows_1_13"]
    row_14 = champion["row_14"]
    if claims["rows_1_13_arithmetic_mean_speedup"] != rows[
        "arithmetic_mean_speedup"
    ]:
        raise AssertionError("rows-1-13 video claim drift")
    if claims["row_14_qkv_incremental_geometric_mean"] != row_14[
        "qkv_balanced_geometric_mean"
    ]:
        raise AssertionError("row-14 QKV video claim drift")
    if claims["row_14_preceding_route_high_seconds"] != row_14[
        "preceding_route_organizer_default_seconds"
    ]:
        raise AssertionError("preceding-route exact video claim drift")
    if claims["row_14_current_route_exact_b32_status"] != row_14[
        "current_route_exact_b32_status"
    ]:
        raise AssertionError("current-route exact status drift")
    if claims["official_mfu_or_combined_score"] is not None:
        raise AssertionError("storyboard invents an official score")
    if claims["official_mfu_or_combined_score"] != champion["verification"][
        "official_mfu_or_combined_score"
    ]:
        raise AssertionError("storyboard official-score boundary drift")
    if claims["public_safe_package_tests_passed"] != PUBLIC_SAFE_TESTS_PASSED:
        raise AssertionError("public-safe package pass count drift")
    if claims["public_safe_package_tests_skipped"] != PUBLIC_SAFE_TESTS_SKIPPED:
        raise AssertionError("public-safe package skip count drift")
    if claims["authorized_input_package_tests_passed"] != AUTHORIZED_INPUT_TESTS_PASSED:
        raise AssertionError("authorized-input package pass count drift")
    narration = " ".join(section["narration"] for section in sections)
    if "passes 33 tests with one expected missing-input skip" not in narration:
        raise AssertionError("narration omits public-safe package test boundary")
    if "authorized-input copy passes all 101 tests" not in narration:
        raise AssertionError("narration omits authorized-input package test boundary")
    if claims["video_published"] is not False:
        raise AssertionError("storyboard overstates publication")
    if claims["video_published"] != champion["publication_state"]["video_published"]:
        raise AssertionError("video publication state drift")

    pending = row_14["current_route_exact_b32_status"] == (
        "pending_contention_controlled_protocol"
    )
    if pending:
        if storyboard["status"] != (
            "provisional_pending_contention_controlled_exact_measurement"
        ):
            raise AssertionError("pending champion requires provisional storyboard")
        if claims["row_14_current_route_exact_b32_seconds"] is not None:
            raise AssertionError("pending storyboard invents current-route exact seconds")
        required_text = (
            "The promoted route's contention-controlled batch-32 result is still pending "
            "and is not relabelled from that older run."
        )
        if required_text not in narration:
            raise AssertionError("narration omits pending exact-row boundary")
        if "PENDING_CONTENTION_CONTROLLED_PROTOCOL" not in normalized_markdown:
            raise AssertionError("Markdown storyboard omits pending status card")
    elif row_14["current_route_exact_b32_status"] == (
        "complete_contention_controlled_protocol"
    ):
        if storyboard["status"] != (
            "final_contention_controlled_exact_measurement_complete"
        ):
            raise AssertionError("complete champion requires final storyboard status")
        if claims["row_14_current_route_exact_b32_seconds"] != row_14[
            "current_route_exact_b32_seconds"
        ]:
            raise AssertionError("storyboard exact-row run list drift")
        if "PENDING_CONTENTION_CONTROLLED_PROTOCOL" in normalized_markdown:
            raise AssertionError("completed storyboard retains pending status card")
        median_text = f"{row_14['current_route_exact_b32_median_seconds']:.3f}"
        if median_text not in narration or median_text not in normalized_markdown:
            raise AssertionError("storyboard omits exact-row median")
    else:
        raise AssertionError("storyboard has an unsupported exact-row status")

    if (
        "Do not call it the promoted direct-head QKV result"
        not in normalized_markdown
    ):
        raise AssertionError("Markdown storyboard omits preceding-route warning")
    if len(storyboard["required_final_captures"]) < 7:
        raise AssertionError("storyboard evidence-shot checklist is incomplete")


def main() -> int:
    storyboard = json.loads(STORYBOARD_PATH.read_text())
    champion = json.loads(MANIFEST_PATH.read_text())
    markdown = MARKDOWN_PATH.read_text()
    validate_storyboard(storyboard, champion, markdown)
    duration = storyboard["sections"][-1]["end_second"]
    words = sum(
        len(re.findall(r"\b[\w.-]+\b", section["narration"]))
        for section in storyboard["sections"]
    )
    print(
        f"video storyboard: OK ({duration}s, {words} narration words, "
        f"row-14 exact status "
        f"{champion['row_14']['current_route_exact_b32_status']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
