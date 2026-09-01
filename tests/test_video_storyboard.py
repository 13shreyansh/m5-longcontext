import copy
import json

import pytest

from scripts.verify_video_storyboard import (
    MANIFEST_PATH,
    MARKDOWN_PATH,
    STORYBOARD_PATH,
    main,
    validate_storyboard,
)
from scripts.build_video_assets import render as render_video_assets
from scripts.apply_final_claims import PENDING_NARRATION


def inputs():
    return (
        json.loads(STORYBOARD_PATH.read_text()),
        json.loads(MANIFEST_PATH.read_text()),
        MARKDOWN_PATH.read_text(),
    )


def test_current_storyboard_matches_completed_champion():
    assert main() == 0


def test_rejects_video_longer_than_three_minutes():
    storyboard, champion, markdown = inputs()
    storyboard["target_max_seconds"] = 181
    with pytest.raises(AssertionError, match="exceeds three minutes"):
        validate_storyboard(storyboard, champion, markdown)


def pending_inputs():
    storyboard, champion, markdown = inputs()
    storyboard = copy.deepcopy(storyboard)
    champion = copy.deepcopy(champion)
    champion["row_14"]["current_route_exact_b32_status"] = (
        "pending_contention_controlled_protocol"
    )
    storyboard["status"] = (
        "provisional_pending_contention_controlled_exact_measurement"
    )
    storyboard["claims"]["row_14_current_route_exact_b32_status"] = (
        "pending_contention_controlled_protocol"
    )
    storyboard["claims"]["row_14_current_route_exact_b32_seconds"] = None
    for section in storyboard["sections"]:
        narration = section["narration"]
        if "The promoted route completed three contention-controlled" in narration:
            section["narration"] = (
                narration.split("The promoted route completed", 1)[0]
                + PENDING_NARRATION
            )
    markdown += "\nPENDING_CONTENTION_CONTROLLED_PROTOCOL\n"
    return storyboard, champion, markdown


def test_rejects_invented_current_route_exact_result():
    storyboard, champion, markdown = pending_inputs()
    storyboard["claims"]["row_14_current_route_exact_b32_seconds"] = 96.7
    with pytest.raises(AssertionError, match="invents current-route exact seconds"):
        validate_storyboard(storyboard, champion, markdown)


def test_rejects_relabelled_preceding_route_narration():
    storyboard, champion, markdown = pending_inputs()
    for section in storyboard["sections"]:
        section["narration"] = section["narration"].replace(
            PENDING_NARRATION,
            "The promoted route completed in 98.589 seconds.",
        )
    with pytest.raises(AssertionError, match="pending exact-row boundary"):
        validate_storyboard(storyboard, champion, markdown)


def complete_inputs():
    return inputs()


def test_accepts_validated_complete_exact_result():
    storyboard, champion, markdown = complete_inputs()
    validate_storyboard(storyboard, champion, markdown)


def test_complete_video_assets_show_median_and_range():
    _, champion, _ = complete_inputs()
    assets = render_video_assets(champion)
    results = assets["02_results.svg"]
    boundaries = assets["03_boundaries.svg"]
    hook = assets["00_hook.svg"]
    row_14 = assets["04_row14_evidence.svg"]
    reproducibility = assets["05_reproducibility.svg"]
    row = champion["row_14"]
    expected = (
        f"MEDIAN {row['current_route_exact_b32_median_seconds']:.3f} s · "
        f"RANGE {row['current_route_exact_b32_min_seconds']:.3f}–"
        f"{row['current_route_exact_b32_max_seconds']:.3f} s"
    )
    assert expected in results
    assert "3 CONTROLLED RUNS · ALL OUTPUTS FINITE" in results
    assert "Power state was not inspected or controlled" in boundaries
    assert "18.626 TiB" in hook
    assert f"{row['current_route_exact_b32_median_seconds']:.3f} s" in hook
    assert "RUN 1" in row_14 and "RUN 2" in row_14 and "RUN 3" in row_14
    assert "98.589 s belongs only to the preceding safe route" in row_14
    assert "101 / 101" in reproducibility
    assert "No push · no publication · no submission" in reproducibility
