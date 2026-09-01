import json
import copy

import pytest

from scripts.verify_devpost_draft import (
    DRAFT_PATH,
    MANIFEST_PATH,
    MARKDOWN_PATH,
    main,
    validate_draft,
)
from scripts.apply_final_claims import PENDING_DEVPOST


def inputs():
    return (
        json.loads(DRAFT_PATH.read_text()),
        json.loads(MANIFEST_PATH.read_text()),
        MARKDOWN_PATH.read_text(),
    )


def pending_inputs():
    draft, champion, markdown = inputs()
    draft = copy.deepcopy(draft)
    champion = copy.deepcopy(champion)
    champion["row_14"]["current_route_exact_b32_status"] = (
        "pending_contention_controlled_protocol"
    )
    draft["claims"]["row_14_current_route_exact_b32_status"] = (
        "pending_contention_controlled_protocol"
    )
    draft["claims"]["row_14_current_route_exact_b32_seconds"] = None
    markdown_lines = markdown.splitlines()
    markdown_lines = [
        PENDING_DEVPOST
        if line.startswith("**Promoted-route contention-controlled result:**")
        else line
        for line in markdown_lines
    ]
    return draft, champion, "\n".join(markdown_lines) + "\n"


def test_current_local_draft_matches_champion_and_publication_state():
    assert main() == 0


def test_rejects_invented_official_score():
    draft, champion, markdown = inputs()
    draft["claims"]["official_mfu_or_combined_score"] = 72.5
    with pytest.raises(AssertionError, match="invents an official score"):
        validate_draft(draft, champion, markdown)


def test_rejects_public_url_before_publication():
    draft, champion, markdown = inputs()
    draft["public_repository_url"] = "https://example.com/not-published"
    with pytest.raises(AssertionError, match="unverified action-time field"):
        validate_draft(draft, champion, markdown)


def test_rejects_invented_pending_exact_result():
    draft, champion, markdown = pending_inputs()
    draft["claims"]["row_14_current_route_exact_b32_seconds"] = 96.7
    with pytest.raises(AssertionError, match="invents current-route exact seconds"):
        validate_draft(draft, champion, markdown)


def test_accepts_validated_complete_exact_result():
    draft, champion, markdown = inputs()
    validate_draft(draft, champion, markdown)


def test_rejects_complete_exact_result_with_wrong_run_list():
    draft, champion, markdown = inputs()
    draft["claims"]["row_14_current_route_exact_b32_seconds"] = [
        98.0,
        99.0,
        101.0,
    ]
    with pytest.raises(AssertionError, match="run list drift"):
        validate_draft(draft, champion, markdown)
