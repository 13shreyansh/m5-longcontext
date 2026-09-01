#!/usr/bin/env python3
"""Build or verify deterministic SRT captions from the video storyboard."""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STORYBOARD = ROOT / "docs" / "VIDEO_STORYBOARD.json"
OUTPUT = ROOT / "docs" / "video_assets" / "narration.srt"


def timestamp(milliseconds: int) -> str:
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, millis = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def sentences(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip())]
    return [part for part in parts if part]


def render(storyboard: dict) -> str:
    sections = storyboard["sections"]
    if not sections or sections[0]["start_second"] != 0:
        raise AssertionError("storyboard does not begin at zero")
    if sections[-1]["end_second"] > storyboard["target_max_seconds"]:
        raise AssertionError("storyboard exceeds declared maximum")

    cues: list[tuple[int, int, str]] = []
    previous_end = 0
    for section in sections:
        start_ms = int(section["start_second"] * 1000)
        end_ms = int(section["end_second"] * 1000)
        if start_ms != previous_end * 1000:
            raise AssertionError(f"timeline gap before {section['id']}")
        if end_ms <= start_ms:
            raise AssertionError(f"non-positive section {section['id']}")
        parts = sentences(section["narration"])
        weights = [max(1, len(re.findall(r"\b[\w.-]+\b", part))) for part in parts]
        total_weight = sum(weights)
        cursor = start_ms
        cumulative = 0
        for index, (part, weight) in enumerate(zip(parts, weights)):
            cumulative += weight
            cue_end = (
                end_ms
                if index == len(parts) - 1
                else start_ms + round((end_ms - start_ms) * cumulative / total_weight)
            )
            if cue_end <= cursor:
                raise AssertionError(f"caption cue collapsed in {section['id']}")
            cues.append((cursor, cue_end, part))
            cursor = cue_end
        previous_end = section["end_second"]

    blocks = []
    for index, (start_ms, end_ms, text) in enumerate(cues, 1):
        wrapped = "\n".join(
            textwrap.wrap(
                text,
                width=68,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
        blocks.append(
            f"{index}\n{timestamp(start_ms)} --> {timestamp(end_ms)}\n{wrapped}"
        )
    return "\n\n".join(blocks) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    storyboard = json.loads(STORYBOARD.read_text())
    expected = render(storyboard)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if args.write:
        OUTPUT.write_text(expected)
    elif not OUTPUT.is_file() or OUTPUT.read_text() != expected:
        raise AssertionError("video caption drift")
    cue_count = expected.count(" --> ")
    print(
        f"video captions: OK ({cue_count} cues, "
        f"{storyboard['sections'][-1]['end_second']}s timeline)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
