#!/usr/bin/env python3
"""Build an ignored, unpublished narrated MP4 from verified video assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
STORYBOARD = ROOT / "docs" / "VIDEO_STORYBOARD.json"
READINESS = ROOT / "docs" / "FINAL_READINESS.json"
SWIFT_BUILDER = ROOT / "scripts" / "build_local_video_draft.swift"


def _within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_video_generation_allowed(readiness: dict[str, object]) -> None:
    boundaries = readiness.get("boundaries")
    if not isinstance(boundaries, dict):
        raise RuntimeError("video generation blocked: readiness boundaries missing")
    if boundaries.get("video_generation_paused_by_user") is not False:
        raise RuntimeError(
            "video generation paused by user; do not create media until explicit "
            "resumption and a rotated runtime-only ElevenLabs credential"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--speech-rate", type=int, default=195)
    args = parser.parse_args()
    if args.speech_rate <= 0:
        raise ValueError("speech-rate must be positive")
    readiness = json.loads(READINESS.read_text())
    try:
        ensure_video_generation_allowed(readiness)
    except RuntimeError as error:
        print(f"VIDEO_GENERATION_BLOCKED: {error}", file=sys.stderr)
        return 3

    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%z")
    output_dir = (args.output_dir or ARTIFACTS / f"local-video-draft-{stamp}").resolve()
    if not _within(output_dir, ARTIFACTS):
        raise ValueError(f"output must remain under ignored {ARTIFACTS}")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    output_dir.mkdir(parents=True)

    storyboard = json.loads(STORYBOARD.read_text())
    sections = storyboard["sections"]
    if not sections or sections[0]["start_second"] != 0:
        raise AssertionError("storyboard must begin at second zero")
    for previous, current in zip(sections, sections[1:]):
        if previous["end_second"] != current["start_second"]:
            raise AssertionError("storyboard sections must be contiguous")
    duration = int(sections[-1]["end_second"])
    if duration > 180:
        raise AssertionError(f"draft exceeds conservative 180-second gate: {duration}")

    narration = "\n\n".join(str(section["narration"]) for section in sections) + "\n"
    narration_path = output_dir / "narration.txt"
    narration_path.write_text(narration)
    audio_path = output_dir / "narration.aiff"
    subprocess.run(
        [
            "/usr/bin/say",
            "-r",
            str(args.speech_rate),
            "-f",
            str(narration_path),
            "-o",
            str(audio_path),
        ],
        cwd=ROOT,
        check=True,
    )
    swift_result = subprocess.run(
        [
            "/usr/bin/swift",
            str(SWIFT_BUILDER),
            str(output_dir),
            str(audio_path),
            str(duration),
        ],
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    print(swift_result.stdout, end="")
    if swift_result.stderr:
        print(swift_result.stderr, end="", file=sys.stderr)
    swift_result.check_returncode()
    duration_prefix = "narration_duration_seconds="
    duration_lines = [
        line for line in swift_result.stdout.splitlines()
        if line.startswith(duration_prefix)
    ]
    if len(duration_lines) != 1:
        raise AssertionError("video builder did not report one narration duration")
    narration_duration = float(duration_lines[0].removeprefix(duration_prefix))
    video_path = output_dir / "track3_local_draft.mp4"
    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise AssertionError("video builder did not produce a nonempty MP4")
    manifest = {
        "schema_version": 1,
        "publication_performed": False,
        "synthetic_narration": True,
        "speech_rate": args.speech_rate,
        "timeline_seconds": duration,
        "narration_duration_seconds": narration_duration,
        "storyboard_sha256": _sha256(STORYBOARD),
        "hook_svg_sha256": _sha256(
            ROOT / "docs" / "video_assets" / "00_hook.svg"
        ),
        "architecture_svg_sha256": _sha256(
            ROOT / "docs" / "video_assets" / "01_architecture.svg"
        ),
        "results_svg_sha256": _sha256(
            ROOT / "docs" / "video_assets" / "02_results.svg"
        ),
        "boundaries_svg_sha256": _sha256(
            ROOT / "docs" / "video_assets" / "03_boundaries.svg"
        ),
        "row14_evidence_svg_sha256": _sha256(
            ROOT / "docs" / "video_assets" / "04_row14_evidence.svg"
        ),
        "reproducibility_svg_sha256": _sha256(
            ROOT / "docs" / "video_assets" / "05_reproducibility.svg"
        ),
        "captions_sha256": _sha256(ROOT / "docs" / "video_assets" / "narration.srt"),
        "narration_aiff_sha256": _sha256(audio_path),
        "draft_mp4_sha256": _sha256(video_path),
        "draft_mp4_bytes": video_path.stat().st_size,
    }
    (output_dir / "DRAFT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"draft_path={video_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
