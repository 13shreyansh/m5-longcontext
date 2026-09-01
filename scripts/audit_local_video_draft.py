#!/usr/bin/env python3
"""Audit the unpublished local Track 3 video draft without publishing it."""

from __future__ import annotations

import aifc
import argparse
import array
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(*arguments: str) -> str:
    completed = subprocess.run(
        arguments,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def analyze_audio(path: Path) -> dict[str, float | int]:
    with aifc.open(str(path), "rb") as audio:
        channels = audio.getnchannels()
        sample_width = audio.getsampwidth()
        sample_rate = audio.getframerate()
        frame_count = audio.getnframes()
        if channels != 1 or sample_width != 2 or sample_rate != 22050:
            raise AssertionError(
                "decoded narration must be mono 16-bit PCM at 22050 Hz"
            )
        frames = audio.readframes(frame_count)

    samples = array.array("h")
    samples.frombytes(frames)
    if sys.byteorder == "little":
        samples.byteswap()
    if not samples:
        raise AssertionError("decoded narration is empty")

    block_size = sample_rate
    silent_threshold = 80.0
    longest_silent_blocks = 0
    current_silent_blocks = 0
    block_rms = []
    clipped_samples = 0
    peak = 0
    sum_squares = 0
    for sample in samples:
        absolute = abs(sample)
        peak = max(peak, absolute)
        clipped_samples += int(absolute >= 32760)
        sum_squares += sample * sample
    for start in range(0, len(samples), block_size):
        block = samples[start : start + block_size]
        rms = math.sqrt(sum(sample * sample for sample in block) / len(block))
        block_rms.append(rms)
        if rms < silent_threshold:
            current_silent_blocks += 1
            longest_silent_blocks = max(longest_silent_blocks, current_silent_blocks)
        else:
            current_silent_blocks = 0

    overall_rms = math.sqrt(sum_squares / len(samples))
    if overall_rms < 100:
        raise AssertionError(f"decoded narration is effectively silent: rms={overall_rms}")
    if longest_silent_blocks > 3:
        raise AssertionError(
            "decoded narration has more than three consecutive silent seconds"
        )
    if clipped_samples / len(samples) > 0.0001:
        raise AssertionError("decoded narration has excessive clipped samples")

    return {
        "audio_duration_seconds": frame_count / sample_rate,
        "audio_peak": peak,
        "audio_overall_rms": overall_rms,
        "audio_minimum_one_second_rms": min(block_rms),
        "audio_longest_silent_whole_seconds": longest_silent_blocks,
        "audio_clipped_sample_count": clipped_samples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("draft_directory", type=Path)
    args = parser.parse_args()
    directory = args.draft_directory.resolve()
    manifest_path = directory / "DRAFT_MANIFEST.json"
    video = directory / "track3_local_draft.mp4"
    source_audio = directory / "narration.aiff"
    manifest = json.loads(manifest_path.read_text())

    if manifest["publication_performed"] is not False:
        raise AssertionError("draft manifest overstates publication state")
    if sha256(video) != manifest["draft_mp4_sha256"]:
        raise AssertionError("draft MP4 hash drift")
    if video.stat().st_size != manifest["draft_mp4_bytes"]:
        raise AssertionError("draft MP4 size drift")
    if sha256(source_audio) != manifest["narration_aiff_sha256"]:
        raise AssertionError("source narration hash drift")
    source_hashes = {
        "hook_svg_sha256": ROOT / "docs/video_assets/00_hook.svg",
        "architecture_svg_sha256": ROOT / "docs/video_assets/01_architecture.svg",
        "results_svg_sha256": ROOT / "docs/video_assets/02_results.svg",
        "boundaries_svg_sha256": ROOT / "docs/video_assets/03_boundaries.svg",
        "row14_evidence_svg_sha256": ROOT
        / "docs/video_assets/04_row14_evidence.svg",
        "reproducibility_svg_sha256": ROOT
        / "docs/video_assets/05_reproducibility.svg",
        "captions_sha256": ROOT / "docs/video_assets/narration.srt",
        "storyboard_sha256": ROOT / "docs/VIDEO_STORYBOARD.json",
    }
    for manifest_key, source_path in source_hashes.items():
        if sha256(source_path) != manifest[manifest_key]:
            raise AssertionError(f"video source hash drift: {source_path.name}")
    if manifest["timeline_seconds"] != 178 or manifest["speech_rate"] != 195:
        raise AssertionError("video timeline or narration-rate contract drift")

    # Spotlight metadata can lag behind a newly written file. Treat it as a
    # codec cross-check when available; AVFoundation below is authoritative for
    # duration, track count, dimensions, and frame decoding.
    metadata = run(
        "mdls",
        "-name",
        "kMDItemCodecs",
        str(video),
    )
    indexed_codecs = re.search(r'"MPEG-4 AAC"', metadata) is not None and re.search(
        r'"H\.264"', metadata
    ) is not None

    with tempfile.TemporaryDirectory(prefix="track3-video-audit-") as temporary:
        decoded_audio = Path(temporary) / "decoded.aiff"
        run(
            "afconvert",
            str(video),
            "-o",
            str(decoded_audio),
            "-f",
            "AIFF",
            "-d",
            "BEI16@22050",
            "-c",
            "1",
            "--read-track",
            "0",
        )
        audio = analyze_audio(decoded_audio)
    if abs(
        audio["audio_duration_seconds"] - manifest["narration_duration_seconds"]
    ) > 0.01:
        raise AssertionError("decoded narration duration drift")

    frame_output = run(
        "swift",
        str(ROOT / "scripts" / "audit_local_video_frames.swift"),
        str(video),
    )
    if "blank_or_solid_frame_samples=0" not in frame_output:
        raise AssertionError("frame audit did not close the blank-frame gate")
    required_stream_boundaries = (
        "video_duration_seconds=178.000",
        "video_tracks=1",
        "audio_tracks=1",
        "video_dimensions=1920x1080",
    )
    for boundary in required_stream_boundaries:
        if boundary not in frame_output:
            raise AssertionError(f"missing decoded stream boundary: {boundary}")

    print(json.dumps(audio, indent=2, sort_keys=True))
    print(f"spotlight_codec_crosscheck={'OK' if indexed_codecs else 'UNAVAILABLE'}")
    print(frame_output, end="")
    print("local_video_draft_audit=OK")
    print("publication_performed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
