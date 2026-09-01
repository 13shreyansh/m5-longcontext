#!/usr/bin/env python3
"""Verify the deterministic contents of a sanitized Track 3 release."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "RELEASE_MANIFEST.sha256"

SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"\bsk[_-][A-Za-z0-9_-]{20,}\b"),
    re.compile(rb"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(rb"\bglpat-[A-Za-z0-9_-]{20,}\b"),
    re.compile(rb"\bAIza[0-9A-Za-z_-]{35}\b"),
    re.compile(rb"\bnpm_[A-Za-z0-9]{20,}\b"),
    re.compile(rb"\bBearer[ \t]+[A-Za-z0-9._~+/-]{20,}\b", re.IGNORECASE),
    re.compile(rb"https?://[^/\s:@]+:[^/\s@]+@"),
)

# Local machine paths can disclose an entrant identity or private workspace even
# when they contain no credential. Keep this separate from SECRET_PATTERNS so a
# failure states the correct privacy boundary.
LOCAL_MACHINE_PATH_PATTERNS = (
    # Split POSIX prefixes so this verifier does not flag its own source text.
    re.compile(rb"/" rb"Users/[A-Za-z0-9._-]+/"),
    re.compile(rb"/" rb"home/[A-Za-z0-9._-]+/"),
    re.compile(rb"/" rb"var/folders/"),
    re.compile(rb"[A-Za-z]:\\\\Users\\\\[^\\\r\n]+\\\\"),
)

CURRENT_SURFACE_FORBIDDEN = {
    "README.md": (
        "machine-readable provisional result identity",
        "PENDING_CONTENTION_CONTROLLED_PROTOCOL",
    ),
    "TECHNICAL_REPORT.md": (
        "current authoritative suite passes **110 tests**",
        "recorded from commit `c58926c`",
        "PENDING_CONTENTION_CONTROLLED_PROTOCOL",
        "./scripts/verify_official_artifacts.sh",
        "experiments/benchmark_case8_half_dense.py",
        "current seed-8635 session's geometric mean",
        "current-champion sessions measured **2.499365x**",
    ),
    "docs/CHAMPION_MANIFEST.json": (
        "./scripts/verify_official_artifacts.sh",
    ),
    "docs/FINAL_ACTION_CHECKLIST.md": (
        "./scripts/verify_official_artifacts.sh",
        "scripts/build_video_assets.py",
        "scripts/build_video_captions.py",
        "scripts/build_local_video_draft.py",
    ),
    "solution/README.md": (
        "still needs the contention-controlled exact B32 protocol",
        "ten-minute quiescence gate",
        "PENDING_CONTENTION_CONTROLLED_PROTOCOL",
    ),
    "docs/DEVPOST_DRAFT.md": (
        "PENDING_CONTENTION_CONTROLLED_PROTOCOL",
    ),
    "docs/VIDEO_STORYBOARD.md": (
        "PENDING_CONTENTION_CONTROLLED_PROTOCOL",
    ),
}

CURRENT_SURFACE_REQUIRED = {
    "README.md": (
        "privacy-redacted and never prints process arguments or local paths",
        "PRECONDITION_BLOCKED",
        "no benchmark result was produced",
        "do not commit, paste, screenshot, or publish them",
        "requirements-lock.txt",
        "docs/DEPENDENCY_LICENSES.md",
        "--require-hashes",
        "--only-binary=:all:",
        "## Evidence map",
        "docs/CORRECTNESS_ORACLE.md",
        "docs/PERFORMANCE_MEASUREMENT.md",
        "docs/PRECISION_CONTRACT.md",
        "docs/REPRODUCIBILITY_CONTRACT.md",
        "docs/SOURCE_TO_EXPERIMENT_LEDGER_2026-08-29.md",
        "docs/UPSTREAM_EXPERIMENT_AUDIT_2026-08-29.md",
        "docs/AI_TOOL_DISCLOSURE_2026-08-29.md",
        "Historical milestone files are an audit trail, not the current result identity.",
    ),
    "TECHNICAL_REPORT.md": (
        ".venv/bin/python scripts/verify_release_manifest.py",
        "requirements-lock.txt",
        "docs/DEPENDENCY_LICENSES.md",
        "--require-hashes",
        "--only-binary=:all:",
        "earlier seed-8635 **process-default** session",
        "Those process-default observations are retained as historical",
        "The current reportable repeated setting is the organizer-default `high` set",
        "signed `docs/FINAL_ACTION_CHECKLIST.md`",
    ),
    "docs/CHAMPION_MANIFEST.json": (
        ".venv/bin/python scripts/verify_release_manifest.py",
    ),
    "docs/FINAL_ACTION_CHECKLIST.md": (
        "verified local pre-action checklist",
        "no external action authorized",
        "The nine unresolved gates",
        "The current validators are **pre-action guards**",
        "Video generation remains paused",
        "**newly rotated**",
        "runtime-only credential",
        "separate action-time authorization",
        "2026-09-01 12:00 SGT",
    ),
    "docs/DEPENDENCY_LICENSES.md": (
        "requirements-lock.txt",
        "18-package resolution",
        "Exact target-wheel evidence",
        "--require-hashes",
        "does **not** redistribute",
        "wheels or their installed",
        "directories. Pip obtains them separately",
        "Licence labels below are observations",
        "solution/third_party/",
    ),
    "docs/AI_TOOL_DISCLOSURE_2026-08-29.md": (
        "### Current pre-submission accounting checkpoint",
        "This table supersedes every historical checkpoint below.",
        "pre-submission checkpoint, not a final submission-time",
        "The values are cumulative and must not be added to earlier snapshots.",
        "Exact host-visible model label | Unavailable; not guessed",
        "Child agents used in this Track 3 task | 0",
        "No ElevenLabs request was made",
        "credential exposed in chat was not",
        "used, stored, logged or copied into this repository.",
    ),
    "docs/SUBMISSION_CONTRACT.md": (
        "latest public refresh: **2026-09-01 00:18 SGT**",
        "announcement, new resource, Discussion topic",
        "deadline change or rules change.",
        "September 1, 2026 at 12:00pm",
    ),
    "docs/DEVPOST_DRAFT.md": (
        "prior local video draft is rejected",
        "ElevenLabs narration with a newly rotated runtime-only credential",
        "Codex AI coding agent (exact host-visible model label unavailable)",
    ),
    "docs/VIDEO_ASSETS.md": (
        "prior assembled draft rejected",
        "video generation paused by user",
        "fails before creating an output directory",
    ),
    "docs/VIDEO_STORYBOARD.md": (
        "prior assembled draft rejected",
        "video generation paused by user",
        "newly rotated runtime-only credential",
    ),
    "docs/SOLUTION_MILESTONE_148_2026-08-30.md": (
        "Historical checkpoint — superseded for current status.",
        "pending/provisional, battery/power and then-current test-count wording is not",
    ),
    "docs/SOLUTION_MILESTONE_149_2026-08-30.md": (
        "Historical checkpoint — superseded for current status.",
        "pending/provisional, battery/power and then-current test-count wording is not",
    ),
    "docs/SOLUTION_MILESTONE_150_2026-08-30.md": (
        "Historical checkpoint — superseded for current status.",
        "pending/provisional, battery/power and then-current test-count wording is not",
    ),
    "docs/SOLUTION_MILESTONE_207_2026-08-31.md": (
        "Historical checkpoint — current timing, superseded test count.",
        "superseded by the current `111/111` suite",
    ),
}


def _is_ignored_local_path(relative: str) -> bool:
    path = Path(relative)
    parts = path.parts
    if relative == "official/torch_transformer_benchmark.py":
        return True
    if parts and parts[0] in {".git", ".venv", "venv", "artifacts", ".pytest_cache"}:
        return True
    if "__pycache__" in parts:
        return True
    return path.suffix in {".pyc", ".pyo", ".log"} or path.name == ".DS_Store"


def _regular_files(root: Path) -> set[str]:
    files: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        # Recipient-local environments and caches legitimately contain
        # symlinks (for example, .venv/bin/python3). Prune excluded paths
        # before enforcing the no-symlink rule on distributed content.
        if _is_ignored_local_path(relative):
            continue
        if path.is_symlink():
            raise AssertionError(f"release contains a symlink: {relative}")
        if path.is_file():
            if relative != MANIFEST_NAME:
                files.add(relative)
    return files


def _verify_relative_markdown_links(root: Path) -> None:
    missing: list[str] = []
    for document in root.rglob("*.md"):
        if _is_ignored_local_path(document.relative_to(root).as_posix()):
            continue
        text = document.read_text(errors="replace")
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
            if (
                "://" in target
                or target.startswith("#")
                or target.startswith("mailto:")
            ):
                continue
            clean_target = target.split("#", 1)[0]
            if not clean_target:
                continue
            resolved = (document.parent / clean_target).resolve()
            try:
                resolved.relative_to(root)
            except ValueError:
                missing.append(
                    f"{document.relative_to(root).as_posix()} -> unsafe {target}"
                )
                continue
            if not resolved.exists():
                missing.append(
                    f"{document.relative_to(root).as_posix()} -> missing {target}"
                )
    if missing:
        raise AssertionError("release Markdown link drift: " + "; ".join(missing))


def _verify_current_result_state(root: Path) -> None:
    for relative, forbidden in CURRENT_SURFACE_FORBIDDEN.items():
        text = (root / relative).read_text()
        for fragment in forbidden:
            if fragment in text:
                raise AssertionError(
                    f"obsolete current-facing claim in {relative}: {fragment!r}"
                )

    for relative, required in CURRENT_SURFACE_REQUIRED.items():
        text = (root / relative).read_text()
        for fragment in required:
            if fragment not in text:
                raise AssertionError(
                    f"missing required release safeguard in {relative}: {fragment!r}"
                )

    lock_text = (root / "requirements-lock.txt").read_text()
    lock_lines = [
        line for line in lock_text.splitlines() if line and not line.startswith("#")
    ]
    requirements = [line for line in lock_lines if not line[0].isspace()]
    hashes = re.findall(r"(?m)^    --hash=sha256:([0-9a-f]{64})$", lock_text)
    if (
        len(lock_lines) != 36
        or len(requirements) != 18
        or len(hashes) != 18
        or any(
            line.count("==") != 1 or not line.endswith(" \\")
            for line in requirements
        )
    ):
        raise AssertionError(
            "dependency lock must contain 18 exact requirements and 18 SHA-256 hashes"
        )
    if len(set(requirements)) != 18 or len(set(hashes)) != 18:
        raise AssertionError(
            "dependency lock contains duplicate requirement or wheel hash"
        )
    dependency_text = (root / "docs/DEPENDENCY_LICENSES.md").read_text()
    wheel_section = dependency_text.split("## Exact target-wheel evidence", 1)[1].split(
        "## Direct upstream identities", 1
    )[0]
    documented_wheels = re.findall(
        r"(?m)^\| `([^`]+\.whl)` \| `([0-9a-f]{64})` \|$", wheel_section
    )
    if len(documented_wheels) != 18:
        raise AssertionError("dependency inventory must identify 18 exact target wheels")
    if {digest for _, digest in documented_wheels} != set(hashes):
        raise AssertionError("dependency inventory wheel hashes differ from the lock")

    champion = json.loads((root / "docs/CHAMPION_MANIFEST.json").read_text())
    if champion.get("status") != "final_contention_controlled_exact_measurement_complete":
        raise AssertionError("release champion is not final")
    if champion.get("row_14", {}).get("current_route_exact_b32_status") != (
        "complete_contention_controlled_protocol"
    ):
        raise AssertionError("release row-14 champion is not complete")
    if champion.get("verification", {}).get("pytest_passed") != 111:
        raise AssertionError("release private-suite count is not 111")
    if champion.get("verification", {}).get("official_mfu_or_combined_score") is not None:
        raise AssertionError("release invents an official score")

    devpost = json.loads((root / "docs/DEVPOST_DRAFT.json").read_text())
    storyboard = json.loads((root / "docs/VIDEO_STORYBOARD.json").read_text())
    readiness = json.loads((root / "docs/FINAL_READINESS.json").read_text())
    if devpost.get("claims", {}).get("row_14_current_route_exact_b32_status") != (
        "complete_contention_controlled_protocol"
    ):
        raise AssertionError("release Devpost row-14 state is not complete")
    if storyboard.get("status") != "final_contention_controlled_exact_measurement_complete":
        raise AssertionError("release storyboard state is not final")
    if readiness.get("local_completion_gates", {}).get(
        "final_contention_controlled_row_14_measurement"
    ) is not True:
        raise AssertionError("release readiness row-14 gate is not complete")
    if readiness.get("boundaries", {}).get("external_action_performed_by_this_script") is not False:
        raise AssertionError("release readiness overstates external action")
    if readiness.get("boundaries", {}).get("official_mfu_or_combined_score_claimed") is not False:
        raise AssertionError("release readiness invents an official score")
    if readiness.get("local_human_review_gates", {}).get(
        "video_draft_human_approved"
    ) is not False:
        raise AssertionError("release readiness overstates video approval")
    if readiness.get("boundaries", {}).get("video_generation_paused_by_user") is not True:
        raise AssertionError("release readiness omits the user video pause")
    if "video_draft_human_approved" not in readiness.get("blocking_gate_ids", []):
        raise AssertionError("release readiness omits the video-approval blocker")


def _verify_release_metadata(root: Path, expected_files: set[str]) -> None:
    metadata = json.loads((root / "RELEASE_METADATA.json").read_text())
    champion = json.loads((root / "docs/CHAMPION_MANIFEST.json").read_text())
    expected_keys = {
        "schema_version",
        "source_commit",
        "production_code_commit",
        "champion_status",
        "official_mfu_or_combined_score",
        "organizer_attachments_distributed",
        "existing_private_git_history_distributed",
        "publication_performed",
        "manifest_file_count",
    }
    if set(metadata) != expected_keys:
        raise AssertionError("release metadata drift: unexpected key set")
    if metadata.get("schema_version") != 1:
        raise AssertionError("release metadata drift: schema version")
    if re.fullmatch(r"[0-9a-f]{40}", str(metadata.get("source_commit"))) is None:
        raise AssertionError("release metadata drift: source commit")
    expected_values = {
        "production_code_commit": champion.get("production_code_commit"),
        "champion_status": champion.get("status"),
        "official_mfu_or_combined_score": champion.get("verification", {}).get(
            "official_mfu_or_combined_score"
        ),
        "organizer_attachments_distributed": False,
        "existing_private_git_history_distributed": False,
        "publication_performed": False,
        "manifest_file_count": len(expected_files),
    }
    for key, expected in expected_values.items():
        if metadata.get(key) != expected:
            raise AssertionError(
                f"release metadata drift: {key}={metadata.get(key)!r}, "
                f"expected={expected!r}"
            )


def verify_release(root: Path = ROOT) -> int:
    root = root.resolve()
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise AssertionError(f"missing {MANIFEST_NAME}")

    expected: dict[str, str] = {}
    for line_number, line in enumerate(manifest_path.read_text().splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\n]+)", line)
        if match is None:
            raise AssertionError(f"malformed manifest line {line_number}")
        digest, relative = match.groups()
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or relative == MANIFEST_NAME:
            raise AssertionError(f"unsafe manifest path: {relative}")
        if relative in expected:
            raise AssertionError(f"duplicate manifest path: {relative}")
        expected[relative] = digest

    actual_files = _regular_files(root)
    if actual_files != set(expected):
        missing = sorted(set(expected) - actual_files)
        extra = sorted(actual_files - set(expected))
        raise AssertionError(f"release file-set drift: missing={missing}, extra={extra}")

    if (root / "official" / "torch_transformer_benchmark.py").is_file():
        # A local evaluator may add this after verifying the distributed bundle.
        # It is intentionally outside the signed release manifest.
        pass

    for relative, expected_digest in sorted(expected.items()):
        data = (root / relative).read_bytes()
        actual_digest = hashlib.sha256(data).hexdigest()
        if actual_digest != expected_digest:
            raise AssertionError(
                f"release hash drift: {relative}={actual_digest}, expected={expected_digest}"
            )
        for pattern in SECRET_PATTERNS:
            if pattern.search(data):
                raise AssertionError(f"secret-like content in release file: {relative}")
        for pattern in LOCAL_MACHINE_PATH_PATTERNS:
            if pattern.search(data):
                raise AssertionError(
                    f"local-machine path in release file: {relative}"
                )

    _verify_release_metadata(root, set(expected))
    _verify_relative_markdown_links(root)
    _verify_current_result_state(root)

    print(
        f"sanitized release: OK ({len(expected)} files; "
        "organizer attachment excluded from signed manifest)"
    )
    return 0


def main() -> int:
    return verify_release()


if __name__ == "__main__":
    raise SystemExit(main())
