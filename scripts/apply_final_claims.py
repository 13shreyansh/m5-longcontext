#!/usr/bin/env python3
"""Apply one validated final-protocol record to all local claim surfaces.

This script never publishes or submits. It accepts only a summary under the
ignored final-measurement directory, revalidates its raw logs, and updates the
local manifest, drafts, storyboard and public README as one fail-closed unit.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

try:
    from scripts.build_final_readiness import build_snapshot as build_readiness
    from scripts.build_video_assets import OUTPUT_DIR as VIDEO_DIR
    from scripts.build_video_assets import render as render_video_assets
    from scripts.build_video_captions import OUTPUT as CAPTION_PATH
    from scripts.build_video_captions import render as render_video_captions
    from scripts.prepare_final_claims import record_from_summary_path
    from scripts.verify_champion_manifest import validate_manifest
    from scripts.verify_devpost_draft import validate_draft
    from scripts.verify_video_storyboard import validate_storyboard
except ModuleNotFoundError:  # Standalone execution puts scripts/ on sys.path.
    from build_final_readiness import build_snapshot as build_readiness  # type: ignore[no-redef]
    from build_video_assets import OUTPUT_DIR as VIDEO_DIR  # type: ignore[no-redef]
    from build_video_assets import render as render_video_assets  # type: ignore[no-redef]
    from build_video_captions import OUTPUT as CAPTION_PATH  # type: ignore[no-redef]
    from build_video_captions import render as render_video_captions  # type: ignore[no-redef]
    from prepare_final_claims import record_from_summary_path  # type: ignore[no-redef]
    from verify_champion_manifest import validate_manifest  # type: ignore[no-redef]
    from verify_devpost_draft import validate_draft  # type: ignore[no-redef]
    from verify_video_storyboard import validate_storyboard  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
CHAMPION = ROOT / "docs" / "CHAMPION_MANIFEST.json"
DRAFT_JSON = ROOT / "docs" / "DEVPOST_DRAFT.json"
DRAFT_MD = ROOT / "docs" / "DEVPOST_DRAFT.md"
STORYBOARD_JSON = ROOT / "docs" / "VIDEO_STORYBOARD.json"
STORYBOARD_MD = ROOT / "docs" / "VIDEO_STORYBOARD.md"
PUBLIC_README = (
    ROOT / "docs" / "PUBLIC_README.md"
    if (ROOT / "docs" / "PUBLIC_README.md").is_file()
    else ROOT / "README.md"
)
FINAL_READINESS = ROOT / "docs" / "FINAL_READINESS.json"

PENDING_NARRATION = (
    "The promoted route's contention-controlled batch-32 result is still pending "
    "and is not relabelled from that older run."
)
PENDING_DEVPOST = """**Current boundary:** the promoted direct-head route's contention-controlled
exact batch-32 result is `PENDING_CONTENTION_CONTROLLED_PROTOCOL`. The 98.588985210-second
point belongs to the preceding route and is not relabelled as the promoted
result. No official MFU or combined organizer score is claimed."""
PENDING_STORYBOARD_STATUS = """Status: **provisional — do not record the performance section until the final
contention-controlled row-14 protocol and champion freeze are complete**."""
PENDING_STORYBOARD_GATE = """Current promoted-route exact B32 status:

```text
PENDING_CONTENTION_CONTROLLED_PROTOCOL
```

The existing `98.588985210`-second organizer-default point belongs to the
**preceding** safe projection route. It may be shown only with that label. Do
not call it the promoted direct-head QKV result. After the final protocol:

1. update `docs/CHAMPION_MANIFEST.json` with the verified promoted-route runs;
2. update both storyboard files from that manifest;
3. rerun the storyboard verifier;
4. rebuild and fresh-test the sanitized release; and
5. only then record the final performance segment.

The protocol does not inspect or gate on battery, charging, thermal, or power
mode. Keep the pending card until three finite runs pass the declared runtime
contention and spread gates. Never substitute an estimate."""


def replace_once(text: str, old: str, new: str, label: str) -> str:
    old_count = text.count(old)
    new_count = text.count(new)
    if old_count == 1 and new_count == 0:
        return text.replace(old, new)
    if old_count == 0 and new_count == 1:
        return text
    raise AssertionError(
        f"{label} pending/completed block counts are not exactly one state"
    )


def apply_claims(
    record: dict,
    champion: dict,
    draft: dict,
    storyboard: dict,
    draft_markdown: str,
    storyboard_markdown: str,
    public_readme: str,
) -> tuple[dict, dict, dict, str, str, str]:
    if record.get("status") != "validated_final_claim_values":
        raise AssertionError("claims were not produced by final validation")
    if record.get("row_14_current_route_exact_b32_status") != (
        "complete_contention_controlled_protocol"
    ):
        raise AssertionError("validated record is not a completed protocol")
    if record.get("official_mfu_or_combined_score") is not None:
        raise AssertionError("refusing to apply an invented official score")
    if record.get("power_state_checked") is not False:
        raise AssertionError("validated record must be explicitly power-agnostic")
    if record.get("control_policy") != (
        "power_agnostic_external_compute_contention_monitoring"
    ):
        raise AssertionError("validated record has an unexpected control policy")
    seconds = record.get("row_14_current_route_exact_b32_seconds")
    if not isinstance(seconds, list) or len(seconds) != 3:
        raise AssertionError("validated record does not contain three runs")

    champion = copy.deepcopy(champion)
    draft = copy.deepcopy(draft)
    storyboard = copy.deepcopy(storyboard)
    status = record["row_14_current_route_exact_b32_status"]
    median = record["row_14_current_route_exact_b32_median_seconds"]
    minimum = record["row_14_current_route_exact_b32_min_seconds"]
    maximum = record["row_14_current_route_exact_b32_max_seconds"]
    run_text = ", ".join(f"{value:.3f}" for value in seconds)

    champion["status"] = "final_contention_controlled_exact_measurement_complete"
    champion["snapshot_sgt"] = record["protocol_finished_at_local"]
    row = champion["row_14"]
    for field in (
        "row_14_current_route_exact_b32_status",
        "row_14_current_route_exact_b32_seconds",
        "row_14_current_route_exact_b32_median_seconds",
        "row_14_current_route_exact_b32_min_seconds",
        "row_14_current_route_exact_b32_max_seconds",
        "row_14_current_route_exact_b32_max_over_min",
    ):
        row[field.removeprefix("row_14_")] = record[field]
    row["current_route_exact_b32_all_outputs_finite"] = record[
        "all_outputs_finite"
    ]
    row["current_route_exact_b32_summary_sha256"] = record["summary_sha256"]
    row["current_route_exact_b32_log_sha256"] = record["log_sha256"]
    row["current_route_exact_b32_protocol"] = {
        "runs": record["runs"],
        "seed": record["seed"],
        "cooldown_seconds": record["cooldown_seconds"],
        "matmul_precision": record["matmul_precision"],
        "started_at_local": record["protocol_started_at_local"],
        "finished_at_local": record["protocol_finished_at_local"],
        "control_policy": record["control_policy"],
        "power_state_checked": record["power_state_checked"],
        "runtime_process_poll_seconds": record[
            "runtime_process_poll_seconds"
        ],
        "quiet_window_seconds": record["quiet_window_seconds"],
        "quiet_window_before_each_run": record[
            "quiet_window_before_each_run"
        ],
    }

    draft["claims"]["row_14_current_route_exact_b32_status"] = status
    draft["claims"]["row_14_current_route_exact_b32_seconds"] = seconds
    storyboard["status"] = "final_contention_controlled_exact_measurement_complete"
    storyboard["claims"]["row_14_current_route_exact_b32_status"] = status
    storyboard["claims"]["row_14_current_route_exact_b32_seconds"] = seconds

    complete_narration = (
        "The promoted route completed three contention-controlled batch-32 runs in "
        f"{run_text} seconds, with a {median:.3f}-second median and all outputs "
        "finite. Power state was not inspected or used as a gate. The preceding "
        "route remains separately labelled."
    )
    pending_sections = 0
    completed_sections = 0
    for section in storyboard["sections"]:
        if PENDING_NARRATION in section["narration"]:
            section["narration"] = section["narration"].replace(
                PENDING_NARRATION, complete_narration
            )
            pending_sections += 1
        elif complete_narration in section["narration"]:
            completed_sections += 1
    if (pending_sections, completed_sections) not in {(1, 0), (0, 1)}:
        raise AssertionError(
            "storyboard pending/completed narration counts are not one state"
        )

    complete_devpost = (
        "**Promoted-route contention-controlled result:** the three exact batch-32 runs "
        f"completed in **{run_text} seconds**, with a **{median:.3f}-second "
        f"median** and a {minimum:.3f}–{maximum:.3f}-second range. All returned "
        "values were finite, no external high-CPU Python, Node or Codex "
        "process was observed "
        "by the one-second monitor, and power state was not inspected or used "
        "as a gate. The 98.588985210-second point remains separately labelled "
        "as the preceding route. No official MFU or combined organizer score "
        "is claimed."
    )
    draft_markdown = replace_once(
        draft_markdown, PENDING_DEVPOST, complete_devpost, "Devpost"
    )

    complete_storyboard_status = """Status: **numerical claims synchronized to the final contention-controlled
measurement; prior assembled draft rejected; video generation paused by user**.
Do not record or assemble media until explicit resumption. After resumption,
use ElevenLabs narration with a newly rotated runtime-only credential and
require a complete human watch before treating the replacement as usable."""
    storyboard_markdown = replace_once(
        storyboard_markdown,
        PENDING_STORYBOARD_STATUS,
        complete_storyboard_status,
        "storyboard status",
    )
    complete_storyboard_gate = f"""Current promoted-route exact B32 status:

```text
COMPLETE_CONTENTION_CONTROLLED_PROTOCOL
RUNS_SECONDS={run_text}
MEDIAN_SECONDS={median:.3f}
RANGE_SECONDS={minimum:.3f}–{maximum:.3f}
ALL_OUTPUTS_FINITE=True
POWER_STATE_CHECKED=False
```

The existing `98.588985210`-second organizer-default point belongs to the
**preceding** safe projection route. It may be shown only with that label. Do
not call it the promoted direct-head QKV result. The final performance segment
must show the new three-run median and range from the signed manifest."""
    storyboard_markdown = replace_once(
        storyboard_markdown,
        PENDING_STORYBOARD_GATE,
        complete_storyboard_gate,
        "storyboard exact gate",
    )

    pending_public = (
        "The promoted direct-head QKV route measured `1.019126x` over that "
        "preceding projection route in four balanced `B=1,S=100000` sessions "
        "and still awaits its contention-controlled exact `B=32` measurement."
    )
    complete_public = (
        "The promoted direct-head QKV route measured `1.019126x` over that "
        "preceding projection route in four balanced `B=1,S=100000` sessions. "
        f"Its three contention-controlled exact `B=32` runs were `{run_text}` seconds "
        f"(median `{median:.3f}` seconds, range `{minimum:.3f}`–`{maximum:.3f}`), "
        "with every output finite. Power state was not inspected or used as a gate."
    )
    public_readme = replace_once(
        public_readme, pending_public, complete_public, "public README"
    )
    return (
        champion,
        draft,
        storyboard,
        draft_markdown,
        storyboard_markdown,
        public_readme,
    )


def validate_outputs(
    champion: dict,
    draft: dict,
    storyboard: dict,
    draft_markdown: str,
    storyboard_markdown: str,
) -> tuple[dict[str, str], str]:
    validate_manifest(champion)
    validate_draft(draft, champion, draft_markdown)
    validate_storyboard(storyboard, champion, storyboard_markdown)
    return render_video_assets(champion), render_video_captions(storyboard)


def atomic_write_texts(prepared: list[tuple[Path, str]]) -> None:
    temporary: list[tuple[Path, Path]] = []
    try:
        for path, content in prepared:
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(f".{path.name}.final-claims-tmp")
            if temp.exists():
                raise FileExistsError(f"refusing stale temporary file: {temp}")
            temp.write_text(content)
            temporary.append((temp, path))
        for temp, path in temporary:
            os.replace(temp, path)
    finally:
        for temp, _ in temporary:
            if temp.exists():
                temp.unlink()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    record = record_from_summary_path(args.summary)
    inputs = (
        json.loads(CHAMPION.read_text()),
        json.loads(DRAFT_JSON.read_text()),
        json.loads(STORYBOARD_JSON.read_text()),
        DRAFT_MD.read_text(),
        STORYBOARD_MD.read_text(),
        PUBLIC_README.read_text(),
    )
    outputs = apply_claims(record, *inputs)
    assets, captions = validate_outputs(
        outputs[0], outputs[1], outputs[2], outputs[3], outputs[4]
    )
    readiness = build_readiness(
        outputs[0],
        outputs[1],
        outputs[3],
        outputs[2],
        outputs[4],
        supplied_video_assets=assets,
        supplied_captions=captions,
    )
    if not args.write:
        print(json.dumps(record, indent=2, sort_keys=True))
        print(
            "FINAL_CLAIMS_READY: all local documents and recording assets "
            "validate; add --write to update them"
        )
        return 0
    prepared = [
        (CHAMPION, json.dumps(outputs[0], indent=2, sort_keys=False) + "\n"),
        (DRAFT_JSON, json.dumps(outputs[1], indent=2, sort_keys=False) + "\n"),
        (
            STORYBOARD_JSON,
            json.dumps(outputs[2], indent=2, sort_keys=False) + "\n",
        ),
        (DRAFT_MD, outputs[3]),
        (STORYBOARD_MD, outputs[4]),
        (PUBLIC_README, outputs[5]),
        *((VIDEO_DIR / name, content) for name, content in assets.items()),
        (CAPTION_PATH, captions),
        (FINAL_READINESS, json.dumps(readiness, indent=2, sort_keys=True) + "\n"),
    ]
    atomic_write_texts(prepared)
    print(
        "FINAL_CLAIMS_APPLIED: local documents and recording assets updated; "
        "no external action performed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
