#!/usr/bin/env python3
"""Build or verify deterministic, manifest-backed 16:9 video evidence cards."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "CHAMPION_MANIFEST.json"
OUTPUT_DIR = ROOT / "docs" / "video_assets"

BG = "#07090d"
PANEL = "#11151c"
CYAN = "#25f4ee"
RED = "#fe2c55"
WHITE = "#f7f9fc"
MUTED = "#aeb8c5"
GREEN = "#52e08a"
YELLOW = "#ffd166"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def shell(title: str, subtitle: str, body: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1920" height="1080" viewBox="0 0 1920 1080" role="img" aria-label="{esc(title)}">
  <rect width="1920" height="1080" fill="{BG}"/>
  <rect x="0" y="0" width="18" height="1080" fill="{CYAN}"/>
  <rect x="18" y="0" width="7" height="1080" fill="{RED}"/>
  <text x="92" y="112" fill="{WHITE}" font-family="-apple-system, BlinkMacSystemFont, Inter, sans-serif" font-size="64" font-weight="750">{esc(title)}</text>
  <text x="94" y="168" fill="{MUTED}" font-family="-apple-system, BlinkMacSystemFont, Inter, sans-serif" font-size="27">{esc(subtitle)}</text>
  {body}
  <text x="94" y="1028" fill="{MUTED}" font-family="ui-monospace, SFMono-Regular, Menlo, monospace" font-size="20">TikTok TechJam 2026 · Track 3 · evidence card generated from CHAMPION_MANIFEST.json</text>
</svg>
'''


def hook_card(manifest: dict) -> str:
    row = manifest["row_14"]
    shape = row["shape"]
    body = f'''
  <rect x="92" y="230" width="1734" height="650" rx="38" fill="{PANEL}"/>
  <text x="142" y="310" fill="{RED}" font-family="-apple-system, sans-serif" font-size="29" font-weight="800">THE EXPLICIT REFERENCE CANNOT FIT</text>
  <text x="142" y="475" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="154" font-weight="900">18.626 TiB</text>
  <text x="146" y="535" fill="{MUTED}" font-family="-apple-system, sans-serif" font-size="31">one float32 attention-score tensor</text>
  <text x="146" y="592" fill="{MUTED}" font-family="ui-monospace, monospace" font-size="27">B{shape[0]} · S{shape[1]} · D{shape[2]} · H{shape[3]} · L{shape[5]}</text>
  <path d="M850 490 H1035" stroke="{CYAN}" stroke-width="9"/><path d="M1012 462 L1050 490 L1012 518" fill="none" stroke="{CYAN}" stroke-width="9"/>
  <text x="1100" y="310" fill="{GREEN}" font-family="-apple-system, sans-serif" font-size="29" font-weight="800">THE BOUNDED ROUTE COMPLETES</text>
  <text x="1100" y="455" fill="{GREEN}" font-family="-apple-system, sans-serif" font-size="122" font-weight="900">{row['current_route_exact_b32_median_seconds']:.3f} s</text>
  <text x="1104" y="518" fill="{MUTED}" font-family="-apple-system, sans-serif" font-size="31">local three-run median</text>
  <text x="1104" y="575" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="30" font-weight="650">all 3.2768 billion outputs finite</text>
  <rect x="142" y="688" width="1534" height="116" rx="24" fill="#16272b"/>
  <text x="190" y="740" fill="{CYAN}" font-family="-apple-system, sans-serif" font-size="28" font-weight="800">BOUND THE MEMORY · KEEP THE PYTORCH INTERFACE · PROVE THE RESULT</text>
  <text x="190" y="780" fill="{MUTED}" font-family="-apple-system, sans-serif" font-size="25">One Apple M5 Pro · no official MFU · no full-length explicit-reference claim</text>'''
    return shell(
        "100,000 tokens — turn an impossible tensor into a complete run",
        esc(manifest["declared_machine"]) + " · published 100,000-token stress shape",
        body.lstrip("\n"),
    )


def architecture_card(manifest: dict) -> str:
    row = manifest["row_14"]
    shape = row["shape"]
    attention = row["attention_tile"]
    qkv = row["qkv_tile"]
    body = f'''
  <rect x="92" y="230" width="360" height="220" rx="28" fill="{PANEL}" stroke="{CYAN}" stroke-width="3"/>
  <text x="132" y="292" fill="{CYAN}" font-family="-apple-system, sans-serif" font-size="25" font-weight="700">01 · INPUT CONTRACT</text>
  <text x="132" y="347" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="34" font-weight="650">PyTorch float32</text>
  <text x="132" y="393" fill="{MUTED}" font-family="ui-monospace, monospace" font-size="22">B{shape[0]} · S{shape[1]} · D{shape[2]}</text>
  <path d="M452 340 H518" stroke="{MUTED}" stroke-width="6"/><path d="M508 326 L530 340 L508 354" fill="none" stroke="{MUTED}" stroke-width="6"/>
  <rect x="530" y="230" width="390" height="220" rx="28" fill="{PANEL}" stroke="{RED}" stroke-width="3"/>
  <text x="570" y="292" fill="{RED}" font-family="-apple-system, sans-serif" font-size="25" font-weight="700">02 · DIRECT-HEAD QKV</text>
  <text x="570" y="347" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="34" font-weight="650">One native write</text>
  <text x="570" y="393" fill="{MUTED}" font-family="ui-monospace, monospace" font-size="22">BM{qkv[0]} · BN{qkv[1]} · BK{qkv[2]}</text>
  <path d="M920 340 H986" stroke="{MUTED}" stroke-width="6"/><path d="M976 326 L998 340 L976 354" fill="none" stroke="{MUTED}" stroke-width="6"/>
  <rect x="998" y="230" width="420" height="220" rx="28" fill="{PANEL}" stroke="{CYAN}" stroke-width="3"/>
  <text x="1038" y="292" fill="{CYAN}" font-family="-apple-system, sans-serif" font-size="25" font-weight="700">03 · BOUNDED ATTENTION</text>
  <text x="1038" y="347" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="34" font-weight="650">Online causal softmax</text>
  <text x="1038" y="393" fill="{MUTED}" font-family="ui-monospace, monospace" font-size="22">BQ{attention[0]} · BK{attention[1]} · D{attention[2]}</text>
  <path d="M1418 340 H1484" stroke="{MUTED}" stroke-width="6"/><path d="M1474 326 L1496 340 L1474 354" fill="none" stroke="{MUTED}" stroke-width="6"/>
  <rect x="1496" y="230" width="330" height="220" rx="28" fill="{PANEL}" stroke="{GREEN}" stroke-width="3"/>
  <text x="1536" y="292" fill="{GREEN}" font-family="-apple-system, sans-serif" font-size="25" font-weight="700">04 · OUTPUT</text>
  <text x="1536" y="347" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="34" font-weight="650">Float32 boundary</text>
  <text x="1536" y="393" fill="{MUTED}" font-family="-apple-system, sans-serif" font-size="22">Residual · FFN · norm</text>
  <rect x="92" y="520" width="1734" height="362" rx="32" fill="{PANEL}"/>
  <text x="142" y="590" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="34" font-weight="700">Why the 100,000-token row becomes runnable</text>
  <text x="142" y="660" fill="{CYAN}" font-family="-apple-system, sans-serif" font-size="54" font-weight="800">Never materialize the quadratic score tensor.</text>
  <text x="142" y="724" fill="{MUTED}" font-family="-apple-system, sans-serif" font-size="29">Process one batch item at a time · stream key blocks · keep float32 online-softmax state.</text>
  <rect x="142" y="772" width="740" height="68" rx="18" fill="#16272b"/>
  <text x="176" y="816" fill="{GREEN}" font-family="-apple-system, sans-serif" font-size="25" font-weight="700">SUPPORTED → Apple M5 Pro · Metal 4 · MLX NAX</text>
  <rect x="920" y="772" width="840" height="68" rx="18" fill="#27171c"/>
  <text x="954" y="816" fill="{RED}" font-family="-apple-system, sans-serif" font-size="25" font-weight="700">UNSUPPORTED → fail closed to the preceding bounded route</text>'''
    return shell(
        "A bounded Transformer path for one personal Apple GPU",
        "The published stress row stays PyTorch-compatible without allocating O(S²) attention scores.",
        body,
    )


def results_card(manifest: dict) -> str:
    rows = manifest["rows_1_13"]
    row = manifest["row_14"]
    sessions = "  ·  ".join(f"{value:.6f}×" for value in rows["complete_session_speedups"])
    status = row["current_route_exact_b32_status"]
    if status == "pending_contention_controlled_protocol":
        exact_line = "PROMOTED EXACT B32: PENDING CONTROLLED PROTOCOL"
        exact_colour = YELLOW
        boundary_badge = "DO NOT RELABEL THE 98.589-SECOND POINT"
        boundary_badge_fill = "#2a2516"
    elif status == "complete_contention_controlled_protocol":
        exact_line = (
            "PROMOTED EXACT B32: "
            f"MEDIAN {row['current_route_exact_b32_median_seconds']:.3f} s · "
            f"RANGE {row['current_route_exact_b32_min_seconds']:.3f}–"
            f"{row['current_route_exact_b32_max_seconds']:.3f} s"
        )
        exact_colour = GREEN
        boundary_badge = "3 CONTROLLED RUNS · ALL OUTPUTS FINITE"
        boundary_badge_fill = "#16272b"
    else:
        raise AssertionError("unsupported exact-row status for video assets")
    body = f'''
  <rect x="92" y="230" width="825" height="610" rx="34" fill="{PANEL}" stroke="{CYAN}" stroke-width="3"/>
  <text x="142" y="302" fill="{CYAN}" font-family="-apple-system, sans-serif" font-size="27" font-weight="750">PUBLISHED ROWS 1–13</text>
  <text x="142" y="426" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="112" font-weight="850">{rows['arithmetic_mean_speedup']:.6f}×</text>
  <text x="142" y="472" fill="{MUTED}" font-family="-apple-system, sans-serif" font-size="27">arithmetic-mean speedup vs organizer baseline</text>
  <line x1="142" y1="520" x2="867" y2="520" stroke="#313944" stroke-width="2"/>
  <text x="142" y="584" fill="{WHITE}" font-family="ui-monospace, monospace" font-size="26">{esc(sessions)}</text>
  <text x="142" y="642" fill="{GREEN}" font-family="-apple-system, sans-serif" font-size="42" font-weight="750">{rows['fresh_float32_passed']}/{rows['fresh_float32_total']} fresh checks passed</text>
  <text x="142" y="704" fill="{MUTED}" font-family="-apple-system, sans-serif" font-size="25">Three complete synchronized sessions · matmul precision: {esc(rows['matmul_precision'])}</text>
  <rect x="142" y="758" width="400" height="52" rx="15" fill="#16272b"/>
  <text x="170" y="793" fill="{GREEN}" font-family="-apple-system, sans-serif" font-size="22" font-weight="700">EVERY ROW FASTER · ALL PASS</text>
  <rect x="949" y="230" width="877" height="610" rx="34" fill="{PANEL}" stroke="{RED}" stroke-width="3"/>
  <text x="999" y="302" fill="{RED}" font-family="-apple-system, sans-serif" font-size="27" font-weight="750">PUBLISHED ROW 14 · STRESS SHAPE</text>
  <text x="999" y="394" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="65" font-weight="820">{row['qkv_balanced_geometric_mean']:.6f}×</text>
  <text x="1320" y="386" fill="{MUTED}" font-family="-apple-system, sans-serif" font-size="24">promoted QKV incremental gain</text>
  <text x="999" y="454" fill="{GREEN}" font-family="-apple-system, sans-serif" font-size="32" font-weight="700">{row['qkv_positive_pairs']}/{row['qkv_total_pairs']} balanced pairs positive</text>
  <line x1="999" y1="505" x2="1776" y2="505" stroke="#313944" stroke-width="2"/>
  <text x="999" y="574" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="43" font-weight="750">98.589 s</text>
  <text x="1212" y="570" fill="{MUTED}" font-family="-apple-system, sans-serif" font-size="24">preceding route · organizer-default high</text>
  <text x="999" y="636" fill="{exact_colour}" font-family="ui-monospace, monospace" font-size="25" font-weight="700">{esc(exact_line)}</text>
  <text x="999" y="696" fill="{MUTED}" font-family="-apple-system, sans-serif" font-size="25">Full explicit reference: infeasible 18.626 TiB score tensor</text>
  <rect x="999" y="758" width="690" height="52" rx="15" fill="{boundary_badge_fill}"/>
  <text x="1027" y="793" fill="{exact_colour}" font-family="-apple-system, sans-serif" font-size="22" font-weight="700">{esc(boundary_badge)}</text>'''
    return shell(
        "Measured results — every number keeps its boundary",
        esc(manifest["declared_machine"]) + " · synchronized local measurements · no official MFU inferred",
        body,
    )


def boundaries_card(manifest: dict) -> str:
    exact_status = manifest["row_14"]["current_route_exact_b32_status"]
    if exact_status == "pending_contention_controlled_protocol":
        exact_boundary = "× No promoted exact-B32 controlled latency yet"
    elif exact_status == "complete_contention_controlled_protocol":
        exact_boundary = "× Power state was not inspected or controlled"
    else:
        raise AssertionError("unsupported exact-row status for video assets")
    body = f'''
  <rect x="92" y="230" width="825" height="650" rx="34" fill="{PANEL}" stroke="{GREEN}" stroke-width="3"/>
  <text x="142" y="302" fill="{GREEN}" font-family="-apple-system, sans-serif" font-size="30" font-weight="800">WHAT THE EVIDENCE PROVES</text>
  <text x="142" y="382" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="31" font-weight="650">✓ Rows 1–13 beat baseline in three full sessions</text>
  <text x="142" y="449" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="31" font-weight="650">✓ {manifest['rows_1_13']['fresh_float32_passed']}/{manifest['rows_1_13']['fresh_float32_total']} fresh float32 comparisons pass</text>
  <text x="142" y="516" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="31" font-weight="650">✓ Row 14 executes with bounded attention memory</text>
  <text x="142" y="583" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="31" font-weight="650">✓ S8192 reference + continuity checks pass</text>
  <text x="142" y="650" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="31" font-weight="650">✓ Sources, licences + six hashes are locked</text>
  <text x="142" y="717" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="31" font-weight="650">✓ Unsupported native paths fail closed</text>
  <rect x="142" y="775" width="685" height="58" rx="16" fill="#16272b"/>
  <text x="172" y="814" fill="{GREEN}" font-family="-apple-system, sans-serif" font-size="23" font-weight="700">DEVICE-SPECIFIC · REPEATABLE · AUDITABLE</text>
  <rect x="949" y="230" width="877" height="650" rx="34" fill="{PANEL}" stroke="{RED}" stroke-width="3"/>
  <text x="999" y="302" fill="{RED}" font-family="-apple-system, sans-serif" font-size="30" font-weight="800">WHAT THE EVIDENCE DOES NOT PROVE</text>
  <text x="999" y="382" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="31" font-weight="650">× No official MFU or combined organizer score</text>
  <text x="999" y="449" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="31" font-weight="650">× No full S100000 explicit-reference equality</text>
  <text x="999" y="516" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="31" font-weight="650">× No CUDA, ROCm or other-Mac portability claim</text>
  <text x="999" y="583" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="31" font-weight="650">× No causal claim from cross-session latency drift</text>
  <text x="999" y="650" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="31" font-weight="650">{esc(exact_boundary)}</text>
  <text x="999" y="717" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="31" font-weight="650">× No redistribution right for organizer attachments</text>
  <rect x="999" y="775" width="735" height="58" rx="16" fill="#27171c"/>
  <text x="1029" y="814" fill="{RED}" font-family="-apple-system, sans-serif" font-size="23" font-weight="700">LIMITATIONS STAY VISIBLE IN THE SAME FRAME</text>'''
    return shell(
        "Strong evidence is specific about what it does not prove",
        "Correctness first · whole-model timing second · publication claims last",
        body,
    )


def row_14_evidence_card(manifest: dict) -> str:
    row = manifest["row_14"]
    run_seconds = row["current_route_exact_b32_seconds"]
    run_lines = "".join(
        f'<text x="170" y="{405 + index * 100}" fill="{WHITE}" font-family="ui-monospace, monospace" font-size="48" font-weight="750">RUN {index + 1}  {value:.3f} s</text>'
        for index, value in enumerate(run_seconds)
    )
    body = f'''
  <rect x="92" y="230" width="760" height="650" rx="34" fill="{PANEL}" stroke="{CYAN}" stroke-width="3"/>
  <text x="142" y="306" fill="{CYAN}" font-family="-apple-system, sans-serif" font-size="28" font-weight="800">EXACT B32 / S100000 PROTOCOL</text>
  {run_lines}
  <line x1="142" y1="700" x2="802" y2="700" stroke="#313944" stroke-width="2"/>
  <text x="142" y="770" fill="{GREEN}" font-family="-apple-system, sans-serif" font-size="43" font-weight="850">MEDIAN {row['current_route_exact_b32_median_seconds']:.3f} s</text>
  <text x="142" y="822" fill="{MUTED}" font-family="-apple-system, sans-serif" font-size="25">max/min {row['current_route_exact_b32_max_over_min']:.6f} · limit 1.15</text>
  <rect x="884" y="230" width="942" height="650" rx="34" fill="{PANEL}" stroke="{RED}" stroke-width="3"/>
  <text x="934" y="306" fill="{RED}" font-family="-apple-system, sans-serif" font-size="28" font-weight="800">EVIDENCE BOUNDARY</text>
  <text x="934" y="390" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="32" font-weight="650">✓ all outputs finite</text>
  <text x="934" y="454" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="32" font-weight="650">✓ 60-second clean window before each run</text>
  <text x="934" y="518" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="32" font-weight="650">✓ one-second external-process monitor</text>
  <text x="934" y="582" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="32" font-weight="650">✓ zero monitored contenders in all intervals</text>
  <text x="934" y="646" fill="{YELLOW}" font-family="-apple-system, sans-serif" font-size="32" font-weight="650">△ power state not inspected or gated</text>
  <text x="934" y="710" fill="{RED}" font-family="-apple-system, sans-serif" font-size="32" font-weight="650">× full explicit reference remains infeasible</text>
  <rect x="934" y="766" width="806" height="66" rx="18" fill="#27171c"/>
  <text x="966" y="809" fill="{RED}" font-family="-apple-system, sans-serif" font-size="24" font-weight="750">98.589 s belongs only to the preceding safe route</text>'''
    return shell(
        "Row 14 — the exact promoted-route result",
        "Three untrimmed processes · matmul precision high · seed 9200 · contention fails closed",
        body.lstrip("\n"),
    )


def reproducibility_card(manifest: dict) -> str:
    source_count = len(manifest["solution_file_sha256"])
    body = f'''
  <rect x="92" y="230" width="520" height="650" rx="34" fill="{PANEL}" stroke="{CYAN}" stroke-width="3"/>
  <text x="142" y="305" fill="{CYAN}" font-family="-apple-system, sans-serif" font-size="27" font-weight="800">SOURCE IDENTITY</text>
  <text x="142" y="420" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="92" font-weight="900">{source_count}</text>
  <text x="260" y="410" fill="{MUTED}" font-family="-apple-system, sans-serif" font-size="29">production hashes</text>
  <text x="142" y="490" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="29" font-weight="650">MLX · exact commit · MIT</text>
  <text x="142" y="546" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="29" font-weight="650">triton-msl · exact commit · MIT</text>
  <text x="142" y="602" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="29" font-weight="650">clean export verified</text>
  <rect x="142" y="700" width="390" height="80" rx="20" fill="#16272b"/>
  <text x="178" y="750" fill="{GREEN}" font-family="-apple-system, sans-serif" font-size="25" font-weight="750">PROVENANCE: PASS</text>
  <rect x="646" y="230" width="560" height="650" rx="34" fill="{PANEL}" stroke="{GREEN}" stroke-width="3"/>
  <text x="696" y="305" fill="{GREEN}" font-family="-apple-system, sans-serif" font-size="27" font-weight="800">TWO PACKAGE MODES</text>
  <text x="696" y="425" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="78" font-weight="900">33 + 1</text>
  <text x="696" y="472" fill="{MUTED}" font-family="-apple-system, sans-serif" font-size="26">public-safe: pass + expected skip</text>
  <text x="696" y="610" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="78" font-weight="900">101 / 101</text>
  <text x="696" y="657" fill="{MUTED}" font-family="-apple-system, sans-serif" font-size="26">authorized input + empty native cache</text>
  <rect x="696" y="730" width="430" height="66" rx="18" fill="#16272b"/>
  <text x="730" y="773" fill="{GREEN}" font-family="-apple-system, sans-serif" font-size="24" font-weight="750">FRESH REBUILD: PASS</text>
  <rect x="1240" y="230" width="586" height="650" rx="34" fill="{PANEL}" stroke="{RED}" stroke-width="3"/>
  <text x="1290" y="305" fill="{RED}" font-family="-apple-system, sans-serif" font-size="27" font-weight="800">PUBLIC-SAFE BY DESIGN</text>
  <text x="1290" y="408" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="30" font-weight="650">✓ organizer attachment absent</text>
  <text x="1290" y="472" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="30" font-weight="650">✓ private history absent</text>
  <text x="1290" y="536" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="30" font-weight="650">✓ credentials and paths absent</text>
  <text x="1290" y="600" fill="{WHITE}" font-family="-apple-system, sans-serif" font-size="30" font-weight="650">✓ unsupported inputs fail closed</text>
  <text x="1290" y="676" fill="{MUTED}" font-family="-apple-system, sans-serif" font-size="26">No push · no publication · no submission</text>
  <rect x="1290" y="730" width="456" height="66" rx="18" fill="#27171c"/>
  <text x="1324" y="773" fill="{RED}" font-family="-apple-system, sans-serif" font-size="24" font-weight="750">EXTERNAL GATES STAY FALSE</text>'''
    return shell(
        "Reproducibility is part of the result",
        "Pinned sources · deterministic package · fresh install · explicit publication boundary",
        body.lstrip("\n"),
    )


def render(manifest: dict) -> dict[str, str]:
    if manifest["verification"]["official_mfu_or_combined_score"] is not None:
        raise AssertionError("refusing video assets with an invented official score")
    if manifest["row_14"]["current_route_exact_b32_status"] not in {
        "pending_contention_controlled_protocol",
        "complete_contention_controlled_protocol",
    }:
        raise AssertionError("unsupported exact-row status for video assets")
    return {
        "00_hook.svg": hook_card(manifest),
        "01_architecture.svg": architecture_card(manifest),
        "02_results.svg": results_card(manifest),
        "03_boundaries.svg": boundaries_card(manifest),
        "04_row14_evidence.svg": row_14_evidence_card(manifest),
        "05_reproducibility.svg": reproducibility_card(manifest),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    expected = render(manifest)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    for name, content in expected.items():
        path = OUTPUT_DIR / name
        if args.write:
            path.write_text(content)
        elif not path.is_file() or path.read_text() != content:
            failures.append(name)
        digest = hashlib.sha256(content.encode()).hexdigest()
        print(f"{digest}  docs/video_assets/{name}")
    if failures:
        raise AssertionError("video asset drift: " + ", ".join(failures))
    print(f"video assets: OK ({len(expected)} manifest-backed 1920x1080 SVG cards)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
