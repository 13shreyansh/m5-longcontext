# Official Track 3 statement — public-safe reconciliation

This note records the official challenge contract used by the retained result
without redistributing or linking to the organizer's unlicensed attachments.

## Challenge

Implement the fastest acceptable causal Transformer layer for the
participant's chosen machine. The organizer's PyTorch comparison accepts an
output element when absolute error is at most `0.002` **or** relative error is
at most `0.02`; output shape must match. The webinar fixed float32 as the
baseline, input and returned-output precision, allowed internal quantization,
allowed compilation and the first execution to be excluded from timing, and
said the result may target one participant-owned device.

The published appendix contains 14 rows. Rows 1–13 use modest sequence lengths
and can be compared directly with the explicit reference. Row 14 is
`(B=32,S=100000,D=1024,H=16,F=1024,L=2,causal=True)`. Its explicit float32
attention score tensor alone would require 18.626 TiB, so that baseline is not
physically runnable on the declared 64-GB Mac.

## Source update used

The statement displayed a 2026-08-27 18:25 SGT update adding the 14-row test
shape appendix and changing the PyTorch command defaults from `atol=0.001,
rtol=0.01` to `atol=0.002, rtol=0.02`. The current attachment used in the
private evidence repository had 25,017 bytes and SHA-256
`5529c96a80799b51f68092e1444a30b17994554dffdf52da98ba701489a7f36e`.
It is deliberately absent from the public-safe package.

Canonical statement:
<https://bytedance.larkoffice.com/wiki/DNtSwxgeciCS2nkiUefc5qqtnkf>.

## Remaining unknowns

The exact numerical MFU definition, row weights, aggregation, bandwidth
adjustment, padding distribution, skip/OOM treatment and official command
sequence were not published in a form sufficient to calculate a combined
organizer score. This repository therefore reports synchronized latency,
correctness counts and explicit comparison boundaries rather than inventing an
official MFU or combined score.

The organizer attachments displayed no SPDX identifier, copyright header or
adjacent redistribution grant. They remain private evaluator inputs. The
public package includes only the entrant solution and third-party material
whose MIT notices are retained.

