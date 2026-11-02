"""Run KESTREL's DSP chain and the AX.25 decoder over real satellite recordings.

    python3 tools/real_signals.py tests/data
    python3 tools/real_signals.py path/to/satellite-recordings   # the full set

For each 48 kHz FM-audio recording:
  1. decode AFSK 1200 and G3RUH 9600 AX.25; every frame shown passed its FCS;
  2. re-modulate the audio onto a carrier (Doc 15 §2), place it 250 kHz off
     centre in a 1.92 MS/s window over white noise at 30 and 12 dB, and run
     the airborne pipeline exactly as in flight: detect, channelize, classify.

Prints one row per recording and a summary. Doc 15 is written from its output.
"""
from __future__ import annotations

import argparse
import collections
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kestrel.air.classifier import MIN_CONFIDENCE
from kestrel.air.pipeline import EdgePipeline, FeatureClassifier, SweepSegment
from kestrel.lab.ax25 import decode_afsk1200, decode_fsk9600_g3ruh, parse_ax25
from kestrel.lab.realsignal import (busiest_window, embed_in_scene,
                                    fm_remodulate, load_wav)

CENTER, OFFSET, FACTOR = 436.0e6, 250e3, 40
SNRS = (30, 12)


def decode(audio, fs):
    rows = []
    for layer, fn in (("afsk1200", decode_afsk1200), ("g3ruh9600", decode_fsk9600_g3ruh)):
        for raw in fn(audio, fs):
            try:
                f = parse_ax25(raw)
                rows.append((layer, f"{f.source}>{f.destination}", True))
            except ValueError:
                rows.append((layer, "non-AX.25 HDLC", False))
    return rows


def dsp(audio, fs, deviation_hz, rng):
    bb = fm_remodulate(busiest_window(audio, fs, 0.15), fs, deviation_hz)
    out = {}
    for snr in SNRS:
        iq, rate = embed_in_scene(bb, fs, FACTOR, OFFSET, snr, rng)
        pipe = EdgePipeline(FeatureClassifier())
        res = pipe.process_frame(iq[:262_144], SweepSegment(CENTER, rate), 0, (0.0, 0.0))
        dets = []
        for det, snip in res:
            cls, conf = pipe.classifier.classify(det, snip)
            dets.append((det.freq_hz - CENTER - OFFSET, det.bandwidth_hz, cls, conf))
        out[snr] = dets
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory")
    ap.add_argument("--deviation-hz", type=float, default=3500.0)
    ap.add_argument("--max-seconds", type=float, default=30.0)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    paths = sorted(glob.glob(os.path.join(args.directory, "*.wav")))
    stats = collections.Counter()
    labels = collections.Counter()
    freq_err = []

    print(f"{'recording':<24} {'decoded':<34} " + "  ".join(
        f"{s:>2} dB: n  dF(Hz)  class      conf" for s in SNRS))
    for p in paths:
        try:
            audio, fs = load_wav(p, max_s=args.max_seconds)
        except ValueError:
            continue
        name = os.path.basename(p)
        frames = decode(audio, fs)
        stats["recordings"] += 1
        if frames:
            stats["decoded"] += 1
            stats["frames"] += len(frames)
            stats["non_ax25"] += any(not ok for *_, ok in frames)
        dec = (f"{len(frames)}x {frames[0][0]} {frames[0][1]}" if frames else "-")[:34]

        cols = []
        for snr, dets in dsp(audio, fs, args.deviation_hz, rng).items():
            stats[f"single_det_{snr}"] += len(dets) == 1
            if dets:
                df, bw, cls, conf = dets[0]
                freq_err.append(abs(df))
                labels[(snr, cls)] += 1
                stats[f"authorizable_{snr}"] += cls != "unknown" and conf >= MIN_CONFIDENCE
                cols.append(f"{len(dets):>2} {df:>+7.0f}  {cls:<10} {conf:.2f}")
            else:
                cols.append(f"{0:>2} {'':>7}  {'(missed)':<10}     ")
        print(f"{name:<24} {dec:<34} " + "  ".join(f"    {c}" for c in cols))

    n = stats["recordings"]
    print(f"\n{n} recordings. AX.25 decoded from {stats['decoded']} "
          f"({stats['frames']} FCS-valid frames; {stats['non_ax25']} carried "
          f"non-AX.25 payloads inside valid HDLC).")
    for snr in SNRS:
        cls = ", ".join(f"{c} {k}" for (s, c), k in sorted(labels.items()) if s == snr)
        print(f"  {snr} dB: exactly one detection in {stats[f'single_det_{snr}']}/{n}; "
              f"labelled {cls}; authorizable (>= {MIN_CONFIDENCE}) "
              f"{stats[f'authorizable_{snr}']}/{n}")
    if freq_err:
        print(f"  frequency error: median {np.median(freq_err):.0f} Hz, "
              f"p95 {np.percentile(freq_err, 95):.0f} Hz")


if __name__ == "__main__":
    main()
