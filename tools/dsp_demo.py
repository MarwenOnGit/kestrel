"""Show the DSP funnel working on synthesized IQ, end to end.

    python3 tools/dsp_demo.py

Prints an ASCII spectrum, what the detector found, what the classifier made of
it, and what the warrant gate decided - the whole Doc 04 §1 funnel, with no
hardware and no ground truth given to the pipeline.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kestrel.air.detector import noise_floor, welch_psd
from kestrel.air.pipeline import EdgePipeline, FeatureClassifier, SweepSegment
from kestrel.air.sdr_source import SimulatedSdrSource
from kestrel.air.warrant_gate import WarrantGate
from kestrel.common import crypto
from kestrel.common.warrant import WarrantToken
from kestrel.ground import signer
from kestrel.ground.audit import AuditLog
from kestrel.sim.world import Aircraft, Emitter, KestrelSim

POLY = [[0, 0], [9000, 0], [9000, 9000], [0, 9000]]


def ascii_spectrum(psd, nf, center_hz, rate_sps, width=76, height=14):
    n = len(psd)
    cols = np.array_split(psd, width)
    peaks = np.array([c.max() for c in cols])
    lo, hi = nf - 3, max(peaks.max(), nf + 10)
    rows = []
    for r in range(height):
        level = hi - (hi - lo) * r / (height - 1)
        rows.append("".join("#" if p >= level else " " for p in peaks))
    out = [f"  {hi:6.1f} dB |{rows[0]}"]
    out += [f"           |{r}" for r in rows[1:-1]]
    out.append(f"  {lo:6.1f} dB |{rows[-1]}")
    out.append("           +" + "-" * width)
    out.append(f"      {(center_hz - rate_sps/2)/1e6:.2f} MHz"
               + " " * (width - 26) + f"{(center_hz + rate_sps/2)/1e6:.2f} MHz")
    return "\n".join(out)


def main():
    emitters = [
        Emitter("TGT-dmr", (2200, 3100), 462.30e6, "dmr", eirp_dbm=37,
                burst_s=100, period_s=100),
        Emitter("TGT-fm", (2600, 3400), 462.55e6, "fm_voice", eirp_dbm=37,
                burst_s=100, period_s=100),
        # Outside the tuned window. A 10 MHz LTE carrier sitting inside a
        # 2 MHz window does not appear as a signal - it BECOMES the noise
        # floor, and swamps the narrowband emitters we are after. That is a
        # real effect (Doc 12 §5.2), reproduced by the band-filling test in
        # tests/test_dsp.py; here we tune away from it, which is what the
        # sweep planner would do in the field.
        Emitter("DECOY-lte", (2400, 3200), 470.0e6, "lte", eirp_dbm=30,
                burst_s=100, period_s=100),
        Emitter("DECOY-lora", (2100, 3000), 462.95e6, "lora", eirp_dbm=20,
                burst_s=100, period_s=100),
    ]
    sim = KestrelSim(emitters, [Aircraft("K1", xy=(2000.0, 3000.0))],
                     noise_floor_dbm=-110.0, seed=0)

    src = SimulatedSdrSource(sim, "K1", seed=0)
    center, rate = 462.9e6, 2e6
    src.tune(center, rate)
    iq, t_ns = src.frame(262_144)

    print("\n" + "=" * 88)
    print("  KESTREL - edge DSP pipeline on synthesized IQ")
    print("=" * 88)
    print(f"\n  Tuned: {center/1e6:.2f} MHz, {rate/1e6:.0f} MHz span, "
          f"{len(iq):,} complex samples "
          f"({iq.nbytes/1e6:.1f} MB, {iq.nbytes*rate/len(iq)/1e6:.0f} MB/s sustained)")

    psd = welch_psd(iq)
    nf = noise_floor(psd)
    print(f"  Noise floor (25th pct): {nf:.1f} dB\n")
    print(ascii_spectrum(psd, nf, center, rate))

    # ---- warrant: DMR and FM voice only, 462.0-463.0 MHz -------------
    auth_sk, auth_pk = crypto.generate_authority_keypair()
    _, seal_pk = crypto.generate_seal_keypair()
    token = WarrantToken.load(
        signer.quick_warrant(seal_pk, auth_sk, area_polygon=POLY,
                             freq_ranges_hz=[[462.0e6, 463.0e6]],
                             signal_classes=["dmr", "fm_voice"]), auth_pk)
    audit = AuditLog(os.path.join(tempfile.mkdtemp(), "a.jsonl"), durable=False)
    gate = WarrantGate(token, audit)

    pipe = EdgePipeline(FeatureClassifier())
    results = pipe.process_frame(iq, SweepSegment(center, rate), t_ns, (2000.0, 3000.0))

    print(f"\n  Warrant {token.warrant_id}: 462.0-463.0 MHz, classes "
          f"{token.body['signal_classes']}\n")
    print(f"  {'frequency':>12} {'bw':>9} {'snr':>7}  {'class':<16} {'conf':>5}  verdict")
    print("  " + "-" * 84)

    for det, snip in results:
        det.signal_class, det.class_confidence = pipe.classifier.classify(det, snip)
        allowed, reasons = gate.decide(det, (2000.0, 3000.0))
        verdict = "RECORD" if allowed else "count only: " + ", ".join(reasons[:2])
        print(f"  {det.freq_hz/1e6:>9.4f} MHz {det.bandwidth_hz/1e3:>7.1f}k "
              f"{det.snr_db:>6.1f}  {det.signal_class:<16} "
              f"{det.class_confidence:>5.2f}  {verdict}")

    truth = {e.freq_hz: e.emitter_id for e in emitters}
    print("\n  Ground truth (never shown to the pipeline):")
    for f, name in sorted(truth.items()):
        print(f"    {f/1e6:9.4f} MHz  {name}")

    h = pipe.health()
    print(f"\n  Pipeline health: {h['candidates_seen']} candidates in "
          f"{h['frames']} frame(s), {h['dropped_over_budget']} dropped over budget")
    ok, _ = audit.verify()
    print(f"  Audit chain: {len([e for e in audit.entries()])} entries, "
          f"verifies={ok}")
    print()


if __name__ == "__main__":
    main()
