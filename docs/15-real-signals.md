# 15 — First Contact with Real Signals

**As of 2026-11-02.** Every number before this document came from a simulator
whose error model I wrote myself (Doc 13 §6). This is the first time the DSP chain met a
waveform nobody synthesized. The short version: **detection held up;
classification did not, and failed in the direction that matters for a
warrant.**

    python3 tools/real_signals.py tests/data                      # 2 fixtures, seconds
    python3 tools/real_signals.py <satellite-recordings checkout> # all 92
    python3 -m pytest tests/test_real_signals.py -q

---

## 1. The data

[`daniestevez/satellite-recordings`](https://github.com/daniestevez/satellite-recordings):
92 short recordings of real amateur-satellite downlinks, public domain,
collected to test decoders. Two of them are vendored as test fixtures
(`tests/data/`, attribution in its README).

Why amateur satellites: they are the only real RF that is legally public,
plentiful, documented, and comes with a built-in correctness check. AX.25
frames carry a 16-bit FCS, so a decoded frame is either right or absent.

## 2. What is real and what is not

The recordings are **FM-discriminator audio** at 48 kHz: the ground station
had already demodulated the RF. KESTREL's pipeline wants complex IQ, so
`kestrel/lab/realsignal.py` puts the audio back on a carrier.

| Real, from the pass | Not real, added here |
|---|---|
| bit timing and clock error of the satellite's radio | the carrier itself, placed 250 kHz off centre |
| symbol shaping and transmit filtering (GFSK, AFSK) | white noise at a chosen SNR (30 and 12 dB over 20 kHz) |
| the receiver's IF/discriminator response | FM deviation, **assumed ±3.5 kHz**, because audio level does not record it |
| the noise that was actually on the pass | no Doppler drift, no fading, no neighbours |

Results below say "re-modulated", never "RF". The deviation assumption
matters for anything that reads frequency excursion. Section 3 says which
results depend on it.

`nusat.wav` is the one stereo 192 kHz file in the set, almost certainly I/Q.
The loader takes its left channel as audio, which is wrong for that file. It
stays in the table because it changes nothing below, and dropping it quietly
would be worse.

## 3. Results, all 92 recordings

### Protocol layer: `kestrel/lab/ax25.py`

AFSK 1200 (Bell 202) and 9600 bit/s FSK with G3RUH scrambling, then NRZI,
HDLC deframing with bit destuffing, FCS, and the AX.25 header.

- **10 recordings decoded, 17 FCS-valid frames**, from Aalto-1, OPS-SAT
  (`DP0OPS>DL0ESA`), Irazú, Światowid (`SR6SAT-6`, AFSK, ASCII telemetry),
  TigriSat, KOYO, SE01, AZ02, US01 and US04.
- The other 82 use other physical layers (BPSK, Mobitex, CCSDS and
  vendor-specific framing). That is not a decoder failure. This module only
  covers the two layers most common on amateur satellites.
- **Three satellites send CRC-valid HDLC whose headers are not AX.25.** SE01
  puts plain ASCII callsigns (`ON01SE`) where AX.25 requires them shifted
  left one bit. TigriSat pads one frame's callsign with `0x44` instead of
  `0x40`. KOYO uses a hyphen inside a callsign field. The first version of
  the parser printed these as garbage callsigns. It now refuses them
  (`non-conforming address`), and a test pins that behaviour down. The
  lesson: a valid checksum proves the link layer is intact, not that the
  protocol you assumed is the one in use.

### Detection: the part that works

| | 30 dB | 12 dB |
|---|---|---|
| Exactly one detection | 90 / 92 | 92 / 92 |
| Median frequency error | 65 Hz (both SNRs pooled) | |
| p95 frequency error | 394 Hz | |

CFAR, the global threshold, the edge guard and the 99%-power bandwidth from
Doc 13 §5a all held up on waveforms the synthesizer never produced. The two
double detections at 30 dB (KOYO, Zhou Enlai) split one strong burst into two
spans. This does not depend on the deviation assumption: detection is a power
measurement.

### Classification: the part that does not

| | 30 dB | 12 dB |
|---|---|---|
| Labelled `dmr` (0.87) | 87 | 90 |
| Labelled `fm_voice` (0.86) | 5 | 0 |
| `unknown` | 0 | 2 |
| **Above the gate's 0.85 threshold** | **92 / 92** | **90 / 92** |

None of these satellites is DMR. All of them would be authorized for capture
under a warrant covering DMR or FM voice in their band, which in Europe means
most of 430–440 MHz.

**Why.** `FeatureClassifier` is closed-set. Its rule reads: constant envelope,
narrower than 60 kHz, frequency variance above 10⁶ Hz², therefore DMR at 0.87.
That rule describes nearly every narrowband digital FM signal in existence. In
the simulator it looked correct because the simulator only ever produces
twelve kinds of signal, and DMR is the only one in its library that fits.
Against the real world, "the only one I know that fits" became "certainly
this one".

**Two fixes tried and rejected.** Both measured on the same scenes, with a
pulse-shaped DMR (4FSK, root-raised-cosine, β = 0.2) added as a control:

| Feature | Simulator's DMR | Realistic DMR | Real satellites |
|---|---|---|---|
| Symbol-rate line in \|d f/dt\| (peak / median) | 540 at 30 dB, 15 at 12 dB | 20–27 at 30 dB, 15–19 at 12 dB | 12–20 |
| Deviation levels in the instantaneous-frequency histogram (30 / 20 / 12 dB) | 4 / 4 / 4 | 5 / 3 / 1 | 2–7, scattered |

(Realistic DMR, 5 trials per SNR. The symbol-rate peak does land on 4.8 kHz in
4 of 5 trials at 30 dB, but in none at 12 dB.)

Both features separate the simulator's DMR from the satellites very cleanly,
and both fail on realistic DMR. The synthesizer's DMR uses rectangular
symbols, which no real handset transmits, and that is exactly what both
features were detecting. Adopting either one would have fitted the classifier
to a flaw in my own simulator, which is the trap Doc 13 §3 bug 4 describes,
in a new place.

**What was done instead.**

1. **Recorded as a strict xfail.**
   `test_classifier_does_not_claim_a_protocol_it_cannot_identify` fails the
   build if it ever starts passing by accident, the same as the
   multi-emitter limitation in Doc 13 §4.1.
2. **Mitigated in the legal layer, where it can be done honestly today.**
   Warrants already support `minimization_rules.exclude_freq_ranges_hz`. A
   voice warrant over 430–440 MHz that excludes 435–438 MHz refuses the
   misclassified downlink with `explicitly_excluded`, and that is tested.
   Excluding known non-target allocations should be the default practice in
   warrant drafting until the classifier can be trusted, not an option.
3. **Turned into Step 4 requirements.** The trained classifier needs
   *open-set rejection*: an explicit "not in my library" outcome, trained with
   negatives. These 92 recordings, re-modulated across deviations and SNRs,
   are that negative set. Its acceptance test is already written (the strict
   xfail above). The synthesizer's DMR and P25 also need pulse shaping before
   any result trained on them means anything.

## 4. Updates to earlier claims

- Doc 13 §5a said the demo scene was classified correctly. That is still
  true, and it is not evidence the classifier works. It is evidence that the
  classifier recognizes the synthesizer.
- The legal invariants (`over_collection_count = 0`) in Doc 13 §2 hold in the
  simulator because every emitter there is one the warrant anticipated. On
  real spectrum, the gate is only as good as the label it is handed. The gate
  enforced the warrant correctly here; the classifier gave it the wrong input.
- Detection numbers carry over. Classification numbers do not.

## 5. What would make this stronger

- Real IQ instead of re-modulated audio: an RTL-SDR on a 435 MHz pass, or
  SatNOGS observations that publish raw IQ. That brings Doppler drift, which
  the CFAR and channelizer have never seen.
- Real DMR off air, as the positive class. Without it, every claim about
  DMR rests on a synthesizer now known to be unrealistic.
- Sweep the deviation assumption (±2.5 to ±5 kHz) to show the classification
  result does not depend on it. The `freq_var > 10⁶` boundary suggests it
  will not, since every deviation in that range clears it many times over,
  but that is an argument, not a measurement.
