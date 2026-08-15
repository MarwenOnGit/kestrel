# 00 — RF from Zero

**Assumes you know nothing about radio.** Read once, then the rest of the
documentation is readable. ~20 minutes.

---

## 1. What a radio signal actually is

Electricity wiggling in a wire makes an invisible wave that travels at the speed
of light. Wiggle it faster and you get a different **frequency**. Frequency is
measured in **hertz (Hz)** = wiggles per second.

- 1 kHz = one thousand per second
- 1 MHz = one million
- 1 GHz = one billion

Everything that transmits picks a frequency and wiggles there:

| Thing | Frequency |
|---|---|
| FM radio station | 88–108 MHz |
| Handheld walkie-talkie / marine VHF | 30–520 MHz |
| Mobile phone | 700 MHz – 3.8 GHz |
| WiFi | 2.4 GHz and 5.8 GHz |
| Satellite phone (Thuraya/Iridium) | 1.5–1.7 GHz |
| Consumer drone control link | 2.4 / 5.8 GHz |
| LoRa tracker hidden in cargo | 433 / 868 / 915 MHz |

**Spectrum** is the whole range of frequencies laid out side by side, like keys
on a piano. Our job is to watch a lot of piano at once and notice which keys are
being pressed, by whom, and from where.

### Why frequency choice matters to us
Low frequencies (VHF, 30–300 MHz) travel far, bend around hills, and go through
foliage — which is exactly why people operating in remote terrain use handheld
VHF/UHF radios. But low frequencies need **big antennas**. High frequencies
(GHz) need small antennas but travel line-of-sight only. This single trade-off
drives most of our hardware decisions.

> **Why fly at all?** Radio is mostly line-of-sight. On the ground, a hill
> between you and a transmitter hides it completely. At 120 m altitude you see
> over the hills — a drone at altitude can hear emitters across an entire
> valley that a ground team would need days to cover. Altitude is the single
> biggest sensitivity upgrade available, worth more than any amplifier.

---

## 2. What a software-defined radio (SDR) is

An old radio has fixed circuits: an FM radio can only ever do FM. A **software-
defined radio** does the tuning and decoding in *software*, so the same box can
listen to a walkie-talkie, a phone band, or a satellite by changing code.

An SDR gives you two knobs and one output:
- **Center frequency** — where in the spectrum to look.
- **Sample rate / bandwidth** — how *wide* a slice to look at, at once.
- **Output: IQ data.**

### IQ data — the one term you must not skip
A radio wave has two properties at every instant: how strong it is
(**amplitude**) and where it is in its wiggle cycle (**phase**). The neat way to
record both is as two numbers, called **I** and **Q**. A stream of IQ pairs is a
complete recording of everything the antenna heard in that slice of spectrum.

**Everything in this project is math performed on IQ data.** Recording IQ =
recording the signal itself. This is why IQ files are both extremely valuable
(you can re-analyze them forever) and legally sensitive (they contain content).

An IQ stream is big: sample rate × 2 numbers × bytes per number. At 50 MHz
bandwidth with 16-bit samples that's ~200 MB **per second**. This is why we
cannot stream everything to the ground, and why the drone must decide onboard
what matters. That constraint is the origin of the whole architecture.

---

## 3. Seeing the spectrum: FFT, noise floor, waterfall

Raw IQ looks like meaningless jitter. The **FFT (Fast Fourier Transform)** is a
mathematical operation that converts it into a chart: *how much energy is present
at each frequency*. That chart is the **spectrum**.

```
 power
   ^                    ▲ a signal (someone transmitting)
   |                    █
   |         ▲          █
   |         █          █           ▲
   |~~~~~~~~~█~~~~~~~~~~█~~~~~~~~~~~█~~~~~~~~  ← noise floor (always there)
   +----------------------------------------> frequency
```

- **Noise floor** — the permanent background hiss of the universe and your own
  electronics. Nothing is ever perfectly quiet.
- **A signal** is anything poking above the noise floor. Finding those pokes is
  called **detection**, and it is step one of everything we do.
- **SNR (signal-to-noise ratio)** — how far above the floor the poke rises, in
  **dB**. Higher SNR = easier to classify and locate. +10 dB means ten times the
  power; +3 dB means double.
- A **waterfall** is the spectrum drawn repeatedly over time, scrolling — so you
  can see bursts appear and disappear. This is the classic SIGINT display.

**Detection is easy. Identification and location are the hard parts.** An FFT
tells you *something* is at 462.5 MHz. It does not tell you it's a walkie-talkie,
and it certainly doesn't tell you it's in the wadi 3 km northeast. That's what
Docs 05 and 07 are for.

---

## 4. Modulation — how information rides on the wave

A bare wave carries no information. You must vary ("modulate") it:

- **AM** — vary the *strength*. Old, simple, used in aviation.
- **FM** — vary the *frequency* slightly. Analog walkie-talkies, FM broadcast.
- **Digital modes** (PSK, QAM, FSK, OFDM) — encode 1s and 0s as changes in phase
  and/or amplitude. Everything modern: DMR, TETRA, LTE, 5G, WiFi, LoRa.

Each modulation leaves a **distinctive shape** in the spectrum and in the IQ
statistics — a fingerprint of *kind*. That is precisely what an AI classifier
learns to recognize (Doc 05). A human expert can often identify a mode by eye
from a waterfall; we are teaching a small neural network to do it in
milliseconds, thousands of times per minute.

**Important distinction:**
- Recognizing the **type** of signal ("this is DMR digital voice") = reading its
  outward shape. Always possible. Not sensitive.
- Recovering the **content** ("what was said") = demodulation, and if encrypted,
  impossible without keys. **We never attempt to break encryption.** Under a
  warrant we may demodulate *unencrypted* authorized traffic; that's it.

---

## 5. Direction finding and geolocation — the "where"

Three ideas, in increasing power:

**(a) Signal strength.** Fly around; the direction where it's loudest is roughly
where the transmitter is. Crude, easily fooled by terrain and antenna
orientation. We use it only as a sanity check.

**(b) Bearing / AOA (angle of arrival).** With **two or more antenna elements** a
known distance apart, the wave reaches one slightly before the other. From that
tiny difference you compute the **compass direction** to the transmitter.
Multiple antennas in a known geometry = an **array**. Two aircraft each taking a
bearing → the lines **cross** at the source. This is triangulation.

**(c) TDOA (time difference of arrival).** Three or more aircraft, spread out,
all hear the same burst. Because they share precise time, the differences in
*when* each heard it define curves that intersect at the source.

> **Why clocks dominate accuracy.** Radio travels about **30 cm per
> nanosecond**. If two aircraft's clocks disagree by 100 ns, your fix is off by
> ~30 m. This is why each aircraft carries a **GPSDO** (GPS-disciplined
> oscillator) — an ordinary oscillator continuously corrected against satellite
> time — instead of a normal crystal. Timing *is* accuracy.

**Error ellipse.** No fix is a perfect point. Honest geolocation always outputs a
*region*: "95% confident the emitter is inside this ellipse." Spreading the
aircraft further apart and observing from varied angles shrinks it. Doc 06's
information-gain flight is literally an algorithm for shrinking this ellipse as
fast as possible.

---

## 6. The RF chain — why there's stuff between antenna and radio

`Antenna → filter → low-noise amplifier (LNA) → SDR`

- **Antenna** — converts radio waves into electrical signals. Its size is tied to
  the wavelength it must catch, which is why VHF antennas are big.
- **Filter** — blocks frequencies you don't want. Without it, one nearby FM
  broadcast tower can swamp the receiver and deafen it to everything else
  (**desensitization**). Filters are not optional in the real world.
- **LNA (low-noise amplifier)** — boosts the tiny signal *right at the antenna*,
  before cable losses degrade it. Its **noise figure** (lower is better; want
  <1.5 dB) sets the floor on how weak a signal you can ever hear.
- Then the **SDR** digitizes it into IQ.

**Coherent channels.** If we want time/phase comparisons (for DF and TDOA), all
receiver channels must share a single clock. Radios that do this are
**coherent**. Cheap USB dongles each keep their own time and therefore *cannot*
do real direction finding. This is the single reason we buy USRP-class
hardware.

> **The metronome analogy.** Four microphones all running off one metronome let
> you tell that a sound reached mic #1 a hair before mic #2 — and from that,
> which direction it came from. Four microphones each with their own watch tell
> you nothing of the sort.

---

## 7. Units cheat sheet

| Symbol | Means | Feel for it |
|---|---|---|
| Hz / kHz / MHz / GHz | frequency | walkie-talkie ≈ 150 MHz, phone ≈ 1.8 GHz |
| dB | ratio, logarithmic | +3 dB = 2×, +10 dB = 10×, +20 dB = 100× |
| dBm | absolute power | 0 dBm = 1 mW; a handheld radio ≈ +37 dBm (5 W); we detect around −110 dBm |
| SNR (dB) | signal above noise | <5 dB hard, 10–20 dB comfortable, >30 dB loud |
| Sample rate (Msps) | IQ samples/sec | roughly equals bandwidth you can see |
| Noise figure (dB) | how much an amp degrades SNR | lower is better; <1.5 dB is good |

---

## 8. The mental model to carry forward

> The drone is a **flying ear with a clock and a conscience.**
>
> - The **ear** (antenna + LNA + SDR) turns invisible waves into IQ numbers.
> - The **FFT** turns numbers into a picture of who's talking.
> - The **AI** turns the picture into "what kind of thing is that."
> - The **clock** — shared across the swarm — turns "what" into "where."
> - The **conscience** (warrant gate) decides what may be kept, and seals it so
>   even the drone can't read it back.

Next: [`01-architecture.md`](01-architecture.md).
