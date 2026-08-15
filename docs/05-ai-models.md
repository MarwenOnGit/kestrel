# 05 — AI Models

Three models run onboard. All are small. Training happens on the ground with a
big GPU; deployment happens on the Jetson after shrinking.

| Model | Job | Size on Jetson | Latency |
|---|---|---|---|
| **Classifier** | "What kind of signal is this?" | 2–6 MB | ~1 ms |
| **RF Weather Map** | "Is this normal for here and now?" | ~1 MB | ~0.3 ms |
| **Fingerprinter** | "Is this the *same radio* as last Tuesday?" | ~3 MB | ~1 ms |

Total well under 10 MB. **The AI is not what strains the hardware** — the
wideband DSP is (Doc 04). This surprises people, and it's why full autonomy on
limited hardware is achievable.

---

## 1. Model 1 — The signal classifier

### Input representation

Two options; use both in an ensemble if you can afford it, otherwise pick (a).

**(a) 1D CNN on raw IQ** — feed the normalized complex samples as a 2×N tensor
(I and Q as channels). Learns modulation structure directly. Best accuracy for
digital modes. This is the RadioML/O'Shea approach and is the recommended
default.

**(b) 2D CNN on spectrogram** — turn the snippet into a picture (time × frequency)
and use standard image networks. More robust to timing offsets, easier to debug
visually, slightly worse on fine modulation distinctions.

### Architecture (start here, it works)

```python
# kestrel/models/classifier_def.py
import torch, torch.nn as nn

class SignalCNN(nn.Module):
    """1D CNN over raw IQ. ~400k params — deliberately small."""
    def __init__(self, n_classes=10, n_samples=1024):
        super().__init__()
        def blk(i, o, k=7, s=2):
            return nn.Sequential(
                nn.Conv1d(i, o, k, stride=s, padding=k//2),
                nn.BatchNorm1d(o), nn.ReLU(inplace=True))
        self.body = nn.Sequential(
            blk(2, 32), blk(32, 64), blk(64, 64),
            blk(64, 128), blk(128, 128),
            nn.AdaptiveAvgPool1d(1))
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):          # x: (B, 2, N) float32
        return self.head(self.body(x))
```

### Classes

```
fm_voice, am_voice, dmr, tetra, p25, lte, nr5g,
lora, wifi, bluetooth, ocusync(drone), satcom_terminal, unknown
```

### Critical: the `unknown` class

Softmax always outputs *something*, so a network shown a signal type it has never
seen will confidently pick the nearest class. That's dangerous here — a
misclassification that says "dmr" could pull a signal inside the warrant scope.

Two defenses, use both:
1. **Explicit `unknown` class** trained on a wide grab-bag of out-of-set signals.
2. **Confidence gate:** if `max(softmax) < 0.85`, override to `unknown`.

And the architectural backstop: **`unknown` is never a warrant match** (Doc 08).
Uncertainty always reduces collection.

### Training data — the actual bottleneck

The model is easy; the data is the project. Four sources, in order of value:

1. **Your own recordings.** Fly/bench-record each class with known ground truth.
   Highest value, real hardware impairments. Budget serious time here.
2. **Public datasets.** RadioML 2016/2018 for modulation basics.
3. **GNU Radio synthesis.** Generate each mode in software with controlled
   SNR/offsets. Infinite volume, but "too clean" — always mix with real.
4. **Augmentation** (essential, see below).

**Augmentation is what makes it survive contact with reality:**
```python
def augment(iq, fs):
    iq = iq * np.exp(2j*np.pi*np.random.uniform(-500, 500)/fs*np.arange(len(iq)))  # freq offset
    iq = iq * np.exp(1j*np.random.uniform(0, 2*np.pi))          # random phase
    snr = np.random.uniform(-5, 30)                              # varied SNR
    p = np.mean(np.abs(iq)**2)
    n = np.sqrt(p/(2*10**(snr/10)))*(np.random.randn(len(iq))+1j*np.random.randn(len(iq)))
    iq = iq + n
    iq = iq / (np.std(iq) + 1e-12)                               # power normalize — ALWAYS
    return iq
```
Train across **−5 to +30 dB SNR**. A model trained only on strong signals fails
exactly where you need it: at the edge of range.

### Training loop

```bash
python3 -m kestrel.train.classifier \
    --data data/iq_library --epochs 60 --batch 256 \
    --snr-range -5 30 --out kestrel/models/classifier.pt
```
Hold out **by recording session**, not randomly — otherwise snippets from the
same recording land in both train and test and you'll report fake accuracy.

### Shrinking for the Jetson

```bash
# 1. Export to ONNX
python3 -m kestrel.train.export_onnx --ckpt classifier.pt --out classifier.onnx

# 2. Compile with TensorRT, INT8 quantized  (run ON the Jetson)
/usr/src/tensorrt/bin/trtexec \
    --onnx=classifier.onnx \
    --saveEngine=classifier.trt \
    --int8 --calib=calib_cache \
    --workspace=2048
```

| Technique | Effect | Accuracy cost |
|---|---|---|
| INT8 quantization | ~4× smaller, ~3× faster | <1% if calibrated |
| Pruning (30–50%) | smaller, faster | ~1% |
| Distillation | keeps big-model accuracy at small size | recovers most losses |
| TensorRT compile | 2–5× speedup on Jetson | none |

Target: **< 2 ms per inference**, so even a busy band (50 candidates/s) uses
under 10% of the GPU.

---

## 2. Model 2 — The RF Weather Map *(signature idea #3)*

**The problem it solves:** supervised classification only recognizes what you
labeled. Real operations turn up emitters nobody has ever labeled — a modified
radio, an improvised burst transmitter, a mode from a country you don't have
samples for. Those are often *precisely* the interesting ones.

**The idea:** learn what "normal" sounds like for each place and time, with **no
labels at all**, and flag deviations.

- Divide the operating area into tiles (e.g. 500 m geohash).
- Bucket by hour-of-day and day-type (weekday/weekend).
- For each (tile, hour) learn a compact model of the *normal* spectrum
  occupancy — an autoencoder over the binned PSD vector.
- **Novelty score = reconstruction error.** High error = "this is not what this
  place usually sounds like at this hour."

```python
class SpectrumAutoencoder(nn.Module):
    """Learns 'normal' spectrum occupancy. Unsupervised — no labels needed."""
    def __init__(self, n_bins=512, latent=16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_bins, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, latent))
        self.dec = nn.Sequential(
            nn.Linear(latent, 32), nn.ReLU(),
            nn.Linear(32, 128), nn.ReLU(),
            nn.Linear(128, n_bins))

    def forward(self, x):
        return self.dec(self.enc(x))

def novelty_score(model, psd_vector):
    with torch.no_grad():
        recon = model(psd_vector)
        return float(torch.mean((recon - psd_vector)**2))
```

**Why this is powerful for your mission:** a repeater in an empty valley, a
satellite terminal where no infrastructure exists, or a burst transmitter on a
smuggling route are all *statistically abnormal for that tile* even when the
classifier says `unknown`. The Weather Map turns "I don't recognize this" into
"and that's exactly why you should look."

**Bootstrapping:** fly baseline sorties over the area *before* operations to
learn normal. This baseline is itself a deliverable — a durable RF map of the
region that improves every sortie.

**Novelty never widens collection.** A high novelty score raises an alert and
directs the swarm to investigate (Doc 06); it does **not** authorize recording.
Only the warrant token does that.

---

## 3. Model 3 — Emitter fingerprinting

**The idea:** manufacturing tolerances make every physical radio slightly unique
— tiny frequency offsets, power-amplifier nonlinearity, transient shape at
key-up. These "RF fingerprints" persist across transmissions.

**What it buys you:** the ability to say *"the same handheld that transmitted
from the ridge on Tuesday is transmitting from the wadi now"* — across days and
sorties. This is what makes pattern-of-life real rather than a list of
frequencies.

Implemented as a **metric-learning embedding**: train with triplet loss so that
snippets from the same physical radio land close together in a 64-dimensional
space, and different radios land apart.

```python
class FingerprintNet(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.body = SignalCNN(n_classes=dim)   # reuse the backbone
    def forward(self, x):
        e = self.body(x)
        return e / (e.norm(dim=1, keepdim=True) + 1e-8)   # L2-normalized

# Training: triplet loss (anchor, positive=same radio, negative=different radio)
loss = nn.TripletMarginLoss(margin=0.3)(anchor, positive, negative)
```

At inference, compare the embedding against the pattern-of-life database
(Doc 08); a cosine similarity above threshold is a **candidate** re-identification
— always presented to a human analyst as a probability, never as a fact.

**Honest limitations, state them in any report:**
- Fingerprints drift with temperature, battery voltage, and age.
- Two radios from the same production batch can be very close.
- This identifies a **device**, not a **person**. Never let a report imply
  otherwise.

---

## 4. Running all three on the Jetson

```python
# kestrel/air/classifier.py  (sketch)
class EdgeInference:
    def __init__(self):
        self.clf  = TRTEngine("models/classifier.trt")
        self.wmap = TRTEngine("models/weathermap.trt")
        self.fp   = TRTEngine("models/fingerprint.trt")

    def process(self, snippet, psd_vec, tile, hour):
        logits = self.clf.run(snippet)
        p = softmax(logits)
        cls, conf = argmax_class(p)
        if conf < 0.85:
            cls = "unknown"                       # uncertainty → unknown
        nov = self.wmap.run(psd_vec, ctx=(tile, hour))
        emb = self.fp.run(snippet) if cls != "unknown" else None
        return cls, conf, nov, emb
```

Batch candidates when several arrive in the same frame — GPU efficiency is
mostly about batch size.

---

## 5. Continual learning (safely)

The system gets better every sortie, but **never retrains itself in flight.**
The loop is:

1. Aircraft logs low-confidence and high-novelty detections (metadata + snippet
   *only if authorized*).
2. After landing, an analyst reviews and labels a sample of them.
3. Ground GPU retrains; accuracy is measured on a frozen held-out set.
4. A new model is **signed** and deployed to aircraft only if it beats the
   incumbent on that frozen set.
5. Model version is recorded in the audit log with every detection it produced.

**Why models are signed and versioned:** if a detection ends up in court, you
must be able to state exactly which model version produced the classification and
show its measured accuracy. Unversioned models make evidence indefensible.

---

## 6. What NOT to use AI for

| Task | Use | Not |
|---|---|---|
| Flight stabilization | PX4 classical control | Learned controller |
| Warrant gate decision | Deterministic code | Any model |
| Timing / TDOA math | Closed-form + least squares | Learned regression |
| Legal scope | Signed token | Inference |

The warrant gate in particular must be **auditable line by line**. A neural
network deciding what may be recorded is unexplainable, and therefore
indefensible. Keep AI on the "what is it / where should I look" side of the
line, and keep deterministic code on the "what may I keep" side.

---

Next: [`06-autonomy-agent.md`](06-autonomy-agent.md).
