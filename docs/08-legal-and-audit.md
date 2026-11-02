# 08 — Legal Scope, Sealed Collection, and Audit

The subsystem that makes everything else usable. A collection platform that
cannot prove what it collected — and cannot prove what it *didn't* — produces
inadmissible evidence and personal liability for its operators.

**This is engineered, not promised.**

---

## 1. The threat model (be honest about it)

The risks this subsystem defends against:

| Risk | Consequence | Defense |
|---|---|---|
| Over-collection beyond the warrant | Evidence thrown out; unlawful surveillance | Warrant gate (§3), fail-closed |
| Scope altered in flight | Undetectable over-collection | Signed token (§2) |
| Aircraft captured / lost | Collected material exposed | Sealed collection (§4) |
| Insider exfiltration | Material leaves without record | Two-person export (§6) |
| Log tampering after the fact | Cannot prove compliance | Hash chain (§5) |
| "The AI decided to record it" | Unexplainable, indefensible | Deterministic gate, no AI (§3) |
| Mission creep across sorties | Gradual scope expansion | Per-sortie token, expiry, geofence |

---

## 2. The Warrant Token *(signature idea #4, part 1)*

The legal authorization becomes a **cryptographic capability token**. The
aircraft's collection ability is *defined* by it.

```json
{
  "warrant_id": "W-2026-0142",
  "issuing_authority": "…",
  "operator_id": "…",
  "co_signer_id": "…",
  "freq_ranges_hz": [[462000000, 468000000], [1610000000, 1660000000]],
  "signal_classes": ["dmr", "fm_voice", "satcom_terminal"],
  "area_polygon": [[31.10,34.70],[31.30,34.70],[31.30,34.90],[31.10,34.90]],
  "valid_from": "2026-08-14T04:00:00Z",
  "valid_until": "2026-08-14T16:00:00Z",
  "max_recording_seconds": 3600,
  "max_recording_per_emitter_seconds": 300,
  "minimization_rules": {"stop_on_class": ["unknown"], "exclude_freq_ranges_hz": [[121500000,121600000]]},
  "seal_pubkey": "base64…",
  "nonce": "base64…",
  "signature": "base64…"
}
```

Four properties that matter:

1. **Signed** (Ed25519, key in an HSM). The aircraft verifies before every
   recording decision, not just at boot. Tampering is detected immediately.
2. **Time-boxed.** After `valid_until` the token is refused and the in-memory
   copy is wiped. No indefinite authority.
3. **Geofenced.** Outside `area_polygon`, no content capture — even for an
   otherwise-matching signal.
4. **Two-person issued.** `operator_id` and `co_signer_id` must be distinct and
   both signatures present.

```python
# kestrel/air/warrant_gate.py
import nacl.signing, nacl.exceptions, json, time
from shapely.geometry import Point, Polygon

class WarrantToken:
    def __init__(self, raw, authority_pubkey):
        self.doc = self._verify(raw, authority_pubkey)
        self.polygon = Polygon(self.doc["area_polygon"])
        self.recorded_seconds = 0.0
        self.per_emitter = {}

    def _verify(self, raw, pub):
        vk = nacl.signing.VerifyKey(pub)
        body, sig = raw["body"], raw["signature"]
        vk.verify(json.dumps(body, sort_keys=True).encode(), sig)  # raises on tamper
        if body["operator_id"] == body["co_signer_id"]:
            raise ValueError("two-person rule violated")
        return body

    def valid_now(self, now=None):
        now = now or time.time()
        return (self._t(self.doc["valid_from"]) <= now
                <= self._t(self.doc["valid_until"]))
```

**Emergency revocation:** a signed `REVOKE` message on the mesh immediately
invalidates the token in flight; the aircraft stops recording and returns.
Revocation is fail-safe — if it can't be verified, the aircraft still stops.

---

## 3. The Warrant Gate

The single decision point for whether content may be captured. **Deterministic.
No AI. Auditable line by line.**

```python
# kestrel/air/warrant_gate.py
class WarrantGate:
    """The ONLY place in the codebase that may authorize content recording."""

    def __init__(self, token, audit):
        self.token, self.audit = token, audit

    def decide(self, det, aircraft_pos, now):
        reasons = []

        if not self.token.valid_now(now):            reasons.append("token_expired")
        if not self._in_area(aircraft_pos):          reasons.append("outside_geofence")
        if det.signal_class == "unknown":            reasons.append("class_unknown")
        if det.class_confidence < 0.85:              reasons.append("low_confidence")
        if det.signal_class not in self.token.doc["signal_classes"]:
                                                     reasons.append("class_not_authorized")
        if not self._freq_authorized(det.freq_hz):   reasons.append("freq_not_authorized")
        if self._excluded(det.freq_hz):              reasons.append("explicitly_excluded")
        if self.token.recorded_seconds >= self.token.doc["max_recording_seconds"]:
                                                     reasons.append("budget_exhausted")
        if self._emitter_budget_exhausted(det):      reasons.append("emitter_budget_exhausted")

        allowed = (len(reasons) == 0)
        self.audit.record_decision(det, allowed, reasons)   # ALWAYS logged, both ways
        return allowed, reasons
```

**Design rules, enforced by code review and tests:**
- Every rejection reason is logged. You can prove *why* something was not
  collected, which matters as much as proving what was.
- The gate is called **before** any write path. There is no code path from IQ to
  disk that bypasses it — enforced by a test (§7).
- Default is **deny**. New signal classes are not collectable until explicitly
  added to a token.
- `unknown` never passes. AI uncertainty can never widen scope.

---

## 4. Sealed Collection *(signature idea #4, part 2)*

**The idea:** the drone encrypts every recording to a public key whose private
half never leaves the ground HSM. **The aircraft cannot read back what it
recorded.** Neither can anyone who captures the aircraft, extracts the disk, or
compromises the onboard software after the fact.

```python
# kestrel/air/sealer.py
from nacl.public import SealedBox, PublicKey
import hashlib, json, os

class Sealer:
    """Write-only collection. Encrypt-to-ground; no decryption capability here."""

    def __init__(self, seal_pubkey_b64, vault="/vault/sealed"):
        self.box = SealedBox(PublicKey(base64.b64decode(seal_pubkey_b64)))
        self.vault = vault

    def seal(self, iq_bytes, meta):
        # session key for bulk data, sealed to the ground public key
        key = os.urandom(32)
        ct  = aes_gcm_encrypt(key, iq_bytes)
        sealed_key = self.box.encrypt(key)           # only ground HSM can open
        digest = hashlib.sha256(iq_bytes).hexdigest()
        del key                                       # not retained anywhere

        path = f"{self.vault}/{meta['detection_id']}.iqz"
        with open(path, "wb") as f:
            f.write(json.dumps({**meta, "sha256": digest,
                                "sealed_key": b64(sealed_key)}).encode() + b"\n\x00")
            f.write(ct)
        return path, digest
```

Note what is **absent**: there is no `unseal()` in `kestrel/air/`. The capability
does not exist in airborne code. A reviewer can verify this by grep, and a test
asserts it (§7).

**Layered with** full-disk LUKS encryption (Doc 03 §4) — so the disk is protected
at rest *and* each recording is individually sealed.

**On expiry or abort, the vault closes.** No new writes are possible; sealed
files remain, because they are evidence. An earlier draft called this
"crypto-erase", but the aircraft never holds key material that can decrypt a
recording, so there is nothing to erase that would protect the take. In the
implementation (`Sealer.close()`) each recording is a sealed box with its own
ephemeral key, and that key is gone as soon as the box is made. See Doc 14 §T6.

**Provenance.** Sealing gives confidentiality, not attribution: the seal
public key is printed in every warrant. Each recording is therefore also
signed by the airframe's own key over the sidecar and the ciphertext hash, and
checked on the ground before it is opened. See Doc 14 §T1.

---

## 5. The hash-chained audit log

Append-only. Each entry includes the hash of the previous entry, so any
modification or deletion anywhere in the history is detectable.

```python
# kestrel/ground/audit.py  (mirrored onboard)
import hashlib, json, time

class AuditLog:
    def __init__(self, path):
        self.path = path
        self.prev = self._last_hash() or "0"*64

    def append(self, event_type, payload):
        entry = {"seq": self._next_seq(), "t": time.time_ns(),
                 "type": event_type, "payload": payload, "prev": self.prev}
        entry["hash"] = hashlib.sha256(
            json.dumps(entry, sort_keys=True).encode()).hexdigest()
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        self.prev = entry["hash"]
        return entry["hash"]

    def verify(self):
        """Re-walk the chain. Any break = tampering."""
        prev = "0"*64
        for line in open(self.path):
            e = json.loads(line)
            if e["prev"] != prev: return False, e["seq"]
            body = {k: v for k, v in e.items() if k != "hash"}
            if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != e["hash"]:
                return False, e["seq"]
            prev = e["hash"]
        return True, None
```

**Logged events (minimum):**
`TOKEN_LOADED`, `TOKEN_REJECTED`, `SORTIE_START/END`, `DETECTION`,
`GATE_DECISION` (allow **and** deny, with reasons), `RECORDING_START/STOP`,
`RECORDING_SEALED` (with SHA-256), `MODEL_VERSION`, `GPS_LOCK_LOST`,
`STORAGE_FULL`, `REVOCATION`, `EXPORT_REQUESTED/APPROVED/COMPLETED`.

**Anchoring (optional but recommended):** publish the chain head hash at the end
of each sortie to an independent append-only service or a notarized record. It
prevents wholesale replacement of an entire log after the fact.

---

## 6. Evidence export — the second human gate

Sealed recordings are opened only on the ground, only with two people:

1. Analyst requests export, citing warrant ID and detection IDs.
2. A **second authorized person** approves.
3. HSM performs the unseal — the private key never leaves the device.
4. Integrity check: recomputed SHA-256 must match the sealed sidecar.
5. Export package is written with a full provenance manifest.
6. Every step is logged to the audit chain.

```python
def export(detection_ids, requester, approver, hsm, audit):
    if requester == approver:
        raise PermissionError("two-person rule")
    audit.append("EXPORT_REQUESTED", {"ids": detection_ids, "by": requester})
    ...
    audit.append("EXPORT_APPROVED", {"by": approver})
    for rid in detection_ids:
        blob = hsm.unseal(load(rid))
        assert sha256(blob) == sidecar(rid)["sha256"], "integrity failure"
        write_evidence_package(rid, blob, provenance_manifest(rid, audit))
    audit.append("EXPORT_COMPLETED", {"ids": detection_ids})
```

**Provenance manifest** accompanying every exported item — this is what a court
will actually examine:
warrant ID and text · aircraft ID and calibration date · GPSDO lock status ·
model versions used for classification · geolocation method, GDOP, error ellipse ·
full audit chain excerpt with verification result · chain of custody signatures.

---

## 7. Tests that must pass on every commit

```python
def test_no_recording_without_valid_token():
    gate = WarrantGate(expired_token(), audit)
    allowed, reasons = gate.decide(matching_detection(), pos_inside, now())
    assert not allowed and "token_expired" in reasons

def test_unknown_class_never_recorded():
    allowed, reasons = gate.decide(det(signal_class="unknown"), pos_inside, now())
    assert not allowed and "class_unknown" in reasons

def test_outside_geofence_never_recorded():
    allowed, _ = gate.decide(matching_detection(), pos_outside, now())
    assert not allowed

def test_tampered_token_rejected():
    raw = valid_token_raw(); raw["body"]["freq_ranges_hz"] = [[0, 8e9]]
    with pytest.raises(nacl.exceptions.BadSignatureError):
        WarrantToken(raw, AUTHORITY_PUBKEY)

def test_air_code_has_no_unseal_capability():
    """Structural: the airborne package must not be able to decrypt its take."""
    src = read_all_python("kestrel/air/")
    for forbidden in ("SealedBox.decrypt", "def unseal", "PrivateKey"):
        assert forbidden not in src

def test_no_write_path_bypasses_gate():
    """Structural: only sealer.py writes to the vault, and only gate callers use it."""
    writers = grep_calls_to("open(.*vault", "kestrel/air/")
    assert writers == ["kestrel/air/sealer.py"]

def test_audit_chain_detects_tampering():
    log = AuditLog(tmp); [log.append("DETECTION", {"i": i}) for i in range(10)]
    corrupt_line(tmp, 5)
    ok, seq = log.verify(); assert not ok and seq == 5

def test_every_recording_maps_to_authorized_detection():
    for rec in all_recordings(sortie):
        d = audit.find_detection(rec.detection_ref)
        assert d.warrant_match and token.covers(d.freq_hz, d.signal_class)
```

The last one is the whole subsystem in a single assertion. Run it after every
sortie against real data, not just in CI.

---

## 8. Questions your legal and aviation authorities must answer first

These change **data schemas**, not just paperwork — resolve them before Phase 1
implementation:

1. **Which statutory authority** governs collection? It determines the fields in
   `WarrantToken` and the retention rules.
2. **Minimization requirements** — must recording stop when traffic is determined
   to be non-relevant? If so, that's a real-time onboard rule, and it must be in
   the token schema.
3. **Retention and destruction** — how long may sealed material be held, and what
   is the certified destruction procedure?
4. **Key custody** — who holds the authority signing key and the seal private
   key? They must be *different* people/roles from the flight operators.
5. **Aviation authorization** — BVLOS approval, airspace, and whether autonomous
   operation is permitted without a pilot-in-command per aircraft.
6. **Notification/disclosure obligations** to subjects, after the fact.

Until 1–4 are answered, build against the schema in §2 and treat it as
provisional.

---

## 9. What this system deliberately cannot do

Documented here so it can be stated plainly in any review or authorization
request:

- **No transmit capability.** No jamming, no spoofing, no injection. There is no
  TX path in the hardware design.
- **No decryption of protected communications.** Encrypted traffic is detected
  and located as *metadata*; content is not attacked.
- **No collection outside a signed token.** Structurally impossible, not merely
  prohibited.
- **No identification of persons.** The system locates and fingerprints
  *devices*. Any association with a person is made by human investigators using
  other evidence, and reports must say so.
- **No autonomous scope expansion.** The AI can direct attention; it can never
  authorize collection.

---

Next: [`09-simulator.md`](09-simulator.md).
