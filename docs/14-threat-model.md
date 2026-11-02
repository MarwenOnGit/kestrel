# 14 — Threat Model for the Legal Spine

**As of 2026-11-02.** Doc 08 describes what the legal controls are meant to
guarantee. This document attacks them. For each threat it gives what an
attacker can do, what stops them now, the test that proves it, and what
remains open. Nothing here is a claim about flight hardware. It is a claim
about this code, with the residual risk stated rather than rounded down.

## Assets and adversaries

| Asset | Why it matters |
|---|---|
| Recorded content | Intercepted communications. The core privacy harm. |
| Collection metadata | Who was heard, when, where. For pattern-of-life this *is* the intelligence. |
| Evidence integrity | A recording a court relies on must be attributable and unmodified. |
| The warrant's limits | Time, place, band, class, budget, revocation. They are the legality. |

| Adversary | Has |
|---|---|
| **A1** Anyone holding a warrant token | Everything printed in it, including the seal public key. |
| **A2** Whoever recovers an aircraft | The disk, powered or not. Possibly the running system. |
| **A3** An insider with vault or log access after landing | The files, before the ground has checked them. |
| **A4** An RF adversary in the area | Can transmit, including GPS. |

---

## T1 — Forged or misattributed evidence (A1, A3) — **fixed**

**Before.** Recordings were anonymous sealed boxes. The seal public key is
in every warrant, so A1 could encrypt fabricated IQ to it, write a matching
sidecar, and the ground could not tell it from a real capture.
Confidentiality without provenance.

**Now.** Every airframe has an Ed25519 identity, provisioned on the ground
and registered by public key before the sortie (`kestrel/common/identity.py`).
The Sealer signs the sidecar together with the SHA-256 of the ciphertext on
disk. `kestrel/ground/evidence.py` checks the registered key, the signature,
and the ciphertext hash *before* decrypting, so a forged blob never reaches
the unsealer. Every sortie authenticates its whole vault.

**Tests:** `tests/test_evidence.py`. Forgery with only the warrant, a rogue
key claiming a real airframe ID, metadata edits, ciphertext swaps between two
genuine recordings, a stripped signature, and a checkpoint signature replayed
as a recording signature (domain separation).

**Open.** A stolen airframe key signs anything for that airframe until the
ground removes it from the registry. The key belongs in a secure element with
signing exposed and export disabled. Here it is a string, because this is a
simulator.

## T2 — Metadata exposed on a recovered aircraft (A2) — **open, structural**

"The drone cannot read its own take" is true of *content*. It is not true of
metadata. The sidecars and the onboard audit log are plaintext: frequency,
class, aircraft track, gate decisions, geolocation fixes. A recovered
aircraft tells A2 who was being watched.

Sealing the sidecar alone would be theatre, so it was not done. The audit log
holds the same fields, and the agent needs detections and tracks in
plaintext to fly the mission. An aircraft that cannot read what it is doing
cannot do it.

**Mitigations, in order of value.** Full-disk encryption with a TPM-sealed
key (Doc 03 §4) covers a powered-off aircraft. To cover a powered-on one: seal
audit payloads to the ground key per entry, keep the hash chain over the
ciphertext, and leave in plaintext only the few fields `WarrantGate.resume()`
needs (warrant ID, recorded duration, revocation, decision time). That is the
recommended next change. It is a schema change, so it needs the legal
review in Doc 08 §7 first.

## T3 — Rewriting the audit log (A2, A3) — **fixed, with a stated window**

**Before.** An unkeyed hash chain. Anyone with the disk could rewrite from
any entry onward, recompute every later hash, and `verify()` would pass.

**Now.** Signed CHECKPOINT entries over (seq, head) every N entries, plus a
closing checkpoint at sortie end. Each checkpoint is also sent to the ground
as it is made (`on_checkpoint`, the datalink in flight). `verify_audit()`
checks the chain, every checkpoint signature and its coverage, that the log
ends closed, and that it agrees with every checkpoint the ground already
holds.

**Tests:** `tests/test_audit_checkpoints.py`.

**Open, and pinned by tests rather than hidden.**
- Truncating back to an earlier genuine checkpoint gives a log that is
  internally perfect. Only the downlinked copy reveals it.
- An attacker with the airframe key can rewrite anything after the last
  checkpoint the ground received, a window exactly one checkpoint interval
  long. Shorter intervals shrink it; a secure element closes it.

## T4 — Resetting the warrant's limits (A2, A4) — **fixed for reboot and clock; GPS open**

**Before.** Revocation and recording time spent lived in memory and are not
part of the signed token. A power cycle silently un-revoked the warrant and
reset its budget. Expiry trusted the local clock, so winding it back past
expiry made a dead warrant valid again.

**Now.** `WarrantGate.resume()` rebuilds revocation, spent budget and the
latest decision time from the audit log, which is already durable and
checkpointed. There is no second copy to tamper with. `decide()` refuses with
`clock_rollback` when the clock steps more than 1 s behind the latest time it
has seen, including across a reboot.

**Tests:** `tests/test_warrant_persistence.py`. One test keeps the original
bug visible: without `resume()`, a reboot forgets a revocation.

**Open.** The geofence trusts the aircraft's GPS position, and GPS time is
spoofable by A4. A spoofed position inside the polygon satisfies the gate.
Mitigations: cross-check against inertial navigation, multi-constellation
receivers with integrity monitoring (RAIM), and logging the GPS integrity
state into the audit chain, so a spoofed sortie is at least visible
afterwards. A clock jumping *forward* is the safe direction, since the
warrant only expires sooner.

## T5 — Misclassification widening collection (A4, or nobody) — **found, mitigated, not fixed**

Doc 15 §3. On 92 real satellite downlinks, the rule-based classifier labelled
every one DMR or FM voice at 0.86–0.87, above the gate's 0.85 threshold. The
gate obeyed the warrant perfectly and would still have recorded signals the
warrant was never about. No attacker is needed for this one. A4 could exploit
it on purpose by transmitting constant-envelope FSK on a warranted channel.

**Mitigated** by exclusion ranges in the warrant (tested). **Not fixed**
until the trained classifier has open-set rejection (Step 4). It is
recorded as a strict xfail so it cannot be fixed by accident and forgotten.

## T6 — "Crypto-erase" — **claim corrected**

The README said the warrant "erases itself", and Doc 08 §4 described a
crypto-erase on expiry. The aircraft never holds key material that can
decrypt a recording, so there is nothing to erase that would protect the take.
What actually happens on expiry or abort is that the vault closes
(`Sealer.close()`) and no further writes are possible. Both documents now say
that.

## T7 — Tampered airborne software — **open**

`tests/test_structural.py` proves properties of the *source*: no decryption
capability in `kestrel/air/`, one write path to the vault, a deterministic
gate, no lab tooling in flight code. Nothing proves the aircraft runs that
source. This needs signed builds, measured boot, and the build hash recorded
in the audit log at sortie start. That last part is cheap and should come
first.

## Out of scope

Jamming and denial of service against the aircraft. RF attacks on the
datalink beyond the integrity of what it carries. Legal process questions,
which belong in Doc 08 §7.
