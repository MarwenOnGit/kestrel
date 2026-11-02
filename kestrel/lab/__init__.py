"""Lab tooling for validating KESTREL against real recordings.

Nothing in here is flight code, and `kestrel/air/` must never import it
(tests/test_structural.py enforces that). It exists to answer one question the
simulator cannot: what happens when the DSP chain meets a signal nobody
synthesized. Inputs are public recordings of amateur satellites.
"""
