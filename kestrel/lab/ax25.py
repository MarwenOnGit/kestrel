"""AX.25 over the two most common amateur-satellite physical layers (Doc 15).

    audio --demod--> bits --NRZI--> --HDLC deframe--> frame --FCS--> AX.25

* AFSK 1200 (Bell 202): mark 1200 Hz, space 2200 Hz, 1200 bit/s.
* FSK 9600 with G3RUH scrambling: baseband NRZ, scrambled with x^17 + x^12 + 1.

The input is FM-discriminator audio, which is what most amateur ground stations
record. Every frame returned has passed the 16-bit FCS, so a decode is either
right or absent: the CRC is the ground truth this module checks itself
against. Clock recovery is deliberately crude (try every sampling phase, keep
what passes the FCS); it is a validation tool, not a modem.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FLAG = [0, 1, 1, 1, 1, 1, 1, 0]


# --------------------------------------------------------------------------
# FCS: CRC-16/X.25 (reflected CCITT, init 0xFFFF, xorout 0xFFFF)
# --------------------------------------------------------------------------
def crc16_x25(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8408 if crc & 1 else crc >> 1
    return crc ^ 0xFFFF


# --------------------------------------------------------------------------
# Line coding
# --------------------------------------------------------------------------
def nrzi_decode(levels: np.ndarray) -> np.ndarray:
    """No transition -> 1, transition -> 0."""
    levels = levels.astype(np.uint8)
    prev = np.concatenate(([levels[0]], levels[:-1]))
    return (levels == prev).astype(np.uint8)


def g3ruh_descramble(bits: np.ndarray) -> np.ndarray:
    """Self-synchronising descrambler for x^17 + x^12 + 1: out[n] = in[n] ^
    in[n-12] ^ in[n-17]. Garbage for the first 17 bits, correct after."""
    b = bits.astype(np.uint8)
    out = b.copy()
    out[12:] ^= b[:-12]
    out[17:] ^= b[:-17]
    return out


def hdlc_frames(bits: np.ndarray, min_bytes: int = 17) -> list[bytes]:
    """Split on flags, remove bit stuffing, keep byte-aligned candidates whose
    FCS checks. 17 bytes = two addresses + control + FCS, the smallest legal
    AX.25 frame."""
    frames, cur, ones, in_frame = [], [], 0, False
    for bit in bits.tolist():
        if bit == 1:
            ones += 1
            if ones > 6:                      # abort sequence
                in_frame, cur = False, []
                continue
            cur.append(1)
            continue
        # bit == 0
        if ones == 6:                         # 01111110: a flag
            body = cur[:-7]                   # drop the flag's leading 0 and six 1s
            if in_frame and len(body) >= 8 * min_bytes and len(body) % 8 == 0:
                raw = np.packbits(np.array(body, dtype=np.uint8)
                                  .reshape(-1, 8)[:, ::-1]).tobytes()
                if crc16_x25(raw[:-2]) == int.from_bytes(raw[-2:], "little"):
                    frames.append(raw[:-2])
            in_frame, cur = True, []
        elif ones == 5:                       # stuffed zero: drop it
            pass
        else:
            cur.append(0)
        ones = 0
    return frames


# --------------------------------------------------------------------------
# Demodulators (FM audio in)
# --------------------------------------------------------------------------
def _moving_sum(x: np.ndarray, n: int) -> np.ndarray:
    c = np.cumsum(np.concatenate(([0], x)))
    return c[n:] - c[:-n]


def afsk1200_soft(audio: np.ndarray, fs: float) -> np.ndarray:
    """Per-sample mark-minus-space energy over one bit period."""
    n = int(round(fs / 1200))
    t = np.arange(len(audio)) / fs
    out = []
    for f in (1200.0, 2200.0):
        z = audio * np.exp(-2j * np.pi * f * t)
        out.append(np.abs(_moving_sum(z.real, n) + 1j * _moving_sum(z.imag, n)))
    return out[0] - out[1]


def _slice_all_phases(soft: np.ndarray, sps: float, step: int = 1):
    for phase in range(0, max(int(round(sps)), 1), step):
        idx = np.arange(phase, len(soft), sps).astype(int)
        yield (soft[idx] > 0).astype(np.uint8)


def decode_afsk1200(audio: np.ndarray, fs: float) -> list[bytes]:
    soft = afsk1200_soft(audio - np.mean(audio), fs)
    found: list[bytes] = []
    for levels in _slice_all_phases(soft, fs / 1200, step=2):
        for f in hdlc_frames(nrzi_decode(levels)):
            if f not in found:
                found.append(f)
    return found


def decode_fsk9600_g3ruh(audio: np.ndarray, fs: float) -> list[bytes]:
    x = audio - np.mean(audio)
    sps = fs / 9600
    # matched filter for NRZ: average over one bit
    soft = np.convolve(x, np.ones(int(round(sps))) / round(sps), mode="same")
    found: list[bytes] = []
    for polarity in (1, -1):
        for levels in _slice_all_phases(polarity * soft, sps):
            bits = nrzi_decode(g3ruh_descramble(levels))
            for f in hdlc_frames(bits):
                if f not in found:
                    found.append(f)
    return found


# --------------------------------------------------------------------------
# AX.25 header
# --------------------------------------------------------------------------
@dataclass
class Ax25Frame:
    destination: str
    source: str
    digipeaters: list[str] = field(default_factory=list)
    control: int = 0
    pid: int | None = None
    info: bytes = b""

    @property
    def kind(self) -> str:
        if self.control & 1 == 0:
            return "I"
        return "UI" if (self.control & 0xEF) == 0x03 else (
            "S" if self.control & 3 == 1 else "U")

    def summary(self) -> str:
        path = ",".join([self.destination] + self.digipeaters)
        text = self.info.decode("ascii", "replace")
        printable = sum(32 <= b < 127 for b in self.info) / max(len(self.info), 1)
        body = (repr(text[:60]) if printable > 0.9
                else self.info[:24].hex() + ("..." if len(self.info) > 24 else ""))
        pid = f" PID=0x{self.pid:02X}" if self.pid is not None else ""
        return f"{self.source}>{path} {self.kind}{pid} [{len(self.info)} B] {body}"


_CALL_CHARS = set(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def _callsign(b: bytes) -> tuple[str, bool]:
    """AX.25 2.2 §3.12: six upper-case alphanumerics shifted left one bit,
    space-padded, then an SSID byte whose low bit marks the last address.
    Anything else is not an AX.25 address - a CRC-valid HDLC frame can carry
    a different protocol entirely, and reporting it as AX.25 would be a lie."""
    raw = bytes(c >> 1 for c in b[:6])
    call = raw.rstrip(b" ")
    if not call or any(c not in _CALL_CHARS for c in call) or b" " in call:
        raise ValueError(f"non-conforming address {raw!r}")
    ssid = (b[6] >> 1) & 0x0F
    name = call.decode("ascii")
    return (f"{name}-{ssid}" if ssid else name), bool(b[6] & 1)


def parse_ax25(frame: bytes) -> Ax25Frame:
    """Raises ValueError on anything that is not a well-formed AX.25 header."""
    addrs, i, last = [], 0, False
    while not last:
        if i + 7 > len(frame) or len(addrs) > 9:
            raise ValueError("address field runs off the frame")
        call, last = _callsign(frame[i:i + 7])
        addrs.append(call)
        i += 7
    if len(addrs) < 2:
        raise ValueError("fewer than two addresses")
    if i >= len(frame):
        raise ValueError("no control field")
    control = frame[i]
    i += 1
    pid = None
    is_i = control & 1 == 0
    is_ui = (control & 0xEF) == 0x03
    if (is_i or is_ui) and i < len(frame):
        pid = frame[i]
        i += 1
    return Ax25Frame(addrs[0], addrs[1], addrs[2:], control, pid, frame[i:])
