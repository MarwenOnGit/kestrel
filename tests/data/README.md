# Test recordings

Two short recordings of real amateur-satellite downlinks, used by
`tests/test_real_signals.py` as fixtures the simulator did not make.

| File | Satellite | Physical layer | Frames |
|---|---|---|---|
| `ops_sat.wav` | OPS-SAT (ESA), DP0OPS | 9600 bit/s FSK, G3RUH scrambled | 1 AX.25 UI frame |
| `swiatowid-ax25.wav` | Światowid, SR6SAT-6 | 1200 bit/s AFSK (Bell 202) | 2 AX.25 UI frames |

Both are 48 kHz mono FM-discriminator audio, unmodified, from Daniel Estévez's
[satellite-recordings](https://github.com/daniestevez/satellite-recordings)
collection, released into the public domain under the Unlicense.
