# 03 — Software Setup

Exact commands to bring up the ground station and each aircraft. Copy-paste
runnable. Versions pinned where it matters.

---

## 1. Repository layout

```
drone-project/
├── README.md
├── docs/                     # this documentation
├── schemas/                  # JSON schemas for the interface contracts
├── kestrel/
│   ├── air/                  # runs on the Jetson
│   │   ├── sdr_source.py     # UHD → IQ frames
│   │   ├── detector.py       # FFT, noise floor, CFAR tripwire
│   │   ├── classifier.py     # TensorRT inference
│   │   ├── weather_map.py    # novelty / anomaly scoring
│   │   ├── warrant_gate.py   # scope enforcement (NO AI HERE)
│   │   ├── sealer.py         # encrypt-to-ground-pubkey recording
│   │   ├── agent.py          # autonomy orchestrator (behavior tree)
│   │   ├── planner.py        # information-gain waypoint planner
│   │   ├── safety.py         # hard vetoes
│   │   ├── swarm.py          # gossip: tasking + recruitment
│   │   └── mavlink_io.py     # PX4 interface
│   ├── ground/
│   │   ├── signer.py         # warrant token creation (HSM)
│   │   ├── fusion.py         # AOA + TDOA → GeoFix
│   │   ├── audit.py          # hash-chained log
│   │   ├── patternoflife.py  # cross-sortie correlation
│   │   ├── export.py         # 2-person evidence export
│   │   └── webmap/           # Leaflet live map
│   ├── models/               # trained + TensorRT-compiled models
│   ├── sim/                  # KestrelSim digital twin
│   └── common/               # schemas, crypto, message types
├── tests/
└── tools/                    # calibration, bench scripts
```

Create it:
```bash
cd /home/dexter/drone-project
mkdir -p schemas tests tools \
  kestrel/{air,ground,models,sim,common} \
  kestrel/ground/webmap
```

---

## 2. Ground station setup (Ubuntu 22.04)

```bash
# System deps
sudo apt update
sudo apt install -y build-essential cmake git python3-pip python3-venv \
    libuhd-dev uhd-host gnuradio gnuradio-dev \
    libsodium-dev pkg-config sqlite3

# Download USRP FPGA images (needed once)
sudo uhd_images_downloader

# Python environment
cd /home/dexter/drone-project
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy matplotlib pyzmq pynacl \
    torch torchvision  \
    fastapi uvicorn folium geopy shapely \
    pytest pytest-cov ruff
```

**Verify GNU Radio + UHD:**
```bash
python3 -c "import uhd; print(uhd.get_version_string())"
gnuradio-config-info --version
```

---

## 3. Aircraft (Jetson AGX Orin) setup

Flash with **JetPack 6.x** using NVIDIA SDK Manager, then:

```bash
# UHD for the USRP
sudo apt install -y libuhd-dev uhd-host
sudo uhd_images_downloader

# 10 GbE tuning — REQUIRED or you will drop samples
sudo sysctl -w net.core.rmem_max=625000000
sudo sysctl -w net.core.wmem_max=625000000
sudo ip link set eth0 mtu 9000     # jumbo frames
# persist:
echo -e "net.core.rmem_max=625000000\nnet.core.wmem_max=625000000" | \
  sudo tee /etc/sysctl.d/99-uhd.conf

# Max performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# Python
pip3 install numpy scipy pyzmq pynacl pymavlink
# TensorRT + torch come with JetPack
```

**Verify the USRP is visible:**
```bash
uhd_find_devices
uhd_usrp_probe            # prints channels, daughterboards, clock sources
```

**Streaming benchmark — must show zero overflows:**
```bash
uhd_rx_cpp --args "type=x4xx" --rate 200e6 --freq 450e6 --duration 60 --null
# 'O' characters in output = overflow = you are losing samples. Fix before proceeding.
```

---

## 4. Encrypted storage (the sealed recording vault)

```bash
# One-time: create an encrypted volume on the NVMe
sudo cryptsetup luksFormat /dev/nvme0n1p1
sudo cryptsetup open /dev/nvme0n1p1 kestrelvault
sudo mkfs.ext4 /dev/mapper/kestrelvault
sudo mkdir -p /vault && sudo mount /dev/mapper/kestrelvault /vault
sudo mkdir -p /vault/sealed && sudo chown $USER /vault/sealed
```

This is **disk-at-rest** encryption. It is the *second* layer. The first layer is
**sealing** (Doc 08 §4): each recording is separately encrypted to the ground
station's public key, so even with the disk unlocked, the aircraft cannot read
its own take.

---

## 5. GPSDO lock (required for TDOA)

```bash
uhd_usrp_probe --args "type=x4xx,clock_source=gpsdo"
```
Then in Python:
```python
import uhd
usrp = uhd.usrp.MultiUSRP("type=x4xx")
usrp.set_clock_source("gpsdo")
usrp.set_time_source("gpsdo")
print("GPS locked:", usrp.get_mboard_sensor("gps_locked").to_bool())
print("Ref locked:", usrp.get_mboard_sensor("ref_locked").to_bool())
usrp.set_time_next_pps(uhd.types.TimeSpec(usrp.get_mboard_sensor("gps_time").to_int() + 1))
```

**Do not proceed to Phase 2 until `gps_locked` and `ref_locked` are both true and
stable for 10 minutes.** Everything in geolocation rests on this.

---

## 6. PX4 / Pixhawk

Flash PX4 stable via QGroundControl. Then configure:
- **Companion computer** link: `MAV_1_CONFIG` = TELEM2, `SER_TEL2_BAUD` = 921600
- Enable **offboard** mode (the agent commands waypoints)
- Set **geofence** (`GF_*` params) and **RTL** failsafes — these are the last-
  resort safety net *below* our own safety layer

Verify from the Jetson:
```bash
pip3 install pymavlink
python3 -c "
from pymavlink import mavutil
m = mavutil.mavlink_connection('/dev/ttyTHS1', baud=921600)
m.wait_heartbeat(); print('PX4 heartbeat OK, sysid', m.target_system)"
```

---

## 7. Mesh radio

Configure per vendor docs. Requirements:
- AES-256 encryption enabled with a mission key
- Aircraft and ground on the same mesh, distinct node IDs
- **Notch your mesh band out of the survey plan** (Doc 02 §5) — otherwise every
  sortie "discovers" your own datalink

Verify:
```bash
ping -c 5 <ground-node-ip>
iperf3 -c <ground-node-ip>      # confirm usable throughput at range
```

---

## 8. Time sync across the swarm

All aircraft derive time from their GPSDO, not from NTP. Verify pairwise:
```bash
python3 tools/check_swarm_time.py --nodes K1,K2,K3
# Must report pairwise offset < 50 ns
```
(Write this tool in Phase 2; it compares `usrp.get_time_now()` latched on a
common PPS edge.)

---

## 9. Running the stack

**Aircraft:**
```bash
cd /home/dexter/drone-project
python3 -m kestrel.air.main --config config/aircraft_K1.yaml \
    --warrant /vault/warrant_token.json
```
The process refuses to start if the warrant token signature is invalid or
expired. That is intentional.

**Ground:**
```bash
python3 -m kestrel.ground.main --config config/ground.yaml
# Web map at http://localhost:8000
```

---

## 10. Smoke test (no radio needed)

Before touching hardware, prove the software chain with simulated IQ:
```bash
pytest tests/ -v
python3 -m kestrel.sim.replay --file tests/data/dmr_sample.iq \
    --expect-class dmr --expect-warrant-match true
```
This uses KestrelSim (Doc 09) and should pass on any laptop. **Develop against
the simulator; use hardware only to validate.**

---

Next: [`04-signal-processing.md`](04-signal-processing.md).
