"""KestrelSim - detection-level digital twin (Doc 09 §3).

Fast fidelity: skips waveform synthesis and produces Detection objects directly
from geometry and a propagation model. Runs ~1000x real time, which is what
makes autonomy development and regression testing practical.

Two details make it useful rather than decorative:
  * bearing error scales with SNR, so the planner genuinely learns that
    geometry and proximity matter;
  * each aircraft carries an independent clock error, so TDOA accuracy
    degrades the way it will in the air.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from kestrel.common.messages import Detection

C = 299_792_458.0


# --------------------------------------------------------------------------
def radio_horizon_m(h_rx_m: float, h_tx_m: float = 2.0, k: float = 4.0 / 3.0) -> float:
    """Line-of-sight distance over a curved earth, with the standard 4/3-earth
    factor for atmospheric refraction.

    At 120 m altitude this is about 50 km to a handheld at ground level. It is
    the single number that decides how much ground one aircraft can cover, and
    it is why altitude buys more sensitivity than any amplifier (Doc 00 §1).
    """
    R = 6_371_000.0
    return math.sqrt(2 * k * R * max(h_rx_m, 0.0)) + math.sqrt(2 * k * R * max(h_tx_m, 0.0))


def path_loss_db(d_m: float, freq_hz: float, excess_db: float = 0.0,
                 h_rx_m: float | None = None, h_tx_m: float = 2.0,
                 beyond_horizon_db_per_km: float = 1.2) -> float:
    """Free-space path loss, plus diffraction loss beyond the radio horizon.

    Free space alone is badly wrong at range: a 5 W handheld 300 km away still
    computes to +11 dB SNR, so without a horizon term the simulator reports
    detections from the other side of the country. Beyond line of sight the
    signal must diffract around the earth, and the loss climbs steeply.
    """
    d_m = max(d_m, 1.0)
    fspl = 20 * math.log10(d_m) + 20 * math.log10(freq_hz) - 147.55

    if h_rx_m is not None:
        horizon = radio_horizon_m(h_rx_m, h_tx_m)
        if d_m > horizon:
            excess_db += beyond_horizon_db_per_km * (d_m - horizon) / 1000.0
    return fspl + excess_db


# --------------------------------------------------------------------------
class BearingModel:
    """The direction-finding array's error behaviour, in one place.

    Both simulation paths need this: the detection-level path (which never
    touches a waveform) and the IQ path (whose single-channel synthesis cannot
    produce a phase front, so the array response has to be modelled rather than
    measured). Keeping one implementation means the two paths cannot silently
    disagree about how good DF is - if they did, a geolocation result would
    depend on which simulator produced it, which would make both meaningless.

    Replacing this with real interferometry is Step 8, and needs multi-channel
    IQ that `iq_synth` does not yet generate. See docs/13 §4.3.
    """

    def __init__(self, rng: np.random.Generator,
                 calibration_bias_sigma_deg: float = 2.0,
                 airframe_bias_amplitude_deg: float = 1.5,
                 multipath_rate: float = 0.03):
        self.rng = rng
        self.calibration_bias_sigma_deg = calibration_bias_sigma_deg
        self.airframe_bias_amplitude_deg = airframe_bias_amplitude_deg
        self.multipath_rate = multipath_rate
        self._cal_bias: dict[str, float] = {}

    def register(self, aircraft_id: str) -> None:
        """Residual calibration error: constant for this airframe for the whole
        sortie. Recalibration reduces it; nothing averages it out."""
        self._cal_bias.setdefault(aircraft_id, float(
            self.rng.normal(0, self.calibration_bias_sigma_deg)))

    @staticmethod
    def sigma_deg(snr_db: float) -> float:
        """Bearing accuracy improves with SNR - real physics, and the reason
        the planner benefits from closing range."""
        return max(0.5, 30.0 / math.sqrt(max(snr_db, 1.0)))

    def bias_deg(self, aircraft_id: str, true_bearing_deg: float) -> float:
        """Systematic bearing error: a constant per-airframe calibration
        residual plus an azimuth-dependent airframe reflection term."""
        const = self._cal_bias.get(aircraft_id, 0.0)
        airframe = self.airframe_bias_amplitude_deg * math.sin(
            2.0 * math.radians(true_bearing_deg))
        return const + airframe

    def measure(self, aircraft_id: str, from_xy, to_xy,
                snr_db: float) -> tuple[float, float]:
        """A measured bearing and the sigma the estimator should be told about.

        Note what the sigma does NOT include: the systematic terms. That is
        deliberate and true to life - an array reports its random uncertainty,
        and the calibration residual is exactly the error it cannot see. The
        tracker carries the systematic budget separately (docs/13 §3.5).
        """
        sigma = self.sigma_deg(snr_db)
        true_b = math.degrees(math.atan2(to_xy[1] - from_xy[1],
                                         to_xy[0] - from_xy[0])) % 360.0
        meas = (true_b + self.bias_deg(aircraft_id, true_b)
                + self.rng.normal(0, sigma)) % 360.0

        # Multipath: an occasional clean, confident, WRONG bearing off terrain.
        # This is what the robust loss in track.solve_batch exists to survive.
        if self.rng.random() < self.multipath_rate:
            meas = (meas + self.rng.normal(0, 20.0)) % 360.0
        return meas, sigma


# --------------------------------------------------------------------------
@dataclass
class Emitter:
    emitter_id: str
    xy: np.ndarray
    freq_hz: float
    mode: str
    eirp_dbm: float = 37.0
    burst_s: float = 4.0
    period_s: float = 240.0
    phase_s: float = 0.0            # offset so emitters do not key together
    mobile: bool = False
    speed_mps: float = 0.0
    note: str = ""
    fingerprint: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        self.xy = np.asarray(self.xy, dtype=float)
        if self.fingerprint is None:
            fp = np.random.randn(64)
            self.fingerprint = fp / np.linalg.norm(fp)

    def is_transmitting(self, t: float) -> bool:
        return ((t + self.phase_s) % self.period_s) < self.burst_s

    def step(self, dt: float, rng: np.random.Generator) -> None:
        if self.mobile and self.speed_mps > 0:
            heading = rng.uniform(0, 2 * math.pi)
            self.xy = self.xy + np.array([math.cos(heading), math.sin(heading)]) \
                * self.speed_mps * dt


@dataclass
class Aircraft:
    aircraft_id: str
    xy: np.ndarray
    alt_m: float = 120.0
    speed_mps: float = 15.0
    endurance_s: float = 35 * 60
    battery_pct: float = 100.0
    clock_error_ns: float = 0.0
    alive: bool = True
    gps_bias: np.ndarray = None

    def __post_init__(self):
        self.xy = np.asarray(self.xy, dtype=float)
        if self.gps_bias is None:
            self.gps_bias = np.zeros(2)

    def step(self, dt: float, waypoint) -> None:
        if not self.alive:
            return
        if waypoint is not None:
            d = np.asarray(waypoint, dtype=float) - self.xy
            n = float(np.linalg.norm(d))
            if n > 1e-6:
                self.xy = self.xy + d / n * min(self.speed_mps * dt, n)
        self.battery_pct = max(0.0, self.battery_pct - dt * 100.0 / self.endurance_s)


# --------------------------------------------------------------------------
class KestrelSim:
    def __init__(self, emitters: list[Emitter], aircraft: list[Aircraft],
                 noise_floor_dbm: float = -110.0,
                 detection_threshold_db: float = 5.0,
                 clock_sigma_ns: float = 15.0,
                 calibration_bias_sigma_deg: float = 2.0,
                 airframe_bias_amplitude_deg: float = 1.5,
                 multipath_rate: float = 0.03,
                 gps_sigma_m: float = 2.5,
                 seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.emitters = emitters
        self.aircraft = aircraft
        self.noise_floor_dbm = noise_floor_dbm
        self.detection_threshold_db = detection_threshold_db
        self.t = 0.0
        self.gps_sigma_m = gps_sigma_m
        self.bearing = BearingModel(
            self.rng,
            calibration_bias_sigma_deg=calibration_bias_sigma_deg,
            airframe_bias_amplitude_deg=airframe_bias_amplitude_deg,
            multipath_rate=multipath_rate)
        for a in aircraft:
            a.clock_error_ns = float(self.rng.normal(0, clock_sigma_ns))
            self.bearing.register(a.aircraft_id)

    # ------------------------------------------------------------------
    @property
    def alive_aircraft(self) -> list[Aircraft]:
        return [a for a in self.aircraft if a.alive]

    def kill_aircraft(self, aircraft_id: str) -> None:
        for a in self.aircraft:
            if a.aircraft_id == aircraft_id:
                a.alive = False

    # ------------------------------------------------------------------
    def step(self, dt: float, waypoints: dict[str, tuple] | None = None,
             generate: bool = True) -> list[Detection]:
        """Advance the world. `generate=False` moves everything but produces no
        detections - used when an `RfSensor` is finding them in samples
        instead, where geometry-derived detections would be both wasted work
        and a temptation to fall back on."""
        waypoints = waypoints or {}
        self.t += dt
        for e in self.emitters:
            e.step(dt, self.rng)
        for a in self.aircraft:
            a.step(dt, waypoints.get(a.aircraft_id))
            a.clock_error_ns += float(self.rng.normal(0, 0.5))   # slow drift
            # Ornstein-Uhlenbeck style random walk with a long correlation time
            a.gps_bias = (0.999 * a.gps_bias
                          + self.rng.normal(0, self.gps_sigma_m * 0.05, 2))
        return self.generate_detections() if generate else []

    # ------------------------------------------------------------------
    def generate_detections(self) -> list[Detection]:
        out: list[Detection] = []
        for a in self.alive_aircraft:
            for e in self.emitters:
                if not e.is_transmitting(self.t):
                    continue
                d = float(np.linalg.norm(e.xy - a.xy))
                slant = math.sqrt(d * d + a.alt_m * a.alt_m)
                rx_dbm = e.eirp_dbm - path_loss_db(slant, e.freq_hz,
                                                   h_rx_m=a.alt_m)
                snr = rx_dbm - self.noise_floor_dbm
                if snr < self.detection_threshold_db:
                    continue

                # SYSTEMATIC error lives in BearingModel. Without it the
                # simulator lies: averaging hundreds of zero-mean errors drives
                # the fix to sub-metre accuracy, which no real DF system
                # achieves (Doc 07 §7 predicts 200-800 m for single-aircraft
                # AOA).
                meas_b, sigma_deg = self.bearing.measure(
                    a.aircraft_id, a.xy, e.xy, snr)

                t_arrival_ns = int((self.t + slant / C) * 1e9 + a.clock_error_ns)
                reported_xy = self.reported_position(a)

                out.append(Detection(
                    aircraft_id=a.aircraft_id,
                    t_gpsdo_ns=t_arrival_ns,
                    freq_hz=e.freq_hz,
                    bandwidth_hz=self._bandwidth_for(e.mode),
                    snr_db=snr,
                    aircraft_pos_xy=(float(reported_xy[0]), float(reported_xy[1])),
                    aircraft_alt_m=a.alt_m,
                    bearing_deg=meas_b,
                    bearing_sigma_deg=sigma_deg,
                    truth_emitter_id=e.emitter_id,
                    truth_class=e.mode,
                    truth_xy=(float(e.xy[0]), float(e.xy[1])),
                ))
        return out

    def reported_position(self, a: Aircraft) -> np.ndarray:
        """Where the aircraft *thinks* it is.

        GPS error floors how well any amount of averaging can locate an
        emitter. It is CORRELATED over minutes (ionosphere, satellite geometry,
        multipath off terrain), not independent per sample: white noise would
        average away over hundreds of detections and leave the fix falsely
        precise; a slowly drifting bias does not, which is how real GPS behaves.
        """
        return a.xy + a.gps_bias + self.rng.normal(0, 0.5, 2)

    @staticmethod
    def _bandwidth_for(mode: str) -> float:
        return {
            "fm_voice": 12_500.0, "am_voice": 8_000.0,
            "dmr": 12_500.0, "tetra": 25_000.0, "p25": 12_500.0,
            "lora": 125_000.0, "wifi": 20e6, "bluetooth": 1e6,
            "lte": 10e6, "nr5g": 20e6,
            "ocusync": 10e6, "satcom_terminal": 200_000.0,
        }.get(mode, 25_000.0)
