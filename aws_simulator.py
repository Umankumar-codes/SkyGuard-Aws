"""
aws_simulator.py — Advanced Automatic Weather Station (AWS) Simulator
Supports multi-station profiles, sensor fault injection, calibration drift,
and genuine meteorological event simulation for SIH 2026 (PS 26073).
"""
import numpy as np


STATION_PROFILES = {
    "AWS-101 [New Delhi / Safdarjung]": {
        "region": "Subtropical Plains (MoES Station 42182)",
        "t_mean": 28.0, "t_std": 3.8,
        "p_mean": 1008.0, "p_std": 5.5,
        "h_mean": 62.0, "h_std": 10.0,
        "elevation": "216m ASL"
    },
    "AWS-204 [Leh / High-Altitude]": {
        "region": "High-Altitude Cold Desert (MoES Station 42027)",
        "t_mean": 9.5, "t_std": 3.2,
        "p_mean": 712.0, "p_std": 4.2,
        "h_mean": 32.0, "h_std": 8.0,
        "elevation": "3,524m ASL"
    },
    "AWS-308 [Mumbai / Colaba Marine]": {
        "region": "Coastal Maritime Station (MoES Station 43057)",
        "t_mean": 31.5, "t_std": 2.2,
        "p_mean": 1012.5, "p_std": 3.5,
        "h_mean": 84.0, "h_std": 7.5,
        "elevation": "11m ASL"
    },
    "AWS-415 [Jaisalmer / Thar Desert]": {
        "region": "Arid Desert Zone (MoES Station 42328)",
        "t_mean": 39.5, "t_std": 4.5,
        "p_mean": 997.0, "p_std": 5.0,
        "h_mean": 22.0, "h_std": 6.5,
        "elevation": "225m ASL"
    }
}


class AWSSimulator:
    """Lightweight and realistic AWS sensor simulator with deterministic normal readings,
    multi-station geography, fault modes, and natural weather phenomenon simulation."""

    def __init__(self, seed=42, station="AWS-101 [New Delhi / Safdarjung]"):
        self.rng = np.random.default_rng(seed)
        self.station = station
        self.set_station(station)
        self.frozen_temp = None
        self.frozen_count = 0
        self.drift_offset = 0.0
        self.drift_active = False
        self.step_counter = 0

    def set_station(self, station_name):
        if station_name in STATION_PROFILES:
            self.station = station_name
        else:
            self.station = list(STATION_PROFILES.keys())[0]
        p = STATION_PROFILES[self.station]
        self.t_mean = p["t_mean"]
        self.p_mean = p["p_mean"]
        self.h_mean = p["h_mean"]
        self.t_std = p["t_std"]
        self.p_std = p["p_std"]
        self.h_std = p["h_std"]
        self.last_normal = (self.t_mean, self.p_mean, self.h_mean)

    def generate_normal(self):
        """Generates realistic diurnal weather stream with micro-fluctuations."""
        self.step_counter += 1
        # Slight diurnal sinusoidal drift
        diurnal_t = 1.2 * np.sin(self.step_counter / 15.0)
        diurnal_p = -0.6 * np.sin(self.step_counter / 15.0)

        temp = float(np.clip(self.rng.normal(self.t_mean + diurnal_t, self.t_std * 0.4),
                             self.t_mean - 12.0, self.t_mean + 12.0))

        # Physical negative correlation between T and P
        pressure_shift = (temp - self.t_mean) * -0.25 + diurnal_p
        pressure = float(np.clip(self.rng.normal(self.p_mean + pressure_shift, self.p_std * 0.35),
                                 self.p_mean - 25.0, self.p_mean + 25.0))

        # Physical negative correlation between T and H (higher temp -> lower RH in non-monsoon)
        hum_shift = (temp - self.t_mean) * -0.8
        humidity = float(np.clip(self.rng.normal(self.h_mean + hum_shift, self.h_std * 0.4),
                                 5.0, 99.0))

        # Apply calibration drift if active
        if self.drift_active:
            self.drift_offset += 0.4
            temp += self.drift_offset

        reading = (round(temp, 1), round(pressure, 1), round(humidity, 1))
        self.last_normal = reading
        self.frozen_count = 0
        return reading

    def inject_temperature_spike(self):
        """Sudden anomalous spike in temperature due to solar radiation shield failure or sensor ADC glitch."""
        spike_val = self.t_mean + 27.5 + float(self.rng.uniform(-1.5, 2.0))
        return (round(spike_val, 1), self.last_normal[1], self.last_normal[2])

    def inject_frozen_sensor(self):
        """Sensor value remains completely static across ticks (Stuck ADC/Transducer)."""
        if self.frozen_temp is None:
            self.frozen_temp = self.last_normal[0]
        self.frozen_count += 1
        return (self.frozen_temp, self.last_normal[1], self.last_normal[2])

    def inject_humidity_fault(self):
        """Relative Humidity drops to near zero or unphysical jump (Capacitive polymer breakdown)."""
        return (self.last_normal[0], self.last_normal[1], 0.2)

    def inject_pressure_fault(self):
        """Severe barometric sensor glitch / transducer leak."""
        bad_pressure = max(600.0, self.p_mean - 160.0)
        return (self.last_normal[0], round(bad_pressure, 1), self.last_normal[2])

    def inject_communication_error(self):
        """Telemetry packet dropped or parity check failure (No observation received)."""
        return None

    def inject_drift(self):
        """Gradual sensor calibration drift (slow thermal bias buildup)."""
        self.drift_active = True
        self.drift_offset = 3.5
        temp = self.last_normal[0] + self.drift_offset
        return (round(temp, 1), self.last_normal[1], self.last_normal[2])

    def inject_genuine_weather_event(self):
        """Simulates a genuine intense convective thunderstorm / squall front.
        Key for SIH: All 3 parameters move in strict physical coherence:
        Temp drops rapidly, Pressure drops, Humidity spikes to 95%+.
        AI should identify this as GENUINE METEOROLOGICAL EVENT (Not a sensor fault!)."""
        storm_temp = round(self.t_mean - 6.5, 1)
        storm_pressure = round(self.p_mean - 8.5, 1)
        storm_humidity = round(min(96.0, self.h_mean + 32.0), 1)
        return (storm_temp, storm_pressure, storm_humidity)

    def reset_fault(self):
        """Resets sensor to nominal baseline."""
        self.frozen_temp = None
        self.frozen_count = 0
        self.drift_active = False
        self.drift_offset = 0.0
        return self.generate_normal()
