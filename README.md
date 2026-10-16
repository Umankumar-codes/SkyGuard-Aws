# SkyGuard AI — Smart India Hackathon 2026 Demo
# Problem Statement: 26073 — AI/ML-Based Intelligent Anomaly Detection for AWS
# Ministry of Earth Sciences | India Meteorological Department

## 1. What is this?

SkyGuard AI is a small, offline desktop demonstration prototype that simulates an
Automatic Weather Station (AWS) and uses a lightweight Isolation Forest machine
learning model to detect abnormal temperature, pressure, and humidity readings.
It is built for judges at the hackathon — not for production weather forecasting.

## 2. Problem being solved

AWS sensors can malfunction (frozen values, spikes, communication errors).
Incorrect observations mislead weather forecasts and disaster warnings.
This demo shows how AI can flag those errors and explain them clearly.

## 3. How it works

- A synthetic AWS simulator generates realistic temperature, pressure, and humidity.
- The Isolation Forest learns normal patterns from synthetic historical data.
- Each new reading is scored: normal vs. anomaly.
- If abnormal, the system shows type, severity (HIGH/MEDIUM), confidence (%),
  explanation, corrected estimate, and updated sensor health.

## 4. AI/ML approach

- **Model:** scikit-learn Isolation Forest (unsupervised, lightweight).
- **Training:** 2,000 synthetic normal observations generated locally.
- **Detection:** Point-level anomalies + frozen-streak rule + communication check.
- **Explainability:** Rule-based explanation mapped to feature deviations.

## 5. How anomaly detection works (simplified)

The Isolation Forest isolates outliers by randomly splitting data.
If an observation requires very few splits, it is far from normal and flagged.
For frozen sensors, we additionally check when temperature repeats exactly.

## 6. How sensor faults are simulated

Buttons inject faults one at a time so you can show each case clearly:
- Temperature Spike (55°C)
- Frozen Sensor (same value repeated)
- Humidity Fault (near 0%)
- Pressure Fault (850 hPa)
- Communication Error (no data)

## 7. How to install

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Requirements: numpy, scikit-learn, matplotlib. Tkinter is included with Python.

## 8. How to build the EXE

```cmd
build_exe.bat
```

This creates `dist\SkyGuardAI.exe`. You can move it to another Windows PC
without Python installed — it runs fully offline.

If PyInstaller fails, install it first:
```cmd
pip install pyinstaller
```

## 9. How to demonstrate to judges (1-minute sequence)

1. Open `SkyGuardAI.exe` (or `python main.py`).
2. Point to header and sensor cards — explain AWS simulation.
3. Click **START MONITORING** — show values changing live.
4. Click **Temperature Spike** — show status flip to 🔴 ANOMALY.
5. Point to Severity (HIGH), Confidence (~95%), and Explanation.
6. Point to **Suggested Corrected Reading** (AI estimate).
7. Show Sensor Health drop, event log entry.
8. Click **DEMO MODE** (optional) — let it run 30-60 seconds automatically.
9. Close by summarizing: "AI detects abnormality, explains why, estimates correction, tracks health — all offline, no cloud needed."

## Files

- `main.py` — GUI and orchestration
- `aws_simulator.py` — sensor simulation
- `anomaly_detector.py` — ML + explainability
- `build_exe.bat` — executable build
- `DEMO_SCRIPT.md` — word-for-word judge script
