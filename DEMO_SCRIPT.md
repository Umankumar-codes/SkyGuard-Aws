# SkyGuard AI — Demo Script for Judges (Smart India Hackathon 2026)

## Opening (10 seconds)

"Automatic Weather Stations continuously send environmental observations — temperature, atmospheric pressure, and relative humidity. But a faulty sensor can report incorrect values: a frozen reading, a sudden spike, or a missing observation. Today I'm showing SkyGuard AI, a lightweight offline prototype that detects those anomalies using machine learning, explains why, and estimates corrected values — all without the internet."

## Normal operation (15 seconds)

- Click **START MONITORING**.
- Point to the three cards: Temperature, Pressure, Humidity.
- Explain: "These are simulated AWS readings. The AI is learning normal weather patterns from synthetic historical data."
- Let it run for 2-3 ticks to show live updates.

## Anomaly demonstration (30 seconds)

Click **Temperature Spike**.

"The temperature suddenly jumps to 55°C. Instead of relying only on a fixed threshold, SkyGuard AI analyzes the learned behavior of the sensor and detects this observation as anomalous."

Point to:
- **SENSOR STATUS:** 🔴 ANOMALY DETECTED
- **Severity:** HIGH
- **Confidence:** ~95%
- **WHY WAS THIS FLAGGED?** explanation text
- **Suggested Corrected Reading:** ~27°C (AI estimate)
- **Sensor Health:** drop shown
- **Event Log:** timestamped entry

Then click **Frozen Sensor** briefly to show frozen detection.

Then click **Communication Error** to show no-data handling.

## Demo Mode (optional, 30-60 seconds)

Click **🎬 DEMO MODE**. Let it run. It automatically walks through normal → spike → frozen → humidity fault → pressure fault → communication error → back to normal.

While it runs, say:

"This demonstration mode is designed for presentations — it runs automatically through the full anomaly lifecycle so you can see detection, explanation, correction, and health tracking without manual clicks."

## Closing (10 seconds)

"SkyGuard AI demonstrates that lightweight, explainable ML can support meteorological monitoring — offline, transparent, and ready for real-world AWS deployment. Thank you."

---

## Click cues for you

- **Start Monitoring**
- **Temperature Spike**
- **Frozen Sensor**
- **Humidity Fault**
- **Pressure Fault**
- **Communication Error**
- **Demo Mode**
- **Stop Monitoring**
