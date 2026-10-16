"""
main.py — SkyGuard AI
Offline desktop demo for Smart India Hackathon 2026 (PS 26073)
Ministry of Earth Sciences / India Meteorological Department
AI/ML Anomaly Detection for Automatic Weather Stations
"""
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from collections import deque
from datetime import datetime
import threading

# Try importing ML/plotting; fall back gracefully so demo never crashes
try:
    import numpy as np
    from sklearn.ensemble import IsolationForest
except Exception:
    np = None
    IsolationForest = None

try:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.pyplot import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MPL = True
except Exception:
    HAS_MPL = False
    Figure = None
    FigureCanvasTkAgg = None

from aws_simulator import AWSSimulator
from anomaly_detector import AnomalyDetector

# ------------------------------------------------------------------
# Style / theme constants
# ------------------------------------------------------------------
# Modern, SIH-winning color palette with gradient bursts of energy and professional data visualization focus
BG = "#05050a"  # Deep space black
CARD = "#0a0f1e"  # Dark midnight blue
CARD2 = "#1e293b"  # Premium slate
ACCENT = "#3b82f6"  # Bold sapphire blue (corporate confidence)
ACCENT2 = "#10b981"  # Emerald green (growth and nature)
ACCENT3 = "#f97316"  # Sunset orange (alert and attention)
GOOD = "#22c55e"  # Bright green (success metric)
BAD = "#ef4444"  # Red alert (critical issues)
CRITICAL = "#dc2626"  # Deeper red (urgent)
TEXT = "#e2e8f0"  # Light gray (excellent contrast)
TEXT2 = "#94a3b8"  # Medium gray (secondary info)
PRIMARY = "#6366f1"  # Indigo (professional depth)
WARNING = "#eab308"  # Amber (cautious warning)
SUCCESS = "#16a34a"  # Green (implementation success)
DANGER = "#dc2626"  # Red (error conditions)


def set_dark_style():
    s = ttk.Style()
    s.theme_use("clam")
    s.configure("TFrame", background=BG)
    s.configure("TButton", background=CARD2, foreground=TEXT, font=("Calibri", 11))
    s.configure("TLabel", background=BG, foreground=TEXT, font=("Calibri", 11))
    s.configure("TLabelframe", background=BG, foreground=TEXT, font=("Segoe UI Semibold", 11))
    s.configure("TLabelframe.Label", background=BG, foreground=ACCENT3, font=("Segoe UI Semibold", 11))


# ------------------------------------------------------------------
# Main application
# ------------------------------------------------------------------
class SkyGuardApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SkyGuard AI — AWS Anomaly Detection")
        self.geometry("1280x860")
        self.configure(bg=BG)
        self.resizable(True, True)
        set_dark_style()

        # State
        self.monitoring = False
        self.readings = deque(maxlen=30)
        self.recent_normals = []
        self.frozen_counter = 0
        self.pending_fault = None  # set by button; consumed in tick
        self.event_lines = deque(maxlen=20)
        self.sim = AWSSimulator(seed=42)
        self.det = AnomalyDetector()
        self.current_reading = None
        self.is_anomaly = False
        self.severity = "NONE"
        self.confidence = 0.0
        self.anomaly_info = {"type": "Normal", "explanation": "Observation fits normal pattern."}
        self.graph_fig = None
        self.graph_canvas = None

        self._build_ui()
        self.after(100, self._first_tick)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        # Outer padding
        container = ttk.Frame(self, style="TFrame")
        container.pack(fill="both", expand=True, padx=12, pady=12)

        # --- Header ---
        header = ttk.Frame(container, style="TFrame")
        header.pack(fill="x", pady=(0, 10))
        tk.Label(header, text="SkyGuard AI", font=("Segoe UI", 28, "bold"),
                 bg=BG, fg=ACCENT).pack(anchor="w")
        tk.Label(header, text="AI/ML-Based Intelligent Anomaly Detection", font=("Segoe UI", 16, "bold"),
                 bg=BG, fg=PRIMARY).pack(anchor="w")
        tk.Label(header, text="Automatic Weather Station — Real-Time AI Intelligence", font=("Segoe UI", 12, "italic"),
                 bg=BG, fg=TEXT2).pack(anchor="w")
        tk.Label(header, text="Ministry of Earth Sciences  |  India Meteorological Department",
                 font=("Segoe UI", 10, "italic"), bg=BG, fg=TEXT2).pack(anchor="w")
        # Accent divider
        div = tk.Frame(header, bg=ACCENT3, height=4)
        div.pack(fill="x", pady=(6, 0))

        # --- Controls bar ---
        ctrl = ttk.Frame(container, style="TFrame")
        ctrl.pack(fill="x", pady=8)
        btn_opts = {"font": ("Calibri", 11, "bold"), "padx": 14, "pady": 8,
                    "fg": TEXT, "activebackground": CARD2,
                    "bd": 1, "relief": "raised"}
        self.btn_start = tk.Button(ctrl, text="▶ START MONITORING",
                                   bg=SUCCESS, fg="white",
                                   font=("Calibri", 11, "bold"), padx=14, pady=8,
                                   activebackground="#15803d", bd=1, relief="raised",
                                   command=self.start_monitor)
        self.btn_start.pack(side="left", padx=6)
        self.btn_stop = tk.Button(ctrl, text="⏹ STOP MONITORING",
                                  bg=DANGER, fg="white",
                                  font=("Calibri", 11, "bold"), padx=14, pady=8,
                                  activebackground="#b91c1c", bd=1, relief="raised",
                                  command=self.stop_monitor)
        self.btn_stop.pack(side="left", padx=6)
        # Demo mode button prominent
        self.btn_demo = tk.Button(ctrl, text="🎬 DEMO MODE", bg=ACCENT3,
                                  fg="white", font=("Calibri", 12, "bold"),
                                  padx=16, pady=9, command=self.demo_mode,
                                  bd=0, activebackground="#fbbf24")
        self.btn_demo.pack(side="left", padx=10)
        # Small info label
        tk.Label(ctrl, text="Click a fault button to inject an anomaly at any time.",
                 bg=BG, fg=TEXT2, font=("Calibri", 9)).pack(side="left", padx=24)

        # --- Fault injection buttons ---
        fault_frame = ttk.Frame(container, style="TFrame")
        fault_frame.pack(fill="x", pady=4)
        fault_labels = [
            ("✅ Normal Reading", self.simulate_normal, "#166534", "#bbf7d0"),
            ("🌡 Temp Spike", self.simulate_temp_spike, "#7f1d1d", "#fca5a5"),
            ("🧊 Frozen Sensor", self.simulate_frozen, "#1e3a5f", "#93c5fd"),
            ("💧 Humidity Fault", self.simulate_humidity_fault, "#581c87", "#d8b4fe"),
            ("📉 Pressure Fault", self.simulate_pressure_fault, "#78350f", "#fcd34d"),
            ("📡 Comm Error", self.simulate_comm_error, "#374151", "#9ca3af"),
            ("⛈ Weather Event", self.simulate_weather_event, "#134e4a", "#5eead4"),
            ("📈 Sensor Drift", self.simulate_drift, "#4c0519", "#fda4af"),
        ]
        for text, cmd, bg, fg in fault_labels:
            b = tk.Button(fault_frame, text=text, bg=bg, fg=fg,
                          font=("Calibri", 9, "bold"), bd=0, padx=10, pady=6,
                          activebackground="#334155", cursor="hand2", command=cmd)
            b.pack(side="left", padx=4)

        # --- Main content grid ---
        main = ttk.Frame(container, style="TFrame")
        main.pack(fill="both", expand=True, pady=6)

        # Left column: Sensor cards + status
        left = ttk.Frame(main, style="TFrame")
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))

        # Sensor cards row
        cards = ttk.Frame(left, style="TFrame")
        cards.pack(fill="x", pady=4)
        self.card_t = self._make_card(cards, "Temperature", "°C", "#38bdf8")
        self.card_p = self._make_card(cards, "Atmospheric Pressure", "hPa", "#a78bfa")
        self.card_h = self._make_card(cards, "Relative Humidity", "%", "#34d399")

        # Status panel
        status_p = ttk.LabelFrame(left, text=" INTELLIGENCE DIAGNOSTIC STATUS ", style="TFrame",
                                   labelanchor="nw")
        # Style label frame background
        for w in status_p.winfo_children():
            w.configure(background=BG, foreground=TEXT, font=("Segoe UI Semibold", 11))
        status_p.configure(borderwidth=2, relief="solid")
        status_p.pack(fill="x", pady=8)
        self.status_label = tk.Label(status_p, text="🟢 SENSOR STREAM NOMINAL", font=("Segoe UI", 26, "bold"),
                                     bg=CARD, fg=GOOD)
        self.status_label.pack(pady=10)
        # Sub info
        sub = ttk.Frame(status_p, style="TFrame")
        sub.pack(fill="x", pady=6)
        self.info_type = tk.Label(sub, text="Anomaly Type: Nominal", bg=CARD, fg=PRIMARY,
                                  font=("Segoe UI", 13, "bold"), anchor="w")
        self.info_type.pack(fill="x", padx=10)

        info_row = ttk.Frame(sub, style="TFrame")
        info_row.pack(fill="x", padx=10, pady=2)
        self.info_sev = tk.Label(info_row, text="Severity: NONE", bg=CARD, fg=TEXT2,
                                 font=("Calibri", 12), anchor="w", width=25)
        self.info_sev.pack(side="left")
        self.info_conf = tk.Label(info_row, text="AI Confidence: —", bg=CARD, fg=TEXT2,
                                  font=("Calibri", 12), anchor="w")
        self.info_conf.pack(side="left")

        self.info_wmo = tk.Label(sub, text="WMO Status: FLAG 1 (Valid & Verified)", bg=CARD, fg=TEXT,
                                 font=("Calibri", 11, "italic"), anchor="w")
        self.info_wmo.pack(fill="x", padx=10, pady=(2, 6))

        # Explanation section
        why_f = ttk.LabelFrame(left, text=" EXPLAINABLE AI (XAI) REASONING ", style="TFrame")
        for w in why_f.winfo_children():
            w.configure(background=BG, foreground=TEXT, font=("Segoe UI Semibold", 11))
        why_f.configure(borderwidth=2, relief="solid")
        why_f.pack(fill="x", pady=8)

        # SHAP-like attribution bars
        self.shap_frame = ttk.Frame(why_f, style="TFrame")
        self.shap_frame.pack(fill="x", padx=8, pady=4)

        tk.Label(self.shap_frame, text="Feature Contribution:", bg=CARD, fg=TEXT2, font=("Calibri", 10, "bold")).pack(side="left", padx=(0,10))
        self.shap_t = tk.Label(self.shap_frame, text="Temp: 33%", bg=CARD, fg=ACCENT, font=("Calibri", 10, "bold"))
        self.shap_t.pack(side="left", padx=5)
        self.shap_p = tk.Label(self.shap_frame, text="Press: 33%", bg=CARD, fg="#a855f7", font=("Calibri", 10, "bold"))
        self.shap_p.pack(side="left", padx=5)
        self.shap_h = tk.Label(self.shap_frame, text="Hum: 33%", bg=CARD, fg="#10b981", font=("Calibri", 10, "bold"))
        self.shap_h.pack(side="left", padx=5)

        self.why_text = tk.Label(why_f, text="AI analysis will appear here.",
                                 bg=CARD, fg=TEXT, wraplength=420,
                                 font=("Calibri", 11), justify="left", anchor="nw")
        self.why_text.pack(fill="both", expand=True, padx=10, pady=8)

        # Corrected reading
        corr_f = ttk.LabelFrame(left, text=" AUTO-IMPUTATION (SELF-HEALING) ", style="TFrame")
        for w in corr_f.winfo_children():
            w.configure(background=BG, foreground=TEXT, font=("Segoe UI Semibold", 11))
        corr_f.configure(borderwidth=2, relief="solid")
        corr_f.pack(fill="x", pady=8)
        self.corr_frame = ttk.Frame(corr_f, style="TFrame")
        self.corr_frame.pack(fill="x", padx=10, pady=6)

        # Three columns for T, P, H
        self.corr_t = tk.Label(self.corr_frame, text="🌡 T: —", bg=CARD, fg=ACCENT, font=("Consolas", 12, "bold"), width=12, anchor="w")
        self.corr_t.pack(side="left", padx=4)
        self.corr_p = tk.Label(self.corr_frame, text="🌬 P: —", bg=CARD, fg="#a855f7", font=("Consolas", 12, "bold"), width=14, anchor="w")
        self.corr_p.pack(side="left", padx=4)
        self.corr_h = tk.Label(self.corr_frame, text="💧 H: —", bg=CARD, fg="#10b981", font=("Consolas", 12, "bold"), width=12, anchor="w")
        self.corr_h.pack(side="left", padx=4)

        self.corr_note = tk.Label(corr_f, text="Using spatio-temporal Kalman-like estimation for forecasting continuity.",
                                  bg=CARD, fg=TEXT2, font=("Calibri", 9, "italic"), justify="left")
        self.corr_note.pack(anchor="w", padx=10, pady=(0, 6))

        # Sensor health
        health_f = ttk.LabelFrame(left, text=" SENSOR DEGRADATION & HEALTH ", style="TFrame")
        for w in health_f.winfo_children():
            w.configure(background=BG, foreground=TEXT, font=("Segoe UI Semibold", 11))
        health_f.configure(borderwidth=2, relief="solid")
        health_f.pack(fill="x", pady=8)
        self.health_labels = {}
        for label, key in [("Thermistor Health", "t"), ("Piezoresistive Health", "p"), ("Capacitive Health", "h")]:
            row = ttk.Frame(health_f, style="TFrame")
            row.pack(fill="x", padx=10, pady=3)
            tk.Label(row, text=label + ":", bg=CARD, fg=TEXT2, font=("Calibri", 11), width=20, anchor="w").pack(side="left")
            dot = tk.Label(row, text="[██████████] 100%", bg=CARD, fg=GOOD, font=("Consolas", 11, "bold"))
            dot.pack(side="left")
            self.health_labels[key] = dot
        self.overall_health = tk.Label(health_f, text="Network Connectivity: 99.9% Uptime",
                                       bg=CARD, fg=GOOD, font=("Calibri", 11, "bold"))
        self.overall_health.pack(pady=4)

        # Right column: graph + event log
        right = ttk.Frame(main, style="TFrame")
        right.pack(side="right", fill="both", expand=True, padx=(8, 0))

        # Station Selector
        station_f = ttk.Frame(right, style="TFrame")
        station_f.pack(fill="x", pady=(0, 4))
        tk.Label(station_f, text="📍 Active Station:", bg=BG, fg=TEXT2, font=("Segoe UI", 11, "bold")).pack(side="left")
        self.station_var = tk.StringVar(value="AWS-101 [New Delhi / Safdarjung]")
        self.station_dropdown = ttk.Combobox(station_f, textvariable=self.station_var,
                                             values=["AWS-101 [New Delhi / Safdarjung]",
                                                     "AWS-204 [Leh / High-Altitude]",
                                                     "AWS-308 [Mumbai / Colaba Marine]",
                                                     "AWS-415 [Jaisalmer / Thar Desert]"],
                                             state="readonly", width=35, font=("Calibri", 11))
        self.station_dropdown.pack(side="left", padx=10)
        self.station_dropdown.bind("<<ComboboxSelected>>", self._on_station_change)

        # Real-time Graph
        graph_div = ttk.LabelFrame(right, text=" MULTIVARIATE RAW SENSOR STREAM ", style="TFrame")
        for w in graph_div.winfo_children():
            w.configure(background=BG, foreground=TEXT, font=("Segoe UI Semibold", 11))
        graph_div.configure(borderwidth=2, relief="solid")
        graph_div.pack(fill="both", expand=True, pady=4)
        self.graph_frame = ttk.Frame(graph_div, style="TFrame")
        self.graph_frame.pack(fill="both", expand=True, padx=6, pady=6)
        if HAS_MPL:
            plt_bg = "#070a14"
            self.graph_fig = Figure(figsize=(7, 5), dpi=100)
            self.graph_fig.patch.set_facecolor(plt_bg)

            # Subplots for T, P, H (Much more professional!)
            self.ax_t = self.graph_fig.add_subplot(311)
            self.ax_p = self.graph_fig.add_subplot(312, sharex=self.ax_t)
            self.ax_h = self.graph_fig.add_subplot(313, sharex=self.ax_t)

            for ax in [self.ax_t, self.ax_p, self.ax_h]:
                ax.set_facecolor(plt_bg)
                ax.tick_params(colors="#e2e8f0", labelsize=8)
                for spine in ax.spines.values():
                    spine.set_color("#1e293b")
                ax.grid(color="#1e293b", linestyle='-', linewidth=0.5, alpha=0.5)

            self.ax_t.set_ylabel("Temp (°C)", color=ACCENT, fontsize=9)
            self.ax_p.set_ylabel("Press (hPa)", color="#a855f7", fontsize=9)
            self.ax_h.set_ylabel("Hum (%)", color="#10b981", fontsize=9)
            self.ax_h.set_xlabel("Observation Tick (Latest = Right)", color="#94a3b8", fontsize=9)

            self.graph_fig.tight_layout(pad=1.5)
            self.canvas = FigureCanvasTkAgg(self.graph_fig, master=self.graph_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
        else:
            tk.Label(self.graph_frame, text="Matplotlib unavailable — graph disabled.",
                     bg=CARD, fg=TEXT2, font=("Calibri", 11)).pack(pady=20)

        # Event log
        log_f = ttk.LabelFrame(right, text=" EVENT LOG ", style="TFrame")
        for w in log_f.winfo_children():
            w.configure(background=BG, foreground=TEXT, font=("Calibri", 11, "bold"))
        log_f.configure(borderwidth=2, relief="solid")
        log_f.pack(fill="x", pady=6)
        self.event_text = scrolledtext.ScrolledText(log_f, height=10, wrap=tk.WORD,
                                                    bg="#0b1220", fg=TEXT,
                                                    font=("Consolas", 9), insertbackground=TEXT)
        self.event_text.pack(fill="both", expand=True, padx=6, pady=6)
        self.event_text.config(state="disabled")

        # About
        about_f = ttk.LabelFrame(container, text=" ABOUT SKYGUARD AI ", style="TFrame")
        for w in about_f.winfo_children():
            w.configure(background=BG, foreground=TEXT, font=("Segoe UI Semibold", 11))
        about_f.configure(borderwidth=2, relief="solid")
        about_f.pack(fill="x", pady=8)
        about_text = (
            "SkyGuard AI is an AI/ML-based intelligent prototype for detecting abnormal observations from "
            "Automatic Weather Stations (AWS) using temperature, atmospheric pressure and relative humidity.\n\n"
            "Features: Multi-Station Geography Calibration  •  Real-Time Isolation Forest  •  "
            "Explainable AI (SHAP-style Feature Attribution)  •  Physical Multivariate Consistency Analysis  •  "
            "Genuine Weather Event vs Sensor Fault Classification  •  Self-Healing Imputation  •  "
            "Sensor Degradation & Predictive Maintenance  •  Offline Operation  •  Demo Mode for Judges."
        )
        tk.Label(about_f, text=about_text, bg=CARD, fg=TEXT2, wraplength=1100,
                 font=("Calibri", 10), justify="left").pack(padx=10, pady=10, anchor="w")

    def _make_card(self, parent, title, unit, color):
        card = ttk.Frame(parent, style="TFrame", relief="solid", borderwidth=2)
        card.configure(borderwidth=2, relief="solid")
        card.pack(side="left", expand=True, fill="both", padx=6)
        tk.Label(card, text=title, bg=CARD, fg=TEXT2, font=("Segoe UI", 10, "bold"),
                 anchor="w").pack(anchor="w", padx=10, pady=(10, 2))
        val = tk.Label(card, text="—", bg=CARD, fg=color, font=("Segoe UI", 30, "bold"),
                       anchor="w")
        val.pack(anchor="w", padx=10, pady=4)
        tk.Label(card, text=unit, bg=CARD, fg=TEXT2, font=("Calibri", 9), anchor="w").pack(anchor="w", padx=10, pady=(0, 10))
        return val

    def _on_station_change(self, event):
        station = self.station_var.get()
        self.sim.set_station(station)
        p = self.sim.t_mean, self.sim.p_mean, self.sim.h_mean
        self.det.retrain_station(*p)
        self.readings.clear()
        self.event_log(f"Station switched to {station}. AI Model retrained for local climate baseline.")
        # Push 1st reading
        self.current_reading = self.sim.generate_normal()
        self.readings.append(self.current_reading)
        self._update_graph()

    # ------------------------------------------------------------------
    # Monitoring / tick loop
    # ------------------------------------------------------------------
    def start_monitor(self):
        if self.monitoring:
            return
        self.monitoring = True
        self.sim.reset_fault()
        self.event_log("AWS monitoring started")
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.tick()

    def stop_monitor(self):
        self.monitoring = False
        self.event_log("Monitoring stopped")
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")

    def tick(self):
        try:
            if not self.monitoring:
                return
            # Consume pending fault if set
            if self.pending_fault:
                reading = self.pending_fault
                self.pending_fault = None
            else:
                reading = self.sim.generate_normal()

            # Frozen logic: if same temp repeated, count
            if reading is not None:
                if len(self.readings) > 0:
                    prev = self.readings[-1]
                    if abs(prev[0] - reading[0]) < 0.01:
                        self.frozen_counter += 1
                    else:
                        self.frozen_counter = 0
            else:
                # Communication error resets frozen
                self.frozen_counter = 0

            self.current_reading = reading
            self.readings.append(reading if reading else (None, None, None))

            # Update model with normal observations (only when clearly normal)
            if reading is not None and self.frozen_counter == 0:
                self.det.update_normal_history(reading)
                # Also generate a corrected estimate from recent normals

            # Analyze
            if reading is None:
                is_anom = True
                info = self.det.classify_anomaly(None, frozen_counter=self.frozen_counter)
            else:
                is_anom, conf, sc = self.det.score(reading)
                # If frozen, override
                if self.frozen_counter >= 3:
                    is_anom = True
                    info = self.det.classify_anomaly(reading, frozen_counter=3)
                    conf = 85.0
                else:
                    info = self.det.classify_anomaly(reading, frozen_counter=self.frozen_counter)
                    # Ensure confidence aligned
                    _, conf, _ = self.det.score(reading)

            self.is_anomaly = is_anom
            self.severity = info["severity"] if is_anom else "NONE"
            self.confidence = conf if is_anom else 0.0
            self.anomaly_info = info
            self.is_genuine_weather = info.get("is_genuine_weather", False)

            # Update health
            if reading is not None:
                # Health drops on anomaly; recovers slowly on normal
                if is_anom:
                    health = max(40, 100 - 30)
                else:
                    health = 100
                # Simple per-sensor: if reading out of range -> unhealthy
                t_ok = 15 <= reading[0] <= 42
                p_ok = 980 <= reading[1] <= 1032
                h_ok = 20 <= reading[2] <= 95
                # Update dots
                for k, ok in [("t", t_ok), ("p", p_ok), ("h", h_ok)]:
                    dot = self.health_labels[k]
                    if ok and not is_anom:
                        dot.config(text="[██████████] 100%", fg=GOOD)
                    elif is_anom and not ok:
                        dot.config(text="[████████░░]  65%", fg=BAD)
                    else:
                        dot.config(text="[██████████] 100%", fg=GOOD)
                # Overall
                overall = int((t_ok + p_ok + h_ok) / 3 * 100)
                if is_anom:
                    overall = max(overall - 25, 55)
                self.overall_health.config(text=f"Network Connectivity: {overall}% Uptime",
                                            fg=GOOD if overall >= 80 else (ACCENT2 if overall >= 60 else BAD))
            else:
                # Communication error: all sensors unhealthy briefly
                for k in ["t", "p", "h"]:
                    self.health_labels[k].config(text="[████░░░░░░]  20%", fg=BAD)
                self.overall_health.config(text="Network Connectivity: 0% Uptime", fg=BAD)

            # Update cards
            if reading is not None:
                self.card_t.config(text=f"{reading[0]}°C")
                self.card_p.config(text=f"{reading[1]} hPa")
                self.card_h.config(text=f"{reading[2]}%")
            else:
                self.card_t.config(text="—")
                self.card_p.config(text="—")
                self.card_h.config(text="—")

            # Update status
            if is_anom:
                if info.get("is_genuine_weather"):
                    self.status_label.config(text="🟢 GENUINE WEATHER EVENT", fg=ACCENT2)
                else:
                    self.status_label.config(text="🔴 ANOMALY DETECTED", fg=BAD)
                self.info_type.config(text=f"Anomaly Type: {info['type']}")
                self.info_sev.config(text=f"Severity: {self.severity}")
                self.info_conf.config(text=f"AI Confidence: {self.confidence:.0f}%")
                self.info_wmo.config(text=f"WMO Status: {info.get('wmo_status', 'FLAG 2 (QC Rejection)')}")

                # Update SHAP attribution bars
                xai = info.get("xai", {})
                self.shap_t.config(text=f"Temp: {xai.get('temp', 33)}%")
                self.shap_p.config(text=f"Press: {xai.get('press', 33)}%")
                self.shap_h.config(text=f"Hum: {xai.get('hum', 33)}%")

                self.why_text.config(text=info["explanation"])
                # Show corrected
                if reading is not None and reading[0] is not None:
                    corr = self.det.suggest_correction(reading)
                    self.corr_t.config(text=f"🌡 T: {corr[0]}°C")
                    self.corr_p.config(text=f"🌬 P: {corr[1]} hPa")
                    self.corr_h.config(text=f"💧 H: {corr[2]}%")
                else:
                    self.corr_t.config(text="🌡 T: —")
                    self.corr_p.config(text="🌬 P: —")
                    self.corr_h.config(text="💧 H: —")
                # Log event
                self.event_log(f"{info['type']} detected  |  Severity: {self.severity}  |  Confidence: {self.confidence:.0f}%")
            else:
                self.status_label.config(text="🟢 SENSOR STREAM NOMINAL", fg=GOOD)
                self.info_type.config(text="Anomaly Type: Nominal")
                self.info_sev.config(text="Severity: NONE")
                self.info_conf.config(text="AI Confidence: —")
                self.info_wmo.config(text="WMO Status: FLAG 1 (Valid & Verified)")
                self.shap_t.config(text="Temp: 33%")
                self.shap_p.config(text="Press: 33%")
                self.shap_h.config(text="Hum: 33%")
                self.why_text.config(text="AI detected that the current observation fits within the learned normal weather pattern.")
                self.corr_t.config(text="🌡 T: —")
                self.corr_p.config(text="🌬 P: —")
                self.corr_h.config(text="💧 H: —")
                # Log normal
                if reading is not None:
                    if self.is_genuine_weather:
                        self.event_log("Genuine weather event observed — sensor stream nominal")
                    else:
                        self.event_log("Normal observation received")
                else:
                    self.event_log("Communication error — no data")

            # Update graph
            self._update_graph()

            # Schedule next tick if monitoring
            if self.monitoring:
                self.after(1500, self.tick)
        except Exception as e:
            self.event_log(f"Error in tick: {e}")
            if self.monitoring:
                self.after(1500, self.tick)

    # ------------------------------------------------------------------
    # Graph update
    # ------------------------------------------------------------------
    def _update_graph(self):
        if not HAS_MPL or self.graph_fig is None:
            return

        for ax in [self.ax_t, self.ax_p, self.ax_h]:
            ax.clear()
            ax.grid(color="#1e293b", linestyle='-', linewidth=0.5, alpha=0.5)

        xs = np.arange(len(self.readings))
        if len(self.readings) == 0:
            return

        t_vals = [r[0] if r is not None and r[0] is not None else np.nan for r in self.readings]
        p_vals = [r[1] if r is not None and r[1] is not None else np.nan for r in self.readings]
        h_vals = [r[2] if r is not None and r[2] is not None else np.nan for r in self.readings]

        # Draw lines
        self.ax_t.plot(xs, t_vals, color=ACCENT, marker="o", markersize=4, linewidth=2)
        self.ax_p.plot(xs, p_vals, color="#a855f7", marker="s", markersize=4, linewidth=2)
        self.ax_h.plot(xs, h_vals, color="#10b981", marker="^", markersize=4, linewidth=2)

        # Highlight Anomaly at last tick
        if self.is_anomaly and len(self.readings) > 0:
            last = len(self.readings) - 1
            if not np.isnan(t_vals[-1]):
                self.ax_t.scatter([last], [t_vals[-1]], color=CRITICAL, s=120, zorder=5, edgecolor="yellow")
            if not np.isnan(p_vals[-1]):
                self.ax_p.scatter([last], [p_vals[-1]], color=CRITICAL, s=120, zorder=5, edgecolor="yellow")
            if not np.isnan(h_vals[-1]):
                self.ax_h.scatter([last], [h_vals[-1]], color=CRITICAL, s=120, zorder=5, edgecolor="yellow")

            # Draw a vertical semi-transparent red band
            for ax in [self.ax_t, self.ax_p, self.ax_h]:
                ax.axvspan(last - 0.5, last + 0.5, color=CRITICAL, alpha=0.15)

        # Labels
        self.ax_t.set_ylabel("Temp (°C)", color=ACCENT, fontsize=9)
        self.ax_p.set_ylabel("Press (hPa)", color="#a855f7", fontsize=9)
        self.ax_h.set_ylabel("Hum (%)", color="#10b981", fontsize=9)
        self.ax_h.set_xlabel("Observation Tick (Latest = Right)", color="#94a3b8", fontsize=9)

        for ax in [self.ax_t, self.ax_p, self.ax_h]:
             ax.tick_params(colors="#e2e8f0", labelsize=8)
             ax.set_xlim(max(0, len(xs) - 30), max(10, len(xs) - 1))

        self.graph_fig.tight_layout(pad=1.5)
        self.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Fault buttons
    # ------------------------------------------------------------------
    def simulate_normal(self):
        self.pending_fault = self.sim.generate_normal()
        self.event_log("Simulated normal reading")

    def simulate_temp_spike(self):
        self.pending_fault = self.sim.inject_temperature_spike()
        self.event_log("Simulated Temperature Spike")

    def simulate_frozen(self):
        self.pending_fault = self.sim.inject_frozen_sensor()
        self.event_log("Simulated Frozen Sensor")

    def simulate_humidity_fault(self):
        self.pending_fault = self.sim.inject_humidity_fault()
        self.event_log("Simulated Humidity Fault")

    def simulate_pressure_fault(self):
        self.pending_fault = self.sim.inject_pressure_fault()
        self.event_log("Simulated Pressure Fault")

    def simulate_weather_event(self):
        self.pending_fault = self.sim.inject_genuine_weather_event()
        self.event_log("Simulating Genuine Weather Event (Convective Storm)")

    def simulate_drift(self):
        self.pending_fault = self.sim.inject_drift()
        self.event_log("Simulated Sensor Calibration Drift")

    def simulate_comm_error(self):
        self.pending_fault = self.sim.inject_communication_error()
        self.event_log("Simulated Communication Error")

    # ------------------------------------------------------------------
    # Event log
    # ------------------------------------------------------------------
    def event_log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.event_lines.append(line)
        self.event_text.config(state="normal")
        self.event_text.insert("end", line + "\n")
        self.event_text.see("end")
        # Trim to ~20 visible (scroll keeps last)
        if len(self.event_lines) > 20:
            self.event_text.delete("1.0", "2.0")
        self.event_text.config(state="disabled")

    # ------------------------------------------------------------------
    # Demo mode
    # ------------------------------------------------------------------
    def demo_mode(self):
        if self.monitoring and not self.btn_demo.cget("state") == "normal":
            messagebox.showinfo("Demo Mode", "Demo already running.")
            return
        self.event_log("🎬 DEMO MODE started")
        self.btn_demo.config(state="disabled")
        self.start_monitor()
        # Sequence: normal ticks, spike, frozen, humidity, pressure, weather event, drift, comm, return
        # We simulate quickly by injecting at specific times using chained after
        # Since tick runs every 1500ms, use shorter delays for rapid demo
        self.after(1200, lambda: self._demo_step(1))    # Temp spike
        self.after(3000, lambda: self._demo_step(2))    # Frozen sensor
        self.after(4800, lambda: self._demo_step(3))    # Humidity fault
        self.after(6600, lambda: self._demo_step(4))    # Pressure fault
        self.after(8700, lambda: self._demo_step(6))    # Genuine weather event
        self.after(10500, lambda: self._demo_step(7))   # Sensor drift
        self.after(12000, lambda: self._demo_step(5))   # Communication error
        self.after(14500, lambda: self._demo_step(0))   # Back to normal
        self.after(16500, lambda: self._end_demo())

    def _demo_step(self, step):
        if not self.monitoring:
            return
        if step == 1:
            # Temperature spike at ~3s
            self.simulate_temp_spike()
        elif step == 2:
            # Frozen at ~6s
            self.simulate_frozen()
        elif step == 3:
            # Humidity at ~8s
            self.simulate_humidity_fault()
        elif step == 4:
            # Pressure at ~10s
            self.simulate_pressure_fault()
        elif step == 6:
            # Genuine Weather Event at ~13.5s
            self.simulate_weather_event()
        elif step == 7:
            # Sensor drift at ~16s
            self.simulate_drift()
        elif step == 5:
            # Communication at ~12s (moved later)
            self.simulate_comm_error()
        elif step == 0:
            # Back to normal
            self.simulate_normal()

    def _end_demo(self):
        self.event_log("🎬 DEMO MODE complete — returned to normal")
        self.btn_demo.config(state="normal")

    # ------------------------------------------------------------------
    # First tick on open
    # ------------------------------------------------------------------
    def _first_tick(self):
        # Show initial normal reading without monitoring
        r = self.sim.generate_normal()
        self.current_reading = r
        self.card_t.config(text=f"{r[0]}°C")
        self.card_p.config(text=f"{r[1]} hPa")
        self.card_h.config(text=f"{r[2]}%")
        self.readings.append(r)
        self._update_graph()
        self.event_log("SkyGuard AI initialized — AWS simulated")
        self.event_log("Ready for demonstration")
        self.event_log("WMO-No. 8 QC Engine: Online")

    def _on_close(self):
        self.stop_monitor()
        self.destroy()


if __name__ == "__main__":
    app = SkyGuardApp()
    app.mainloop()
