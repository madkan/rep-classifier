#!/usr/bin/env python3
"""
Real-time IMU plotter + bicep curl near-failure monitor.

Requires real_time_predictor_bicep_curl.py and bicep_classifier.pkl
in the same directory.

Keys:
  r  -> start recording / stop recording + save
  q  -> quit

Examples:
  python plot_imu_sets.py
  python plot_imu_sets.py --port /dev/tty.usbserial-11130
  python plot_imu_sets.py --csv data/bicep_curl_set_01.csv
"""

import argparse
import os
import queue
import re
import sys
import threading
import time
from collections import deque

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import serial
import serial.tools.list_ports

from real_time_predictor_bicep_curl import (
    parse_serial_line,
    get_completed_reps_from_buffer,
    classify_rep,
    should_skip_small_range_rep,
    load_model_bundle,
    smooth_signal,
    SMOOTH_WINDOW,
    BASELINE_REPS,
    MIN_TIME_BETWEEN_CLASSIFIED_REPS,
)

# ── config ────────────────────────────────────────────────────────────────────
BAUD_RATE       = 115200
WINDOW_SEC      = 12
SAMPLE_HZ       = 100
DATA_DIR        = "data"
NEAR_FAIL_FLASH = 3.0

LINE_RE = re.compile(
    r"AX:(?P<ax>[-\d.]+)\s+AY:(?P<ay>[-\d.]+)\s+AZ:(?P<az>[-\d.]+)"
    r"\s*\|\s*"
    r"GX:(?P<gx>[-\d.]+)\s+GY:(?P<gy>[-\d.]+)\s+GZ:(?P<gz>[-\d.]+)"
    r"\s*\|\s*"
    r"T:(?P<t>[-\d.]+)"
)

C = dict(
    bg      = "#0d1117",
    panel   = "#161b22",
    border  = "#30363d",
    green   = "#3fb950",
    red     = "#f85149",
    orange  = "#e3b341",
    grey    = "#484f58",
    text    = "#e6edf3",
    sub     = "#8b949e",
    raw     = "#2d333b",
    smooth  = "#58a6ff",
    rec     = "#f85149",
    ax_c    = "#4fc3f7",
    ay_c    = "#81d4fa",
    az_c    = "#b3e5fc",
    gx_c    = "#f48fb1",
    gy_c    = "#f06292",
    gz_c    = "#e91e63",
)

plt.rcParams.update({
    "figure.facecolor": C["bg"],
    "axes.facecolor":   C["panel"],
    "axes.edgecolor":   C["border"],
    "axes.labelcolor":  C["text"],
    "xtick.color":      C["sub"],
    "ytick.color":      C["sub"],
    "grid.color":       C["border"],
    "grid.alpha":       0.4,
    "text.color":       C["text"],
    "font.family":      "monospace",
})


# ── helpers ───────────────────────────────────────────────────────────────────
def find_port() -> str:
    ports = serial.tools.list_ports.comports()
    usb = [p for p in ports if "usb" in p.device.lower()]
    if usb:
        return usb[0].device
    if ports:
        return ports[0].device
    print("[ERROR] No serial ports found.", file=sys.stderr)
    sys.exit(1)


def next_set_filename(exercise: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    prefix, suffix = f"{exercise}_set_", ".csv"
    existing = []
    for name in os.listdir(DATA_DIR):
        if name.startswith(prefix) and name.endswith(suffix):
            mid = name[len(prefix):-len(suffix)]
            if mid.isdigit():
                existing.append(int(mid))
    n = 1 if not existing else max(existing) + 1
    return os.path.join(DATA_DIR, f"{exercise}_set_{n:02d}.csv")


def _style(ax, ylabel):
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, linewidth=0.5)
    ax.tick_params(labelsize=7)


# ── shared state ──────────────────────────────────────────────────────────────
_lock = threading.Lock()
_BUF  = int(WINDOW_SEC * SAMPLE_HZ * 2)

_st = dict(
    status    = "Connecting...",
    connected = False,

    # rolling IMU deques — always updated regardless of recording state
    t    = deque(maxlen=_BUF),
    ax_d = deque(maxlen=_BUF),
    ay_d = deque(maxlen=_BUF),
    az_d = deque(maxlen=_BUF),
    gx_d = deque(maxlen=_BUF),
    gy_d = deque(maxlen=_BUF),
    gz_d = deque(maxlen=_BUF),

    # classifier state — reset each time recording starts
    recording           = False,
    record_fp           = None,
    record_path         = None,
    set_count           = 0,
    buffer              = [],
    detected_reps       = [],
    baseline_features   = [],
    rep_count           = 0,
    last_classified_end = None,

    # near-failure display
    near_failure_active = False,
    near_failure_ts     = 0.0,
    last_prediction     = "—",
    last_probability    = None,
)

# classifier only gets samples while recording
data_q: queue.Queue = queue.Queue()


# ── serial reader ─────────────────────────────────────────────────────────────
def serial_reader(port, baud, stop_ev):
    try:
        with serial.Serial(port, baud, timeout=1) as ser:
            with _lock:
                _st["connected"] = True
                _st["status"]    = f"Connected  {port}  @{baud} baud"
            t0 = time.perf_counter()
            while not stop_ev.is_set():
                raw = ser.readline()
                try:
                    line = raw.decode("utf-8", errors="replace").strip()
                except Exception:
                    continue

                sample = parse_serial_line(line)
                if sample is None:
                    m = LINE_RE.search(line)
                    if m:
                        now    = time.perf_counter() - t0
                        sample = dict(
                            t_us   = int(now * 1_000_000),
                            ax_g   = float(m.group("ax")),
                            ay_g   = float(m.group("ay")),
                            az_g   = float(m.group("az")),
                            gx_dps = float(m.group("gx")),
                            gy_dps = float(m.group("gy")),
                            gz_dps = float(m.group("gz")),
                            time_s = now,
                        )

                if sample is not None:
                    with _lock:
                        _st["t"].append(sample["time_s"])
                        _st["ax_d"].append(sample["ax_g"])
                        _st["ay_d"].append(sample["ay_g"])
                        _st["az_d"].append(sample["az_g"])
                        _st["gx_d"].append(sample["gx_dps"])
                        _st["gy_d"].append(sample["gy_dps"])
                        _st["gz_d"].append(sample["gz_dps"])
                        is_rec = _st["recording"]
                    if is_rec:
                        data_q.put(sample)

    except serial.SerialException as e:
        with _lock:
            _st["connected"] = False
            _st["status"]    = f"Disconnected — {e}"


# ── csv replay reader ─────────────────────────────────────────────────────────
def csv_reader(csv_path, delay, stop_ev):
    raw = pd.read_csv(csv_path)
    raw["time_s"] = raw["t_us"] / 1_000_000.0
    with _lock:
        _st["status"] = f"Replaying  {csv_path}"
    for _, row in raw.iterrows():
        if stop_ev.is_set():
            break
        sample = {k: float(row[k]) for k in
                  ["t_us","ax_g","ay_g","az_g","gx_dps","gy_dps","gz_dps","time_s"]}
        with _lock:
            _st["t"].append(sample["time_s"])
            _st["ax_d"].append(sample["ax_g"])
            _st["ay_d"].append(sample["ay_g"])
            _st["az_d"].append(sample["az_g"])
            _st["gx_d"].append(sample["gx_dps"])
            _st["gy_d"].append(sample["gy_dps"])
            _st["gz_d"].append(sample["gz_dps"])
            is_rec = _st["recording"]
        if is_rec:
            data_q.put(sample)
        time.sleep(delay)


# ── classifier thread ─────────────────────────────────────────────────────────
def classifier_thread(model, feature_columns, threshold, stop_ev):
    while not stop_ev.is_set():
        try:
            sample = data_q.get(timeout=0.1)
        except queue.Empty:
            continue

        with _lock:
            if _st["recording"] and _st["record_fp"]:
                _st["record_fp"].write(
                    f"{sample['t_us']:.0f},"
                    f"{sample['ax_g']:.6f},{sample['ay_g']:.6f},{sample['az_g']:.6f},"
                    f"{sample['gx_dps']:.6f},{sample['gy_dps']:.6f},{sample['gz_dps']:.6f}\n"
                )
            _st["buffer"].append(sample)
            buf_snap = list(_st["buffer"])
            last_cet = _st["last_classified_end"]
            baseline = list(_st["baseline_features"])

        buf_df    = pd.DataFrame(buf_snap)
        completed = get_completed_reps_from_buffer(buf_df, last_cet)

        for rep_info in completed:
            end_time = rep_info["end_time"]
            with _lock:
                last_cet2 = _st["last_classified_end"]
            if last_cet2 and end_time <= last_cet2 + MIN_TIME_BETWEEN_CLASSIFIED_REPS:
                continue

            skip, rr = should_skip_small_range_rep(rep_info["rep_df"], baseline)
            if skip:
                with _lock:
                    _st["detected_reps"].append(dict(
                        prediction="skipped", probability=None, rep_num=None,
                        start_time=rep_info["start_time"], end_time=end_time,
                        peak_time=buf_df["time_s"].iloc[rep_info["peak_idx"]],
                    ))
                    _st["last_classified_end"] = end_time
                    # keep a small margin before rep start
                    KEEP_BEFORE_SEC = 0.5
                    cutoff = rep_info["start_time"] - KEEP_BEFORE_SEC
                    _st["buffer"] = [s for s in _st["buffer"]if s["time_s"] > cutoff]
                continue

            with _lock:
                _st["rep_count"] += 1
                rc   = _st["rep_count"]
                base = list(_st["baseline_features"])

            raw_feat, pred, proba = classify_rep(
                rep_df=rep_info["rep_df"],
                rep_count=rc,
                baseline_features=base,
                model=model,
                feature_columns=feature_columns,
                threshold=threshold,
            )
            label = "NEAR FAILURE" if pred == 1 else "normal"

            with _lock:
                if len(_st["baseline_features"]) < BASELINE_REPS:
                    _st["baseline_features"].append(raw_feat)
                _st["detected_reps"].append(dict(
                    prediction=label, probability=proba, rep_num=rc,
                    start_time=rep_info["start_time"], end_time=end_time,
                    peak_time=buf_df["time_s"].iloc[rep_info["peak_idx"]],
                ))
                _st["last_classified_end"] = end_time
                # keep a small margin before rep start
                KEEP_BEFORE_SEC = 0.5
                cutoff = rep_info["start_time"] - KEEP_BEFORE_SEC
                _st["buffer"] = [s for s in _st["buffer"]if s["time_s"] > cutoff]

                _st["last_prediction"]     = label
                _st["last_probability"]    = proba
                if pred == 1:
                    _st["near_failure_active"] = True
                    _st["near_failure_ts"]     = time.time()


# ── GUI ───────────────────────────────────────────────────────────────────────
class GUI:
    def __init__(self, exercise: str, window: float):
        self.exercise = exercise
        self.window   = window

        self.fig = plt.figure(figsize=(15, 9))
        self.fig.patch.set_facecolor(C["bg"])
        self.fig.canvas.manager.set_window_title(f"Bicep Curl Monitor  ·  {exercise}")

        gs = gridspec.GridSpec(
            4, 3,
            figure=self.fig,
            left=0.07, right=0.97,
            top=0.90,  bottom=0.09,
            wspace=0.40, hspace=0.65,
        )

        self.ax_sig  = self.fig.add_subplot(gs[0, 0:2])
        self.ax_acc  = self.fig.add_subplot(gs[1, 0:2])
        self.ax_gyro = self.fig.add_subplot(gs[2, 0:2])
        self.ax_prob = self.fig.add_subplot(gs[3, 0:2])
        self.ax_stat = self.fig.add_subplot(gs[:, 2])

        # ── ay_g signal ───────────────────────────────────────────────────────
        self.ax_sig.set_title("ay_g  (curl axis)", fontsize=8, color=C["text"], pad=4)
        _style(self.ax_sig, "ay_g")
        self.ln_raw,    = self.ax_sig.plot([], [], color=C["raw"],    lw=1,   alpha=0.5, label="raw")
        self.ln_smooth, = self.ax_sig.plot([], [], color=C["smooth"], lw=1.8, label="smooth")
        self.ax_sig.legend(fontsize=7, loc="upper left",
                           facecolor=C["panel"], edgecolor=C["border"], labelcolor=C["text"])

        # ── accel ─────────────────────────────────────────────────────────────
        self.ax_acc.set_title("Accelerometer", fontsize=8, color=C["text"], pad=4)
        _style(self.ax_acc, "g")
        self.ln_ax, = self.ax_acc.plot([], [], color=C["ax_c"], lw=1, label="X")
        self.ln_ay, = self.ax_acc.plot([], [], color=C["ay_c"], lw=1, label="Y")
        self.ln_az, = self.ax_acc.plot([], [], color=C["az_c"], lw=1, label="Z")
        self.ax_acc.legend(fontsize=7, loc="upper left",
                           facecolor=C["panel"], edgecolor=C["border"], labelcolor=C["text"])

        # ── gyro ──────────────────────────────────────────────────────────────
        self.ax_gyro.set_title("Gyroscope", fontsize=8, color=C["text"], pad=4)
        _style(self.ax_gyro, "°/s")
        self.ln_gx, = self.ax_gyro.plot([], [], color=C["gx_c"], lw=1, label="X")
        self.ln_gy, = self.ax_gyro.plot([], [], color=C["gy_c"], lw=1, label="Y")
        self.ln_gz, = self.ax_gyro.plot([], [], color=C["gz_c"], lw=1, label="Z")
        self.ax_gyro.legend(fontsize=7, loc="upper left",
                            facecolor=C["panel"], edgecolor=C["border"], labelcolor=C["text"])

        # ── probability bar ───────────────────────────────────────────────────
        self.ax_prob.set_title("Near-Failure Probability  (last rep)", fontsize=8, color=C["text"], pad=4)
        self.ax_prob.set_xlim(0, 1)
        self.ax_prob.set_ylim(-0.6, 0.6)
        self.ax_prob.set_yticks([])
        self.ax_prob.tick_params(labelsize=7)
        self.ax_prob.grid(axis="x")
        self._pbar = self.ax_prob.barh([0], [0], height=0.5, color=C["green"], align="center")
        self.ax_prob.axvline(0.5, color=C["orange"], lw=1.2, ls="--", alpha=0.9)
        # label inside the axes area (y=0.5 = vertical center)
        self._ptxt = self.ax_prob.text(
            0.5, 0.5, "—",
            ha="center", va="center", fontsize=8,
            color=C["sub"], transform=self.ax_prob.transAxes,
        )

        # ── stats panel ───────────────────────────────────────────────────────
        self.ax_stat.set_facecolor(C["panel"])
        self.ax_stat.axis("off")
        self.ax_stat.set_xlim(0, 1)
        self.ax_stat.set_ylim(0, 1)

        def _row(y, label, init_val, val_color=C["text"]):
            self.ax_stat.text(0.08, y, label, ha="left", va="center",
                              fontsize=8, color=C["sub"],
                              transform=self.ax_stat.transAxes)
            t = self.ax_stat.text(0.92, y, init_val, ha="right", va="center",
                                  fontsize=16, fontweight="bold", color=val_color,
                                  transform=self.ax_stat.transAxes)
            return t

        self.ax_stat.text(0.5, 0.95, "SESSION", ha="center", va="top",
                          fontsize=9, color=C["sub"],
                          transform=self.ax_stat.transAxes)
        self.ax_stat.plot([0.05, 0.95], [0.89, 0.89],
                         color=C["border"], lw=0.8,
                         transform=self.ax_stat.transAxes, clip_on=False)

        self._t_sets     = _row(0.80, "Sets saved",    "0",  C["text"])
        self._t_reps     = _row(0.65, "Reps (set)",    "—",  C["text"])
        self._t_baseline = _row(0.50, "Baseline",      "—",  C["orange"])

        self.ax_stat.plot([0.05, 0.95], [0.42, 0.42],
                         color=C["border"], lw=0.8,
                         transform=self.ax_stat.transAxes, clip_on=False)
        self.ax_stat.text(0.5, 0.38, "LAST REP", ha="center", va="top",
                          fontsize=9, color=C["sub"],
                          transform=self.ax_stat.transAxes)

        self._t_pred     = _row(0.27, "Prediction",    "—",  C["sub"])
        self._t_prob     = _row(0.13, "Probability",   "—",  C["sub"])

        # # ── near-failure overlay ──────────────────────────────────────────────
        # self._flash = FancyBboxPatch(
        #     (0, 0), 1, 1,
        #     boxstyle="round,pad=0",
        #     transform=self.fig.transFigure,
        #     facecolor=C["red"], alpha=0,
        #     zorder=20, clip_on=False,
        # )
        # self.fig.add_artist(self._flash)

        # ── header ────────────────────────────────────────────────────────────
        self._hdr = self.fig.text(
            0.5, 0.965,
            f"BICEP CURL MONITOR  ·  {exercise.upper()}  ·  press R to start recording",
            ha="center", va="top", fontsize=9,
            color=C["sub"], transform=self.fig.transFigure,
        )
        self._rec_dot = self.fig.text(
            0.015, 0.968, "●", ha="left", va="top",
            fontsize=13, color=C["grey"], transform=self.fig.transFigure,
        )

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ── key handling ──────────────────────────────────────────────────────────
    def _on_key(self, event):
        k = (event.key or "").lower()

        if k == "r":
            with _lock:
                was = _st["recording"]

            if not was:
                fp_path = next_set_filename(self.exercise)
                fp      = open(fp_path, "w", encoding="utf-8", buffering=1)
                fp.write("t_us,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps\n")
                # drain stale queued samples
                while not data_q.empty():
                    try:
                        data_q.get_nowait()
                    except queue.Empty:
                        break
                with _lock:
                    _st["recording"]           = True
                    _st["record_fp"]           = fp
                    _st["record_path"]         = fp_path
                    _st["buffer"]              = []
                    _st["detected_reps"]       = []
                    _st["baseline_features"]   = []
                    _st["rep_count"]           = 0
                    _st["last_classified_end"] = None
                    _st["near_failure_active"] = False
                    _st["last_prediction"]     = "—"
                    _st["last_probability"]    = None
                print(f"[REC] started → {fp_path}")
                self._set_hdr(f"● RECORDING  →  {fp_path}", C["rec"])
            else:
                with _lock:
                    fp  = _st["record_fp"]
                    p   = _st["record_path"]
                    _st["recording"]   = False
                    _st["record_fp"]   = None
                    _st["record_path"] = None
                    _st["set_count"]  += 1
                if fp:
                    fp.close()
                print(f"[REC] saved → {p}")
                self._set_hdr(f"Saved  →  {p}  ·  press R for next set", C["green"])

        elif k == "q":
            plt.close("all")

    def _set_hdr(self, msg, color=C["sub"]):
        self._hdr.set_text(msg)
        self._hdr.set_color(color)

    # ── animation tick ────────────────────────────────────────────────────────
    def update(self, _frame):
        with _lock:
            t_d    = list(_st["t"])
            ax_d   = list(_st["ax_d"])
            ay_d   = list(_st["ay_d"])
            az_d   = list(_st["az_d"])
            gx_d   = list(_st["gx_d"])
            gy_d   = list(_st["gy_d"])
            gz_d   = list(_st["gz_d"])
            buf    = list(_st["buffer"])
            det    = list(_st["detected_reps"])
            rec    = _st["recording"]
            status = _st["status"]
            nf_act = _st["near_failure_active"]
            nf_ts  = _st["near_failure_ts"]
            base_n = len(_st["baseline_features"])
            sets   = _st["set_count"]
            last_p = _st["last_prediction"]
            last_pb= _st["last_probability"]

        # ── rec dot ───────────────────────────────────────────────────────────
        self._rec_dot.set_color(C["rec"] if rec else C["grey"])
        self._rec_dot.set_text("● REC" if rec else "●")

        # # ── near-failure flash ────────────────────────────────────────────────
        # age = time.time() - nf_ts
        # if nf_act and age < NEAR_FAIL_FLASH:
        #     self._flash.set_alpha(0.18 * max(0.0, 1.0 - age / NEAR_FAIL_FLASH))
        # else:
        #     self._flash.set_alpha(0)
        #     if nf_act and age >= NEAR_FAIL_FLASH:
        #         with _lock:
        #             _st["near_failure_active"] = False

        # ── idle header ───────────────────────────────────────────────────────
        if not rec:
            self._set_hdr(
                f"BICEP CURL MONITOR  ·  {self.exercise.upper()}  ·  {status}  ·  press R to record",
                C["sub"],
            )

        # ── rolling window ────────────────────────────────────────────────────
        if len(t_d) < 2:
            return

        t_now = t_d[-1]
        t_lo  = t_now - self.window

        def trim(xs):
            return [x for x, ti in zip(xs, t_d) if ti >= t_lo]

        tv  = [ti for ti in t_d if ti >= t_lo]
        axv = trim(ax_d); ayv = trim(ay_d); azv = trim(az_d)
        gxv = trim(gx_d); gyv = trim(gy_d); gzv = trim(gz_d)

        def xlims(a):
            a.set_xlim(t_lo, t_now + 0.1)

        def auto_y(a, *series):
            vals = [v for s in series for v in s]
            if vals:
                lo, hi = min(vals), max(vals)
                pad = max((hi - lo) * 0.15, 0.05)
                a.set_ylim(lo - pad, hi + pad)

        # ── accel / gyro ──────────────────────────────────────────────────────
        self.ln_ax.set_data(tv, axv); self.ln_ay.set_data(tv, ayv); self.ln_az.set_data(tv, azv)
        self.ln_gx.set_data(tv, gxv); self.ln_gy.set_data(tv, gyv); self.ln_gz.set_data(tv, gzv)
        xlims(self.ax_acc); xlims(self.ax_gyro)
        auto_y(self.ax_acc,  axv, ayv, azv)
        auto_y(self.ax_gyro, gxv, gyv, gzv)

        # ── ay_g curl signal ──────────────────────────────────────────────────
        if buf:
            df   = pd.DataFrame(buf)
            t_s  = df["time_s"].values
            ay   = df["ay_g"].values
            sm   = smooth_signal(ay, SMOOTH_WINDOW)
            mask = t_s >= t_lo
            self.ln_raw.set_data(t_s[mask], ay[mask])
            self.ln_smooth.set_data(t_s[mask], sm[mask])
            xlims(self.ax_sig)
            if mask.any():
                pad = max(0.4, (ay[mask].max() - ay[mask].min()) * 0.15)
                self.ax_sig.set_ylim(ay[mask].min() - pad, ay[mask].max() + pad)
            for coll in list(self.ax_sig.collections):
                coll.remove()
            for rep in det:
                rs, re = rep["start_time"], rep["end_time"]
                if re < t_lo or rs > t_now:
                    continue
                col = (C["red"]  if rep["prediction"] == "NEAR FAILURE" else
                       C["grey"] if rep["prediction"] == "skipped"      else
                       C["orange"])
                self.ax_sig.axvspan(rs, re, alpha=0.18, color=col, lw=0)
        else:
            # not recording — show raw ay_g from rolling deque
            self.ln_raw.set_data(tv, ayv)
            self.ln_smooth.set_data([], [])
            xlims(self.ax_sig)
            auto_y(self.ax_sig, ayv)

        # ── probability bar ───────────────────────────────────────────────────
        if last_pb is not None:
            col = C["red"] if last_pb >= 0.5 else C["green"]
            self._pbar[0].set_width(last_pb)
            self._pbar[0].set_facecolor(col)
            self._ptxt.set_text(f"{last_pb:.3f}  ({last_p})")
            self._ptxt.set_color(col)

        # ── stats panel ───────────────────────────────────────────────────────
        counted = len([r for r in det if r["prediction"] != "skipped"])

        self._t_sets.set_text(str(sets))

        if rec:
            self._t_reps.set_text(str(counted) if counted > 0 else "—")
            self._t_reps.set_color(C["text"])
            if base_n < BASELINE_REPS:
                self._t_baseline.set_text(f"{base_n}/{BASELINE_REPS}")
                self._t_baseline.set_color(C["orange"])
            else:
                self._t_baseline.set_text("ready")
                self._t_baseline.set_color(C["green"])
        else:
            self._t_reps.set_text("—")
            self._t_reps.set_color(C["sub"])
            self._t_baseline.set_text("—")
            self._t_baseline.set_color(C["sub"])

        if last_p == "NEAR FAILURE":
            self._t_pred.set_text("⚠ NEAR FAIL")
            self._t_pred.set_color(C["red"])
        elif last_p == "normal":
            self._t_pred.set_text("✓ normal")
            self._t_pred.set_color(C["green"])
        else:
            self._t_pred.set_text("—")
            self._t_pred.set_color(C["sub"])

        if last_pb is not None:
            self._t_prob.set_text(f"{last_pb:.3f}")
            self._t_prob.set_color(C["red"] if last_pb >= 0.5 else C["green"])
        else:
            self._t_prob.set_text("—")
            self._t_prob.set_color(C["sub"])

        # ── recording header update ───────────────────────────────────────────
        if rec:
            if base_n < BASELINE_REPS:
                self._set_hdr(
                    f"● RECORDING  ·  building baseline  {base_n}/{BASELINE_REPS} reps",
                    C["orange"],
                )
            else:
                self._set_hdr(
                    f"● RECORDING  ·  {counted} reps  ·  press R to stop",
                    C["rec"],
                )


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",      default=None)
    parser.add_argument("--baud",      default=BAUD_RATE, type=int)
    parser.add_argument("--window",    default=WINDOW_SEC, type=float)
    parser.add_argument("--exercise",  default="bicep_curl")
    parser.add_argument("--csv",       default=None)
    parser.add_argument("--delay",     default=0.01, type=float)
    parser.add_argument("--threshold", default=None, type=float)
    args = parser.parse_args()

    model, feature_columns, threshold = load_model_bundle()
    if args.threshold is not None:
        threshold = args.threshold

    exercise = args.exercise.strip().lower().replace(" ", "_")
    stop_ev  = threading.Event()

    if args.csv:
        reader = threading.Thread(target=csv_reader,
                                  args=(args.csv, args.delay, stop_ev), daemon=True)
    else:
        port   = args.port or find_port()
        reader = threading.Thread(target=serial_reader,
                                  args=(port, args.baud, stop_ev), daemon=True)
    reader.start()

    clf = threading.Thread(target=classifier_thread,
                           args=(model, feature_columns, threshold, stop_ev), daemon=True)
    clf.start()

    gui = GUI(exercise, args.window)
    ani = FuncAnimation(gui.fig, gui.update, interval=80,    # noqa: F841
                        blit=False, cache_frame_data=False)

    print("\nControls:  R = start/stop recording    Q = quit\n")
    try:
        plt.show()
    finally:
        stop_ev.set()
        with _lock:
            fp = _st.get("record_fp")
        if fp:
            fp.close()


if __name__ == "__main__":
    main()
