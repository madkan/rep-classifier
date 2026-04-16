#!/usr/bin/env python3
"""
Real-time IMU plotter for MPU6050 streamed from an ESP32.

Updated for recording full weightlifting sets instead of gesture clips.

Keys:
  r  -> start a new set recording
  s  -> stop recording

Examples:
  python plot_imu_sets.py
  python plot_imu_sets.py --port /dev/tty.usbserial-11130
  python plot_imu_sets.py --exercise bicep_curl
"""

import argparse
import re
import sys
import threading
import time
from collections import deque
import os

import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import serial
import serial.tools.list_ports

# ── Configuration ────────────────────────────────────────────────────────────
BAUD_RATE  = 115200
WINDOW_SEC = 5
SAMPLE_HZ  = 100
DATA_DIR   = "data"

LINE_RE = re.compile(
    r"AX:(?P<ax>[-\d.]+)\s+AY:(?P<ay>[-\d.]+)\s+AZ:(?P<az>[-\d.]+)"
    r"\s*\|\s*"
    r"GX:(?P<gx>[-\d.]+)\s+GY:(?P<gy>[-\d.]+)\s+GZ:(?P<gz>[-\d.]+)"
    r"\s*\|\s*"
    r"T:(?P<t>[-\d.]+)"
)

C = {
    "bg":     "#0f1117",
    "panel":  "#1a1d27",
    "grid":   "#2a2d3a",
    "ax":     "#4fc3f7",
    "ay":     "#81d4fa",
    "az":     "#b3e5fc",
    "gx":     "#f48fb1",
    "gy":     "#f06292",
    "gz":     "#e91e63",
    "temp":   "#ffcc02",
    "text":   "#e0e0e0",
    "title":  "#ffffff",
}


def find_port() -> str:
    ports = serial.tools.list_ports.comports()
    usb = [p for p in ports if "usb" in p.device.lower() or "usbserial" in p.device.lower()]
    if usb:
        return usb[0].device
    if ports:
        return ports[0].device
    print("[ERROR] No serial ports found. Plug in your ESP32 or specify --port.", file=sys.stderr)
    sys.exit(1)


def parse_line(line: str):
    m = LINE_RE.search(line)
    if m:
        return tuple(float(m.group(k)) for k in ("ax", "ay", "az", "gx", "gy", "gz", "t"))
    return None


class SerialReader(threading.Thread):
    """Background thread that fills shared deques and optionally records samples."""

    def __init__(self, port: str, baud: int, buf_size: int):
        super().__init__(daemon=True)
        self.port     = port
        self.baud     = baud
        self.buf_size = buf_size
        self.lock     = threading.Lock()

        self.t    = deque(maxlen=buf_size)
        self.ax   = deque(maxlen=buf_size)
        self.ay   = deque(maxlen=buf_size)
        self.az   = deque(maxlen=buf_size)
        self.gx   = deque(maxlen=buf_size)
        self.gy   = deque(maxlen=buf_size)
        self.gz   = deque(maxlen=buf_size)
        self.temp = deque(maxlen=buf_size)

        self.connected = False
        self.status    = "Connecting..."

        self.recording = False
        self.record_fp = None
        self.record_path = None
        self.record_lock = threading.Lock()

    def start_recording(self, filepath: str):
        with self.record_lock:
            if self.record_fp:
                self.record_fp.close()
            self.record_fp = open(filepath, "w", encoding="utf-8", buffering=1)
            self.record_fp.write("t_us,ax_g,ay_g,az_g,gx_dps,gy_dps,gz_dps,temp_c\n")
            self.record_path = filepath
            self.recording = True

    def stop_recording(self):
        with self.record_lock:
            if self.record_fp:
                self.record_fp.close()
            self.record_fp = None
            self.record_path = None
            self.recording = False

    def _write_sample_if_recording(self, t_us, ax, ay, az, gx, gy, gz, temp):
        with self.record_lock:
            if not self.recording or self.record_fp is None:
                return
            self.record_fp.write(
                f"{t_us},{ax:.6f},{ay:.6f},{az:.6f},{gx:.6f},{gy:.6f},{gz:.6f},{temp:.2f}\n"
            )

    def run(self):
        while True:
            try:
                with serial.Serial(self.port, self.baud, timeout=1) as ser:
                    self.connected = True
                    self.status = f"Connected  {self.port}  @{self.baud} baud"
                    t0 = time.perf_counter()

                    while True:
                        raw = ser.readline()
                        try:
                            line = raw.decode("utf-8", errors="replace").strip()
                        except Exception:
                            continue

                        parsed = parse_line(line)
                        if parsed is None:
                            continue

                        ax, ay, az, gx, gy, gz, temp = parsed
                        now = time.perf_counter() - t0
                        t_us = int(now * 1_000_000)

                        with self.lock:
                            self.t.append(now)
                            self.ax.append(ax)
                            self.ay.append(ay)
                            self.az.append(az)
                            self.gx.append(gx)
                            self.gy.append(gy)
                            self.gz.append(gz)
                            self.temp.append(temp)

                        self._write_sample_if_recording(t_us, ax, ay, az, gx, gy, gz, temp)

            except serial.SerialException as e:
                self.connected = False
                self.status = f"Disconnected — {e}  (retrying...)"
                time.sleep(2)

    def snapshot(self):
        with self.lock:
            return (
                list(self.t),
                list(self.ax), list(self.ay), list(self.az),
                list(self.gx), list(self.gy), list(self.gz),
                list(self.temp),
            )


def style_axes(ax, ylabel, ylim=None):
    ax.set_facecolor(C["panel"])
    ax.tick_params(colors=C["text"], labelsize=8)
    ax.yaxis.label.set_color(C["text"])
    ax.xaxis.label.set_color(C["text"])
    ax.set_ylabel(ylabel, fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(C["grid"])
    ax.grid(True, color=C["grid"], linewidth=0.5, linestyle="--")
    if ylim:
        ax.set_ylim(*ylim)


def next_set_filename(exercise: str) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)

    existing = []
    prefix = f"{exercise}_set_"
    suffix = ".csv"

    for name in os.listdir(DATA_DIR):
        if name.startswith(prefix) and name.endswith(suffix):
            middle = name[len(prefix):-len(suffix)]
            if middle.isdigit():
                existing.append(int(middle))

    next_num = 1 if not existing else max(existing) + 1
    return os.path.join(DATA_DIR, f"{exercise}_set_{next_num:02d}.csv")


def main():
    parser = argparse.ArgumentParser(description="Real-time MPU6050 plotter for recording lifting sets")
    parser.add_argument("--port", default=None, help="Serial port (auto-detected if omitted)")
    parser.add_argument("--baud", default=BAUD_RATE, type=int, help=f"Baud rate (default {BAUD_RATE})")
    parser.add_argument("--window", default=WINDOW_SEC, type=float, help=f"Plot window in seconds (default {WINDOW_SEC})")
    parser.add_argument("--exercise", default="bicep_curl", help="Exercise name used in saved filenames")
    args = parser.parse_args()

    port     = args.port or find_port()
    baud     = args.baud
    window   = args.window
    exercise = args.exercise.strip().lower().replace(" ", "_")
    buf_size = int(window * SAMPLE_HZ * 2)

    print(f"[INFO] Opening {port} at {baud} baud ...")
    reader = SerialReader(port, baud, buf_size)
    reader.start()

    os.makedirs(DATA_DIR, exist_ok=True)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(13, 8), facecolor=C["bg"])
    fig.canvas.manager.set_window_title("MPU6050 — Real-Time Stream")

    gs = gridspec.GridSpec(
        3, 1, figure=fig, hspace=0.45,
        left=0.08, right=0.97, top=0.90, bottom=0.08
    )

    ax_acc  = fig.add_subplot(gs[0])
    ax_gyro = fig.add_subplot(gs[1])
    ax_temp = fig.add_subplot(gs[2])

    style_axes(ax_acc,  "Acceleration (g)")
    style_axes(ax_gyro, "Angular rate (°/s)")
    style_axes(ax_temp, "Temperature (°C)")

    fig.suptitle(
        f"MPU6050 Real-Time Stream — 100 Hz — Exercise: {exercise}",
        color=C["title"], fontsize=13, fontweight="bold"
    )
    status_txt = fig.text(0.5, 0.005, reader.status, ha="center", fontsize=8, color="#888888")

    la_x, = ax_acc.plot([], [], color=C["ax"], lw=1.2, label="Accel X")
    la_y, = ax_acc.plot([], [], color=C["ay"], lw=1.2, label="Accel Y")
    la_z, = ax_acc.plot([], [], color=C["az"], lw=1.2, label="Accel Z")

    lg_x, = ax_gyro.plot([], [], color=C["gx"], lw=1.2, label="Gyro X")
    lg_y, = ax_gyro.plot([], [], color=C["gy"], lw=1.2, label="Gyro Y")
    lg_z, = ax_gyro.plot([], [], color=C["gz"], lw=1.2, label="Gyro Z")

    lt, = ax_temp.plot([], [], color=C["temp"], lw=1.5, label="Temperature")

    for a, lines in [
        (ax_acc,  [la_x, la_y, la_z]),
        (ax_gyro, [lg_x, lg_y, lg_z]),
        (ax_temp, [lt])
    ]:
        a.legend(
            handles=lines, loc="upper left", fontsize=7,
            facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"]
        )

    ax_temp.set_xlabel("Time (s)", fontsize=9)

    def readout(ax, y, text=""):
        return ax.text(
            1.001, y, text, transform=ax.transAxes,
            color=C["text"], fontsize=7.5, va="center",
            fontfamily="monospace"
        )

    ro_ax = readout(ax_acc,  0.83)
    ro_ay = readout(ax_acc,  0.50)
    ro_az = readout(ax_acc,  0.17)
    ro_gx = readout(ax_gyro, 0.83)
    ro_gy = readout(ax_gyro, 0.50)
    ro_gz = readout(ax_gyro, 0.17)
    ro_t  = readout(ax_temp, 0.50)

    def update(_):
        t, ax_, ay_, az_, gx_, gy_, gz_, tmp = reader.snapshot()

        if len(t) < 2:
            status_txt.set_text(reader.status)
            return

        t_now = t[-1]
        t_lo = t_now - window

        def trim(xs, ts):
            return [x for x, ti in zip(xs, ts) if ti >= t_lo]

        tv   = [ti for ti in t if ti >= t_lo]
        axv  = trim(ax_, t)
        ayv  = trim(ay_, t)
        azv  = trim(az_, t)
        gxv  = trim(gx_, t)
        gyv  = trim(gy_, t)
        gzv  = trim(gz_, t)
        tmpv = trim(tmp, t)

        la_x.set_data(tv, axv)
        la_y.set_data(tv, ayv)
        la_z.set_data(tv, azv)

        lg_x.set_data(tv, gxv)
        lg_y.set_data(tv, gyv)
        lg_z.set_data(tv, gzv)

        lt.set_data(tv, tmpv)

        for a in (ax_acc, ax_gyro, ax_temp):
            a.set_xlim(t_lo, t_now)

        def auto_ylim(ax_obj, *series):
            vals = [v for s in series for v in s]
            if vals:
                lo, hi = min(vals), max(vals)
                pad = max((hi - lo) * 0.15, 0.05)
                ax_obj.set_ylim(lo - pad, hi + pad)

        auto_ylim(ax_acc,  axv, ayv, azv)
        auto_ylim(ax_gyro, gxv, gyv, gzv)
        auto_ylim(ax_temp, tmpv)

        ro_ax.set_text(f"AX {ax_[-1]:+7.3f}")
        ro_ay.set_text(f"AY {ay_[-1]:+7.3f}")
        ro_az.set_text(f"AZ {az_[-1]:+7.3f}")
        ro_gx.set_text(f"GX {gx_[-1]:+7.2f}")
        ro_gy.set_text(f"GY {gy_[-1]:+7.2f}")
        ro_gz.set_text(f"GZ {gz_[-1]:+7.2f}")
        ro_t.set_text(f"{tmp[-1]:.2f} °C")

        if reader.recording and reader.record_path:
            status_txt.set_text(f"{reader.status} | RECORDING -> {reader.record_path}")
        else:
            status_txt.set_text(reader.status)

    from matplotlib.animation import FuncAnimation
    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)

    def on_key(event):
        k = (event.key or "").lower()

        if k == "r":
            if reader.recording:
                print("[REC] already recording — press 's' first to stop")
                return

            filepath = next_set_filename(exercise)
            reader.start_recording(filepath)
            print(f"[REC] started -> {filepath}")
            return

        if k == "s":
            if reader.recording:
                print("[REC] stopped")
                reader.stop_recording()
            return

    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.show()
    reader.stop_recording()


if __name__ == "__main__":
    main()