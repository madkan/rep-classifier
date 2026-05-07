import argparse
import threading
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


MODEL_PATH = "rep_classifier_bundle.pkl"

#rep segmentation settings
MIN_REP_DURATION = 1.2
MAX_REP_DURATION = 8.0
MIN_REP_RANGE = 1.0
SMOOTH_WINDOW = 11

#wait this long after a rep ending valley before classifying the rep to prevent duplicates
REP_END_CONFIRMATION_SEC = 0.40

#prevents duplicate classifications if detected valley shifts slightly
MIN_TIME_BETWEEN_CLASSIFIED_REPS = 0.80

#first few reps are used to build baseline
BASELINE_REPS = 3

#if a detected segment is much longer than the baseline reps, skip it as it is probably the failure rep (still tuning this)
MAX_DURATION_RATIO = 1.5

#valley/peak segmentation tuning
VALLEY_PROMINENCE_FRAC = 0.20
MIN_VALLEY_SEPARATION_SEC = 0.60
MIN_PEAK_RISE_FRAC = 0.35

REQUIRED_COLUMNS = [
    "t_us",
    "ax_g",
    "ay_g",
    "az_g",
    "gx_dps",
    "gy_dps",
    "gz_dps",
]

stop_requested = False

def wait_for_stop_key():
    global stop_requested
    input("\nPress Enter again to end the set...\n")
    stop_requested = True

def signal_magnitude(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)

def safe_iqr(x):
    if len(x) == 0:
        return np.nan

    return np.percentile(x, 75) - np.percentile(x, 25)

def time_domain_features(x, prefix):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    if len(x) == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_range": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_iqr": np.nan,
            f"{prefix}_rms": np.nan,
            f"{prefix}_energy": np.nan,
        }

    return {
        f"{prefix}_mean": np.mean(x),
        f"{prefix}_std": np.std(x),
        f"{prefix}_min": np.min(x),
        f"{prefix}_max": np.max(x),
        f"{prefix}_range": np.max(x) - np.min(x),
        f"{prefix}_median": np.median(x),
        f"{prefix}_iqr": safe_iqr(x),
        f"{prefix}_rms": np.sqrt(np.mean(x**2)),
        f"{prefix}_energy": np.sum(x**2),
    }


def jerk_features(x, fs, prefix):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    if len(x) < 2:
        return {
            f"{prefix}_jerk_mean": np.nan,
            f"{prefix}_jerk_std": np.nan,
            f"{prefix}_jerk_max": np.nan,
        }

    jerk = np.diff(x) * fs

    return {
        f"{prefix}_jerk_mean": np.mean(np.abs(jerk)),
        f"{prefix}_jerk_std": np.std(jerk),
        f"{prefix}_jerk_max": np.max(np.abs(jerk)),
    }


def frequency_domain_features(x, fs, prefix):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    if len(x) < 8:
        return {
            f"{prefix}_dominant_freq": np.nan,
            f"{prefix}_spectral_energy": np.nan,
            f"{prefix}_spectral_entropy": np.nan,
            f"{prefix}_low_freq_energy": np.nan,
            f"{prefix}_mid_freq_energy": np.nan,
            f"{prefix}_high_freq_energy": np.nan,
            f"{prefix}_high_low_energy_ratio": np.nan,
        }

    x = x - np.mean(x)

    fft_values = np.fft.rfft(x)
    fft_magnitude = np.abs(fft_values)
    freqs = np.fft.rfftfreq(len(x), d=1 / fs)

    #remove 0 Hz component
    freqs = freqs[1:]
    fft_magnitude = fft_magnitude[1:]

    power = fft_magnitude**2
    total_power = np.sum(power)

    if total_power == 0:
        return {
            f"{prefix}_dominant_freq": 0,
            f"{prefix}_spectral_energy": 0,
            f"{prefix}_spectral_entropy": 0,
            f"{prefix}_low_freq_energy": 0,
            f"{prefix}_mid_freq_energy": 0,
            f"{prefix}_high_freq_energy": 0,
            f"{prefix}_high_low_energy_ratio": 0,
        }

    dominant_freq = freqs[np.argmax(power)]

    low_freq_energy = np.sum(power[(freqs >= 0.1) & (freqs <= 1.0)])
    mid_freq_energy = np.sum(power[(freqs > 1.0) & (freqs <= 3.0)])
    high_freq_energy = np.sum(power[(freqs > 3.0) & (freqs <= 10.0)])

    high_low_energy_ratio = high_freq_energy / (low_freq_energy + 1e-8)

    power_norm = power / total_power
    spectral_entropy = -np.sum(power_norm * np.log2(power_norm + 1e-12))

    return {
        f"{prefix}_dominant_freq": dominant_freq,
        f"{prefix}_spectral_energy": total_power,
        f"{prefix}_spectral_entropy": spectral_entropy,
        f"{prefix}_low_freq_energy": low_freq_energy,
        f"{prefix}_mid_freq_energy": mid_freq_energy,
        f"{prefix}_high_freq_energy": high_freq_energy,
        f"{prefix}_high_low_energy_ratio": high_low_energy_ratio,
    }


def estimate_sampling_rate(time_s):
    if len(time_s) < 2:
        return 100.0

    diffs = np.diff(time_s)
    diffs = diffs[diffs > 0]

    if len(diffs) == 0:
        return 100.0

    median_dt = np.median(diffs)

    if median_dt == 0:
        return 100.0

    return 1.0 / median_dt


def smooth_signal(x, window_size=11):
    x = np.asarray(x, dtype=float)

    if len(x) < window_size:
        return x

    kernel = np.ones(window_size) / window_size
    return np.convolve(x, kernel, mode="same")


def merge_close_valleys(valleys, smooth, min_sep_samples):
    if len(valleys) == 0:
        return valleys

    merged = [valleys[0]]

    for v in valleys[1:]:
        if v - merged[-1] < min_sep_samples:
            #keep the deeper valley
            if smooth[v] < smooth[merged[-1]]:
                merged[-1] = v
        else:
            merged.append(v)

    return np.array(merged)


def extract_rep_features(rep_df):
    time_s = rep_df["time_s"].values
    fs = estimate_sampling_rate(time_s)

    ax = rep_df["ax_g"].astype(float).values
    ay = rep_df["ay_g"].astype(float).values
    az = rep_df["az_g"].astype(float).values

    gx = rep_df["gx_dps"].astype(float).values
    gy = rep_df["gy_dps"].astype(float).values
    gz = rep_df["gz_dps"].astype(float).values

    accel_mag = signal_magnitude(ax, ay, az)
    gyro_mag = signal_magnitude(gx, gy, gz)

    feat = {}

    #original preprocessing features
    feat["duration"] = time_s[-1] - time_s[0]
    feat["max_ay"] = np.max(ay)
    feat["min_ay"] = np.min(ay)
    feat["range_ay"] = feat["max_ay"] - feat["min_ay"]
    feat["mean_ay"] = np.mean(ay)
    feat["std_ay"] = np.std(ay)
    feat["gyro_var"] = np.var(gx)

    #additional features
    #not including raw "ay" here because max_ay/min_ay/range_ay/mean_ay/std_ay already represent the ay axis.
    signals = {
        "ax": ax,
        "az": az,
        "gx": gx,
        "gy": gy,
        "gz": gz,
        "accel_mag": accel_mag,
        "gyro_mag": gyro_mag,
    }

    for name, signal in signals.items():
        feat.update(time_domain_features(signal, name))

    feat.update(jerk_features(accel_mag, fs, "accel_mag"))
    feat.update(jerk_features(gyro_mag, fs, "gyro_mag"))

    feat.update(frequency_domain_features(accel_mag, fs, "accel_mag"))
    feat.update(frequency_domain_features(gyro_mag, fs, "gyro_mag"))

    feat["segment_num_samples"] = len(rep_df)
    feat["segment_sampling_rate"] = fs

    return feat


def add_baseline_features_to_current_rep(current_features, baseline_features):
    feat = dict(current_features)

    baseline_source_columns = [
        "duration",
        "std_ay",
        "gyro_var",
        "range_ay",

        "accel_mag_std",
        "accel_mag_range",
        "accel_mag_rms",
        "accel_mag_jerk_std",
        "accel_mag_spectral_energy",
        "accel_mag_high_freq_energy",
        "accel_mag_high_low_energy_ratio",

        "gyro_mag_std",
        "gyro_mag_range",
        "gyro_mag_rms",
        "gyro_mag_jerk_std",
        "gyro_mag_spectral_energy",
        "gyro_mag_high_freq_energy",
        "gyro_mag_high_low_energy_ratio",
    ]

    for col in baseline_source_columns:
        if col not in feat:
            continue

        baseline_values = [
            rep_feat[col]
            for rep_feat in baseline_features
            if col in rep_feat and not pd.isna(rep_feat[col])
        ]

        if len(baseline_values) == 0:
            baseline_value = np.nan
        else:
            baseline_value = np.mean(baseline_values)

        feat[f"baseline_{col}"] = baseline_value
        feat[f"{col}_ratio_to_baseline"] = feat[col] / (baseline_value + 1e-8)
        feat[f"{col}_diff_from_baseline"] = feat[col] - baseline_value

    return feat


def make_model_input(features, feature_columns):
    row = {}

    for col in feature_columns:
        row[col] = features.get(col, np.nan)

    return pd.DataFrame([row], columns=feature_columns)


def parse_serial_line(line):
    line = line.strip()

    if not line:
        return None

    if "t_us" in line or "ax_g" in line:
        return None

    parts = line.split(",")

    if len(parts) < 7:
        return None

    try:
        sample = {
            "t_us": float(parts[0]),
            "ax_g": float(parts[1]),
            "ay_g": float(parts[2]),
            "az_g": float(parts[3]),
            "gx_dps": float(parts[4]),
            "gy_dps": float(parts[5]),
            "gz_dps": float(parts[6]),
        }
        sample["time_s"] = sample["t_us"] / 1_000_000.0
        return sample

    except ValueError:
        return None


def get_completed_reps_from_buffer(buffer_df, last_classified_end_time):
    if len(buffer_df) < 20:
        return []

    signal = buffer_df["ay_g"].values
    time_s = buffer_df["time_s"].values

    fs = estimate_sampling_rate(time_s)
    smooth = smooth_signal(signal, window_size=SMOOTH_WINDOW)

    min_rep_samples = int(MIN_REP_DURATION * fs)
    min_valley_sep_samples = int(MIN_VALLEY_SEPARATION_SEC * fs)

    if len(smooth) < min_rep_samples * 2:
        return []

    global_range = np.max(smooth) - np.min(smooth)

    if global_range < MIN_REP_RANGE:
        return []

    valley_prominence = VALLEY_PROMINENCE_FRAC * global_range

    valleys, _ = find_peaks(
        -smooth,
        distance=max(1, min_valley_sep_samples),
        prominence=valley_prominence
    )

    if len(valleys) < 2:
        return []

    valleys = merge_close_valleys(
        valleys=valleys,
        smooth=smooth,
        min_sep_samples=min_valley_sep_samples
    )

    if len(valleys) < 2:
        return []

    completed_reps = []

    for i in range(len(valleys) - 1):
        start = valleys[i]
        end = valleys[i + 1]

        start_time = time_s[start]
        end_time = time_s[end]
        duration = end_time - start_time

        if last_classified_end_time is not None:
            if end_time <= last_classified_end_time + MIN_TIME_BETWEEN_CLASSIFIED_REPS:
                continue

        if duration < MIN_REP_DURATION:
            continue

        if duration > MAX_REP_DURATION:
            continue

        if end - start < 3:
            continue

        segment = smooth[start:end + 1]

        local_peak_offset = np.argmax(segment)
        peak_idx = start + local_peak_offset
        peak_value = smooth[peak_idx]

        start_valley_value = smooth[start]
        end_valley_value = smooth[end]

        peak_rise = peak_value - max(start_valley_value, end_valley_value)

        if peak_rise < MIN_PEAK_RISE_FRAC * global_range:
            continue

        seg_range = np.max(segment) - np.min(segment)

        if seg_range < MIN_REP_RANGE:
            continue

        samples_after_end = len(buffer_df) - end

        if samples_after_end < int(REP_END_CONFIRMATION_SEC * fs):
            continue

        rep_df = buffer_df.iloc[start:end].copy()

        completed_reps.append({
            "start_idx": start,
            "end_idx": end,
            "peak_idx": peak_idx,
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "rep_df": rep_df,
        })

    return completed_reps


def classify_rep(rep_df, rep_count, baseline_features, model, feature_columns, threshold):
    raw_features = extract_rep_features(rep_df)

    if len(baseline_features) == 0:
        full_features = add_baseline_features_to_current_rep(
            raw_features,
            [raw_features]
        )
    else:
        full_features = add_baseline_features_to_current_rep(
            raw_features,
            baseline_features
        )

    X_new = make_model_input(full_features, feature_columns)

    proba_near_failure = model.predict_proba(X_new)[0, 1]

    if rep_count <= BASELINE_REPS:
        pred = 0
    else:
        pred = int(proba_near_failure >= threshold)

    label = "NEAR FAILURE" if pred == 1 else "normal"

    print("\n" + "-" * 60)
    print(f"Rep {rep_count}")
    print(f"Prediction: {label}")
    print(f"Near-failure probability: {proba_near_failure:.3f}")
    print(f"Threshold: {threshold}")
    print(f"Duration: {raw_features['duration']:.2f} sec")

    if rep_count <= BASELINE_REPS:
        print("Note: baseline rep, forced prediction to normal")

    print("-" * 60)

    return raw_features, pred, proba_near_failure


def plot_detected_reps(buffer_df, detected_reps, title="Detected Reps"):
    if len(buffer_df) == 0:
        print("No data to plot.")
        return

    time_s = buffer_df["time_s"].values
    ay = buffer_df["ay_g"].values
    smooth = smooth_signal(ay, window_size=SMOOTH_WINDOW)

    plt.figure(figsize=(12, 6))

    plt.plot(time_s, ay, alpha=0.35, label="Raw ay_g")
    plt.plot(time_s, smooth, linewidth=2, label="Smoothed ay_g")

    for i, rep in enumerate(detected_reps, start=1):
        start_time = rep["start_time"]
        end_time = rep["end_time"]
        peak_time = rep.get("peak_time", None)

        prediction = rep.get("prediction", "unknown")
        probability = rep.get("probability", None)
        duration_ratio = rep.get("duration_ratio", None)

        if prediction == "NEAR FAILURE":
            color = "red"
        elif prediction == "skipped":
            color = "gray"
        else:
            color = "orange"

        plt.axvspan(start_time, end_time, alpha=0.25, color=color)

        if peak_time is not None:
            closest_peak_idx = np.argmin(np.abs(time_s - peak_time))
            plt.scatter(
                time_s[closest_peak_idx],
                smooth[closest_peak_idx],
                s=35,
                color="black",
                zorder=5
            )

        mid_time = (start_time + end_time) / 2.0
        closest_idx = np.argmin(np.abs(time_s - mid_time))
        y_val = smooth[closest_idx]

        label_text = str(i)

        if probability is not None:
            label_text += f"\nP={probability:.2f}"

        if duration_ratio is not None:
            label_text += f"\nD={duration_ratio:.2f}"

        plt.text(
            mid_time,
            y_val,
            label_text,
            color="black",
            fontsize=8,
            ha="center",
            va="bottom"
        )

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("ay_g")
    plt.legend()
    plt.grid(True)
    plt.show()


def load_model_bundle():
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    threshold = bundle.get("near_failure_threshold", 0.30)

    print(f"Loaded model from {MODEL_PATH}")
    print(f"Using near-failure threshold: {threshold}")

    return model, feature_columns, threshold


def should_skip_long_rep(rep_df, baseline_features):

    if len(baseline_features) < BASELINE_REPS:
        return False, 1.0

    duration = rep_df["time_s"].iloc[-1] - rep_df["time_s"].iloc[0]

    baseline_durations = [
        feat["duration"]
        for feat in baseline_features
        if "duration" in feat and not pd.isna(feat["duration"])
    ]

    if len(baseline_durations) == 0:
        return False, 1.0

    baseline_duration = np.mean(baseline_durations)
    duration_ratio = duration / (baseline_duration + 1e-8)

    if duration_ratio > MAX_DURATION_RATIO:
        return True, duration_ratio

    return False, duration_ratio


def run_live_serial(port, baud, threshold_override=None):
    try:
        import serial
    except ImportError:
        print("pyserial is not installed. Run: pip install pyserial")
        return

    model, feature_columns, threshold = load_model_bundle()

    if threshold_override is not None:
        threshold = threshold_override
        print(f"Overriding threshold with: {threshold}")

    input("\nPress Enter to start the set...")

    global stop_requested
    stop_requested = False

    stop_thread = threading.Thread(target=wait_for_stop_key)
    stop_thread.daemon = True
    stop_thread.start()

    buffer = []
    last_classified_end_time = None
    baseline_features = []
    rep_count = 0
    detected_reps = []

    with serial.Serial(port, baud, timeout=1) as ser:
        print(f"Reading from {port} at {baud} baud...")

        while not stop_requested:
            line = ser.readline().decode("utf-8", errors="ignore")
            sample = parse_serial_line(line)

            if sample is None:
                continue

            buffer.append(sample)

            buffer_df = pd.DataFrame(buffer)

            completed_reps = get_completed_reps_from_buffer(
                buffer_df,
                last_classified_end_time
            )

            for rep_info in completed_reps:
                end_time = rep_info["end_time"]

                if last_classified_end_time is not None:
                    if end_time <= last_classified_end_time + MIN_TIME_BETWEEN_CLASSIFIED_REPS:
                        continue

                skip_long_rep, duration_ratio = should_skip_long_rep(
                    rep_info["rep_df"],
                    baseline_features
                )

                if skip_long_rep:
                    print(
                        f"Skipping partial/failed rep attempt: "
                        f"duration ratio={duration_ratio:.2f}"
                    )

                    detected_reps.append({
                        "start_time": rep_info["start_time"],
                        "end_time": rep_info["end_time"],
                        "peak_time": None,
                        "prediction": "skipped",
                        "probability": None,
                        "duration_ratio": duration_ratio,
                    })

                    last_classified_end_time = end_time
                    continue

                rep_count += 1

                raw_features, pred, proba = classify_rep(
                    rep_df=rep_info["rep_df"],
                    rep_count=rep_count,
                    baseline_features=baseline_features,
                    model=model,
                    feature_columns=feature_columns,
                    threshold=threshold
                )

                prediction_label = "NEAR FAILURE" if pred == 1 else "normal"

                if len(baseline_features) < BASELINE_REPS:
                    baseline_features.append(raw_features)
                    print(
                        f"Baseline reps collected: "
                        f"{len(baseline_features)}/{BASELINE_REPS}"
                    )

                detected_reps.append({
                    "start_time": rep_info["start_time"],
                    "end_time": rep_info["end_time"],
                    "peak_time": buffer_df["time_s"].iloc[rep_info["peak_idx"]],
                    "prediction": prediction_label,
                    "probability": proba,
                    "duration_ratio": duration_ratio,
                })

                last_classified_end_time = end_time

        final_buffer_df = pd.DataFrame(buffer)

        plot_detected_reps(
            final_buffer_df,
            detected_reps,
            title="Live Set Segmentation Check"
        )

        print("\nSet ended.")
        print(f"Total reps detected: {rep_count}")


def run_from_csv(csv_path, delay=0.01, threshold_override=None):
    model, feature_columns, threshold = load_model_bundle()

    if threshold_override is not None:
        threshold = threshold_override
        print(f"Overriding threshold with: {threshold}")

    raw = pd.read_csv(csv_path)

    missing = [col for col in REQUIRED_COLUMNS if col not in raw.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    raw["time_s"] = raw["t_us"] / 1_000_000.0

    input("\nPress Enter to start replaying the set...")

    buffer = []
    last_classified_end_time = None
    baseline_features = []
    rep_count = 0
    detected_reps = []

    for _, row in raw.iterrows():
        sample = {
            "t_us": row["t_us"],
            "ax_g": row["ax_g"],
            "ay_g": row["ay_g"],
            "az_g": row["az_g"],
            "gx_dps": row["gx_dps"],
            "gy_dps": row["gy_dps"],
            "gz_dps": row["gz_dps"],
            "time_s": row["time_s"],
        }

        buffer.append(sample)
        buffer_df = pd.DataFrame(buffer)

        completed_reps = get_completed_reps_from_buffer(
            buffer_df,
            last_classified_end_time
        )

        for rep_info in completed_reps:
            end_time = rep_info["end_time"]

            if last_classified_end_time is not None:
                if end_time <= last_classified_end_time + MIN_TIME_BETWEEN_CLASSIFIED_REPS:
                    continue

            skip_long_rep, duration_ratio = should_skip_long_rep(
                rep_info["rep_df"],
                baseline_features
            )

            if skip_long_rep:
                print(
                    f"Skipping partial/failed rep attempt: "
                    f"duration ratio={duration_ratio:.2f}"
                )

                detected_reps.append({
                    "start_time": rep_info["start_time"],
                    "end_time": rep_info["end_time"],
                    "peak_time": None,
                    "prediction": "skipped",
                    "probability": None,
                    "duration_ratio": duration_ratio,
                })

                last_classified_end_time = end_time
                continue

            rep_count += 1

            raw_features, pred, proba = classify_rep(
                rep_df=rep_info["rep_df"],
                rep_count=rep_count,
                baseline_features=baseline_features,
                model=model,
                feature_columns=feature_columns,
                threshold=threshold
            )

            prediction_label = "NEAR FAILURE" if pred == 1 else "normal"

            if len(baseline_features) < BASELINE_REPS:
                baseline_features.append(raw_features)
                print(
                    f"Baseline reps collected: "
                    f"{len(baseline_features)}/{BASELINE_REPS}"
                )

            detected_reps.append({
                "start_time": rep_info["start_time"],
                "end_time": rep_info["end_time"],
                "peak_time": buffer_df["time_s"].iloc[rep_info["peak_idx"]],
                "prediction": prediction_label,
                "probability": proba,
                "duration_ratio": duration_ratio,
            })

            last_classified_end_time = end_time

        time.sleep(delay)

    final_buffer_df = pd.DataFrame(buffer)

    plot_detected_reps(
        final_buffer_df,
        detected_reps,
        title="Replay Segmentation Check"
    )

    print("\nReplay ended.")
    print(f"Total reps detected: {rep_count}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--serial",
        type=str,
        default=None,
        help="Serial port, for example /dev/cu.usbserial-210"
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=115200,
        help="Serial baud rate"
    )

    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional raw CSV file to replay for testing"
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.01,
        help="Delay between rows when replaying a CSV"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the saved near-failure threshold"
    )

    args = parser.parse_args()

    if args.csv is not None:
        run_from_csv(
            args.csv,
            delay=args.delay,
            threshold_override=args.threshold
        )

    elif args.serial is not None:
        run_live_serial(
            args.serial,
            args.baud,
            threshold_override=args.threshold
        )

    else:
        print("Choose either live serial mode or CSV replay mode.")
        print()
        print("Live serial example:")
        print("python real_time_predictor.py --serial /dev/cu.usbserial-210 --baud 115200")
        print()
        print("CSV replay example:")
        print("python real_time_predictor.py --csv data/bicep_curl/set_01.csv")
        print()
        print("CSV replay with custom threshold:")
        print("python real_time_predictor.py --csv data/bicep_curl/set_01.csv --threshold 0.30")


if __name__ == "__main__":
    main()