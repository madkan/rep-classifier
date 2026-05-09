import argparse
import re
import threading
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt


MODEL_PATH = "classifiers/lateral_raise_classifier_bundle.pkl"

SIGNAL_COL = "az_g"

MIN_REP_DURATION = 1.5
MAX_REP_DURATION = 8.0
MIN_REP_RANGE = 0.30

CUTOFF_HZ = 5.0
FILTER_ORDER = 4

PROM_FRACTION = 0.50
MERGE_GAP_SEC = 0.80

#wait this long after a rep ending valley before classifying the rep
REP_END_CONFIRMATION_SEC = 0.40

#prevents duplicate classifications if detected valley shifts slightly
MIN_TIME_BETWEEN_CLASSIFIED_REPS = 0.80

#first few reps are used to build baseline
BASELINE_REPS = 3

#if a detected segment has a much smaller movement range than baseline, skip it because it is probably the failed rep
MIN_RANGE_RATIO_ALLOWED = 0.50

#skip reps that do not return enough after the peak
MIN_END_RETURN_RATIO = 0.60

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


def lowpass_filter_signal(signal, fs, cutoff_hz=CUTOFF_HZ):
    signal = np.asarray(signal, dtype=float)

    if len(signal) < 20:
        return signal

    nyq = fs / 2.0
    norm_cutoff = min(cutoff_hz / nyq, 0.99)

    b, a = butter(N=FILTER_ORDER, Wn=norm_cutoff, btype="low")

    #filtfilt needs enough samples for padding
    padlen = 3 * max(len(a), len(b))

    if len(signal) <= padlen:
        return signal

    return filtfilt(b, a, signal)


def merge_short_segments(raw_segments, fs):
    if len(raw_segments) == 0:
        return []

    merge_samples = int(MERGE_GAP_SEC * fs)

    merged = []
    i = 0

    while i < len(raw_segments):
        start = raw_segments[i][0]
        end = raw_segments[i][1]
        while (end - start) < merge_samples and i + 1 < len(raw_segments):
            i += 1
            end = raw_segments[i][1]

        while i + 1 < len(raw_segments):
            next_start = raw_segments[i + 1][0]
            next_end = raw_segments[i + 1][1]

            if (next_end - next_start) < merge_samples:
                i += 1
                end = next_end
            else:
                break

        merged.append((start, end))
        i += 1

    return merged


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

    feat["duration"] = time_s[-1] - time_s[0]

    feat["ay_mean"] = np.mean(ay)
    feat["ay_std"] = np.std(ay)
    feat["ay_min"] = np.min(ay)
    feat["ay_max"] = np.max(ay)
    feat["ay_range"] = feat["ay_max"] - feat["ay_min"]

    feat["mean_ay"] = feat["ay_mean"]
    feat["std_ay"] = feat["ay_std"]
    feat["min_ay"] = feat["ay_min"]
    feat["max_ay"] = feat["ay_max"]
    feat["range_ay"] = feat["ay_range"]

    feat["gyro_var"] = np.var(gx)

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

        "ay_std",
        "ay_range",
        "std_ay",
        "range_ay",

        "gyro_var",

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

    parts = line.split(",")

    if len(parts) >= 7:
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
            pass

    match = re.search(
        r"\((\d+)\).*?"
        r"AX:([-+]?\d*\.?\d+)\s+"
        r"AY:([-+]?\d*\.?\d+)\s+"
        r"AZ:([-+]?\d*\.?\d+).*?"
        r"GX:([-+]?\d*\.?\d+)\s+"
        r"GY:([-+]?\d*\.?\d+)\s+"
        r"GZ:([-+]?\d*\.?\d+)",
        line
    )

    if match is None:
        return None

    try:
        t_ms = float(match.group(1))
        t_us = t_ms * 1000.0

        sample = {
            "t_us": t_us,
            "ax_g": float(match.group(2)),
            "ay_g": float(match.group(3)),
            "az_g": float(match.group(4)),
            "gx_dps": float(match.group(5)),
            "gy_dps": float(match.group(6)),
            "gz_dps": float(match.group(7)),
        }

        sample["time_s"] = sample["t_us"] / 1_000_000.0
        return sample

    except ValueError:
        return None


def get_completed_reps_from_buffer(buffer_df, last_classified_end_time):

    if len(buffer_df) < 20:
        return []

    signal = buffer_df[SIGNAL_COL].astype(float).values
    time_s = buffer_df["time_s"].values

    fs = estimate_sampling_rate(time_s)
    smooth = lowpass_filter_signal(signal, fs, cutoff_hz=CUTOFF_HZ)

    global_range = np.max(smooth) - np.min(smooth)

    if global_range < MIN_REP_RANGE:
        return []

    prom_thresh = PROM_FRACTION * global_range

    peaks, _ = find_peaks(
        smooth,
        distance=max(1, int(0.5 * fs)),
        prominence=prom_thresh
    )

    valleys, _ = find_peaks(
        -smooth,
        distance=max(1, int(0.5 * fs)),
        prominence=prom_thresh
    )

    if len(valleys) == 0 or (len(peaks) > 0 and valleys[0] > peaks[0]):
        valleys = np.insert(valleys, 0, 0)

    if len(peaks) > 0 and valleys[-1] < peaks[-1]:
        valleys = np.append(valleys, len(smooth) - 1)

    if len(peaks) == 0 or len(valleys) < 2:
        return []

    raw_segments = []

    for i in range(len(valleys) - 1):
        start = valleys[i]
        end = valleys[i + 1]

        has_peak = np.any((peaks > start) & (peaks < end))

        if not has_peak:
            continue

        raw_segments.append((start, end))

    merged_segments = merge_short_segments(raw_segments, fs)

    completed_reps = []

    for start, end in merged_segments:
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

        segment = smooth[start:end + 1]

        seg_range = np.max(segment) - np.min(segment)

        if seg_range < MIN_REP_RANGE:
            continue

        samples_after_end = len(buffer_df) - end

        if samples_after_end < int(REP_END_CONFIRMATION_SEC * fs):
            continue

        local_peak_offset = np.argmax(segment)
        peak_idx = start + local_peak_offset

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


def get_adaptive_threshold(rep_count, base_threshold):
    #stricter early in the set, less so later on
    if rep_count <= BASELINE_REPS:
        return 1.0

    if rep_count <= 6:
        return max(base_threshold + 0.20, 0.50)

    if rep_count <= 8:
        return max(base_threshold + 0.10, 0.40)

    return base_threshold


def adaptive_near_failure_decision(rep_count, current_prob, prob_history, base_threshold):
    threshold = get_adaptive_threshold(rep_count, base_threshold)

    if current_prob >= threshold:
        return 1, threshold

    if rep_count > BASELINE_REPS and len(prob_history) > 0:
        previous_prob = prob_history[-1]

        if current_prob >= base_threshold and previous_prob >= base_threshold:
            return 1, base_threshold

    return 0, threshold


def classify_rep(
    rep_df,
    rep_count,
    baseline_features,
    model,
    feature_columns,
    base_threshold,
    prob_history
):
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

    full_features["rep_num_in_set"] = rep_count

    X_new = make_model_input(full_features, feature_columns)

    proba_near_failure = model.predict_proba(X_new)[0, 1]

    if rep_count <= BASELINE_REPS:
        pred = 0
        threshold_used = 1.0
    else:
        pred, threshold_used = adaptive_near_failure_decision(
            rep_count=rep_count,
            current_prob=proba_near_failure,
            prob_history=prob_history,
            base_threshold=base_threshold
        )

    label = "NEAR FAILURE" if pred == 1 else "normal"

    print("\n" + "-" * 60)
    print(f"Rep {rep_count}")
    print(f"Prediction: {label}")
    print(f"Near-failure probability: {proba_near_failure:.3f}")
    print(f"Threshold used: {threshold_used:.2f}")
    print(f"Base threshold: {base_threshold:.2f}")
    print(f"Duration: {raw_features['duration']:.2f} sec")

    if rep_count <= BASELINE_REPS:
        print("Note: baseline rep, forced prediction to normal")

    print("-" * 60)

    return raw_features, pred, proba_near_failure, threshold_used


def should_skip_small_range_rep(rep_df, baseline_features):
    if len(baseline_features) < BASELINE_REPS:
        return False, 1.0

    sig = rep_df[SIGNAL_COL].values
    current_range = np.max(sig) - np.min(sig)

    baseline_ranges = []

    for feat in baseline_features:
        if SIGNAL_COL == "az_g" and "az_range" in feat:
            baseline_ranges.append(feat["az_range"])
        elif SIGNAL_COL == "ay_g" and "ay_range" in feat:
            baseline_ranges.append(feat["ay_range"])

    baseline_ranges = [
        x for x in baseline_ranges
        if not pd.isna(x)
    ]

    if len(baseline_ranges) == 0:
        return False, 1.0

    baseline_range = np.mean(baseline_ranges)
    range_ratio = current_range / (baseline_range + 1e-8)

    if range_ratio < MIN_RANGE_RATIO_ALLOWED:
        return True, range_ratio

    return False, range_ratio


def should_skip_poor_return_rep(rep_df, baseline_features):

    if len(baseline_features) < BASELINE_REPS:
        return False, 1.0

    sig = rep_df[SIGNAL_COL].values

    if len(sig) < 3:
        return False, 1.0

    peak_idx = np.argmax(sig)

    peak_value = sig[peak_idx]
    start_value = sig[0]
    end_value = sig[-1]

    rise = peak_value - start_value
    return_drop = peak_value - end_value

    if rise <= 0:
        return False, 1.0

    return_ratio = return_drop / (rise + 1e-8)

    if return_ratio < MIN_END_RETURN_RATIO:
        return True, return_ratio

    return False, return_ratio


def plot_detected_reps(
    buffer_df,
    detected_reps,
    title="Detected Reps",
    output_path="lateral_raise_segmentation.png"
):
    if len(buffer_df) == 0:
        print("No data to plot.")
        return

    time_s = buffer_df["time_s"].values
    signal = buffer_df[SIGNAL_COL].astype(float).values
    fs = estimate_sampling_rate(time_s)
    smooth = lowpass_filter_signal(signal, fs, cutoff_hz=CUTOFF_HZ)

    plt.figure(figsize=(12, 6))

    plt.plot(time_s, signal, alpha=0.35, label=f"Raw {SIGNAL_COL}")
    plt.plot(time_s, smooth, linewidth=2, label=f"Filtered {SIGNAL_COL}")

    counted_rep_num = 0

    for _, rep in enumerate(detected_reps, start=1):
        start_time = rep["start_time"]
        end_time = rep["end_time"]
        peak_time = rep.get("peak_time", None)

        prediction = rep.get("prediction", "unknown")
        probability = rep.get("probability", None)
        threshold_used = rep.get("threshold_used", None)
        range_ratio = rep.get("range_ratio", None)
        return_ratio = rep.get("return_ratio", None)

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

        if prediction == "skipped":
            label_text = "skip"
        else:
            counted_rep_num += 1
            label_text = str(counted_rep_num)

        if probability is not None:
            label_text += f"\nP={probability:.2f}"

        if threshold_used is not None:
            label_text += f"\nT={threshold_used:.2f}"

        if range_ratio is not None:
            label_text += f"\nR={range_ratio:.2f}"

        if return_ratio is not None:
            label_text += f"\nRet={return_ratio:.2f}"

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
    plt.ylabel(SIGNAL_COL)
    plt.legend()
    plt.grid(True)

    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved set visualization to {output_path}")

    plt.show()


def load_model_bundle():
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    threshold = bundle.get("near_failure_threshold", 0.30)

    print(f"Loaded model from {MODEL_PATH}")
    print(f"Using base near-failure threshold: {threshold}")

    return model, feature_columns, threshold


def run_live_serial(port, baud, threshold_override=None):
    try:
        import serial
    except ImportError:
        print("pyserial is not installed. Run: pip install pyserial")
        return

    model, feature_columns, base_threshold = load_model_bundle()

    if threshold_override is not None:
        base_threshold = threshold_override
        print(f"Overriding base threshold with: {base_threshold}")

    input("\nPress Enter to start the lateral raise set...")

    global stop_requested
    stop_requested = False

    stop_thread = threading.Thread(target=wait_for_stop_key)
    stop_thread.daemon = True
    stop_thread.start()

    buffer = []
    last_classified_end_time = None
    baseline_features = []
    prob_history = []
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

                skip_small_range_rep, range_ratio = should_skip_small_range_rep(
                    rep_info["rep_df"],
                    baseline_features
                )

                if skip_small_range_rep:
                    print(
                        f"Skipping partial/failed rep attempt: "
                        f"range ratio={range_ratio:.2f}"
                    )

                    detected_reps.append({
                        "start_time": rep_info["start_time"],
                        "end_time": rep_info["end_time"],
                        "peak_time": buffer_df["time_s"].iloc[rep_info["peak_idx"]],
                        "prediction": "skipped",
                        "probability": None,
                        "threshold_used": None,
                        "range_ratio": range_ratio,
                        "return_ratio": None,
                    })

                    last_classified_end_time = end_time
                    continue

                skip_poor_return_rep, return_ratio = should_skip_poor_return_rep(
                    rep_info["rep_df"],
                    baseline_features
                )

                if skip_poor_return_rep:
                    print(
                        f"Skipping partial/failed rep attempt: "
                        f"return ratio={return_ratio:.2f}"
                    )

                    detected_reps.append({
                        "start_time": rep_info["start_time"],
                        "end_time": rep_info["end_time"],
                        "peak_time": buffer_df["time_s"].iloc[rep_info["peak_idx"]],
                        "prediction": "skipped",
                        "probability": None,
                        "threshold_used": None,
                        "range_ratio": range_ratio,
                        "return_ratio": return_ratio,
                    })

                    last_classified_end_time = end_time
                    continue

                rep_count += 1

                raw_features, pred, proba, threshold_used = classify_rep(
                    rep_df=rep_info["rep_df"],
                    rep_count=rep_count,
                    baseline_features=baseline_features,
                    model=model,
                    feature_columns=feature_columns,
                    base_threshold=base_threshold,
                    prob_history=prob_history
                )

                prediction_label = "NEAR FAILURE" if pred == 1 else "normal"

                prob_history.append(proba)

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
                    "threshold_used": threshold_used,
                    "range_ratio": range_ratio,
                    "return_ratio": return_ratio,
                })

                last_classified_end_time = end_time

        final_buffer_df = pd.DataFrame(buffer)

        print("\nSet ended.")
        print(f"Total reps detected: {rep_count}")

        plot_detected_reps(
            final_buffer_df,
            detected_reps,
            title="Live Lateral Raise Segmentation Check",
            output_path="live_lateral_raise_segmentation.png"
        )


def run_from_csv(csv_path, delay=0.01, threshold_override=None):
    model, feature_columns, base_threshold = load_model_bundle()

    if threshold_override is not None:
        base_threshold = threshold_override
        print(f"Overriding base threshold with: {base_threshold}")

    raw = pd.read_csv(csv_path)

    missing = [col for col in REQUIRED_COLUMNS if col not in raw.columns]

    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    raw["time_s"] = raw["t_us"] / 1_000_000.0

    input("\nPress Enter to start replaying the lateral raise set...")

    buffer = []
    last_classified_end_time = None
    baseline_features = []
    prob_history = []
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

            skip_small_range_rep, range_ratio = should_skip_small_range_rep(
                rep_info["rep_df"],
                baseline_features
            )

            if skip_small_range_rep:
                print(
                    f"Skipping partial/failed rep attempt: "
                    f"range ratio={range_ratio:.2f}"
                )

                detected_reps.append({
                    "start_time": rep_info["start_time"],
                    "end_time": rep_info["end_time"],
                    "peak_time": buffer_df["time_s"].iloc[rep_info["peak_idx"]],
                    "prediction": "skipped",
                    "probability": None,
                    "threshold_used": None,
                    "range_ratio": range_ratio,
                    "return_ratio": None,
                })

                last_classified_end_time = end_time
                continue

            skip_poor_return_rep, return_ratio = should_skip_poor_return_rep(
                rep_info["rep_df"],
                baseline_features
            )

            if skip_poor_return_rep:
                print(
                    f"Skipping partial/failed rep attempt: "
                    f"return ratio={return_ratio:.2f}"
                )

                detected_reps.append({
                    "start_time": rep_info["start_time"],
                    "end_time": rep_info["end_time"],
                    "peak_time": buffer_df["time_s"].iloc[rep_info["peak_idx"]],
                    "prediction": "skipped",
                    "probability": None,
                    "threshold_used": None,
                    "range_ratio": range_ratio,
                    "return_ratio": return_ratio,
                })

                last_classified_end_time = end_time
                continue

            rep_count += 1

            raw_features, pred, proba, threshold_used = classify_rep(
                rep_df=rep_info["rep_df"],
                rep_count=rep_count,
                baseline_features=baseline_features,
                model=model,
                feature_columns=feature_columns,
                base_threshold=base_threshold,
                prob_history=prob_history
            )

            prediction_label = "NEAR FAILURE" if pred == 1 else "normal"

            prob_history.append(proba)

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
                "threshold_used": threshold_used,
                "range_ratio": range_ratio,
                "return_ratio": return_ratio,
            })

            last_classified_end_time = end_time

        time.sleep(delay)

    final_buffer_df = pd.DataFrame(buffer)

    print("\nReplay ended.")
    print(f"Total reps detected: {rep_count}")

    plot_detected_reps(
        final_buffer_df,
        detected_reps,
        title="Replay Lateral Raise Segmentation Check",
        output_path="replay_lateral_raise_segmentation.png"
    )


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
        help="Override the base near-failure threshold"
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
        print("python real_time_predictor_lateral_raise.py --serial /dev/cu.usbserial-210 --baud 115200")
        print()
        print("CSV replay example:")
        print("python real_time_predictor_lateral_raise.py --csv data/lateral_raise/set_01.csv")
        print()
        print("CSV replay with custom base threshold:")
        print("python real_time_predictor_lateral_raise.py --csv data/lateral_raise/set_01.csv --threshold 0.30")


if __name__ == "__main__":
    main()