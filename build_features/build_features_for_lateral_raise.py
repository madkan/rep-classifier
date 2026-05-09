import os
import glob
import numpy as np
import pandas as pd

INPUT_DATASET = "data/preprocessed/lateral_raises_dataset.csv"
OUTPUT_DATASET = "rep_dataset_lateral_raises_features.csv"

RAW_SET_FOLDER = "data/lateral_raise"

LABEL_COLUMN = "label"
GROUP_COLUMN = "set_id"

#only using last 10 sets
SET_ID_OFFSET = 11


def find_set_file(set_id):
    file_num = int(set_id) + SET_ID_OFFSET

    patterns = [
        f"*{file_num:02d}*.csv",
        f"*{file_num}.csv",
        f"set_{file_num:02d}.csv",
        f"set{file_num:02d}.csv",
        f"{file_num:02d}.csv",
    ]

    for pattern in patterns:
        matches = glob.glob(os.path.join(RAW_SET_FOLDER, pattern))
        if len(matches) > 0:
            return matches[0]

    return None


def find_time_column(raw):
    possible_cols = ["time", "timestamp", "t", "t_s", "seconds", "t_us"]

    for col in possible_cols:
        if col in raw.columns:
            return col

    raise ValueError(
        "Could not find a time column. Expected one of: "
        "time, timestamp, t, t_s, seconds, t_us"
    )


def get_time_seconds(raw):
    time_col = find_time_column(raw)
    t = raw[time_col].astype(float).values

    if time_col == "t_us" or np.nanmax(t) > 100000:
        t = t / 1_000_000.0

    return t


def get_sensor_columns(raw):
    required = ["ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps"]

    missing = [col for col in required if col not in raw.columns]

    if len(missing) > 0:
        raise ValueError(f"Missing required raw sensor columns: {missing}")

    return required


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


def estimate_sampling_rate(t):
    if len(t) < 2:
        return 100

    diffs = np.diff(t)
    diffs = diffs[diffs > 0]

    if len(diffs) == 0:
        return 100

    median_dt = np.median(diffs)

    if median_dt == 0:
        return 100

    return 1.0 / median_dt


def slice_rep(raw, start_time, end_time, set_first_start_time=None):
    t = get_time_seconds(raw)

    mask = (t >= start_time) & (t <= end_time)
    rep = raw.loc[mask].copy()

    if len(rep) > 0:
        return rep

    if set_first_start_time is not None:
        rel_start = start_time - set_first_start_time
        rel_end = end_time - set_first_start_time

        t_rel = t - t[0]
        mask = (t_rel >= rel_start) & (t_rel <= rel_end)
        rep = raw.loc[mask].copy()

    return rep


def extract_features_from_rep_segment(rep):
    get_sensor_columns(rep)

    t = get_time_seconds(rep)
    fs = estimate_sampling_rate(t)

    ax = rep["ax_g"].astype(float).values
    ay = rep["ay_g"].astype(float).values
    az = rep["az_g"].astype(float).values

    gx = rep["gx_dps"].astype(float).values
    gy = rep["gy_dps"].astype(float).values
    gz = rep["gz_dps"].astype(float).values

    accel_mag = signal_magnitude(ax, ay, az)
    gyro_mag = signal_magnitude(gx, gy, gz)

    features = {}

    #lateral raise dataset does NOT already have these, so adding them
    features["ay_mean"] = np.mean(ay)
    features["ay_std"] = np.std(ay)
    features["ay_min"] = np.min(ay)
    features["ay_max"] = np.max(ay)
    features["ay_range"] = features["ay_max"] - features["ay_min"]

    features["mean_ay"] = features["ay_mean"]
    features["std_ay"] = features["ay_std"]
    features["min_ay"] = features["ay_min"]
    features["max_ay"] = features["ay_max"]
    features["range_ay"] = features["ay_range"]

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
        features.update(time_domain_features(signal, name))

    features.update(jerk_features(accel_mag, fs, "accel_mag"))
    features.update(jerk_features(gyro_mag, fs, "gyro_mag"))

    features.update(frequency_domain_features(accel_mag, fs, "accel_mag"))
    features.update(frequency_domain_features(gyro_mag, fs, "gyro_mag"))

    features["segment_num_samples"] = len(rep)
    features["segment_sampling_rate"] = fs

    return features


def empty_extracted_features():
    dummy = {}

    ay_features = [
        "ay_mean",
        "ay_std",
        "ay_min",
        "ay_max",
        "ay_range",
        "mean_ay",
        "std_ay",
        "min_ay",
        "max_ay",
        "range_ay",
    ]

    for col in ay_features:
        dummy[col] = np.nan

    signal_names = [
        "ax", "az", "gx", "gy", "gz", "accel_mag", "gyro_mag"
    ]

    for name in signal_names:
        dummy.update(time_domain_features([], name))

    dummy.update(jerk_features([], 100, "accel_mag"))
    dummy.update(jerk_features([], 100, "gyro_mag"))

    dummy.update(frequency_domain_features([], 100, "accel_mag"))
    dummy.update(frequency_domain_features([], 100, "gyro_mag"))

    dummy["segment_num_samples"] = np.nan
    dummy["segment_sampling_rate"] = np.nan

    return dummy


def add_raw_timeseries_features(df):
    df = df.copy()

    df["rep_num_in_set"] = df.groupby(GROUP_COLUMN).cumcount() + 1

    feature_rows = []
    raw_cache = {}

    first_start_by_set = df.groupby(GROUP_COLUMN)["start_time"].min().to_dict()

    for _, row in df.iterrows():
        set_id = int(row[GROUP_COLUMN])

        if set_id not in raw_cache:
            filepath = find_set_file(set_id)

            if filepath is None:
                print(f"Missing raw set file for set_id={set_id}")
                raw_cache[set_id] = None
            else:
                print(f"Loading set_id={set_id} from {filepath}")
                raw_cache[set_id] = pd.read_csv(filepath)

        raw = raw_cache[set_id]

        if raw is None:
            feature_rows.append(empty_extracted_features())
            continue

        rep = slice_rep(
            raw=raw,
            start_time=row["start_time"],
            end_time=row["end_time"],
            set_first_start_time=first_start_by_set[set_id]
        )

        if len(rep) < 8:
            print(
                f"Warning: short/missing segment for "
                f"set_id={set_id}, rep_id={row['rep_id']}, samples={len(rep)}"
            )
            feature_rows.append(empty_extracted_features())
        else:
            feature_rows.append(extract_features_from_rep_segment(rep))

    features_df = pd.DataFrame(feature_rows)

    duplicate_cols = [col for col in features_df.columns if col in df.columns]
    if duplicate_cols:
        print("\nDropping duplicate generated columns:")
        for col in duplicate_cols:
            print("-", col)
        features_df = features_df.drop(columns=duplicate_cols)

    return pd.concat(
        [df.reset_index(drop=True), features_df.reset_index(drop=True)],
        axis=1
    )


def add_baseline_features(df, baseline_reps=3):
    df = df.copy()

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

    baseline_source_columns = [
        col for col in baseline_source_columns if col in df.columns
    ]

    for feature in baseline_source_columns:
        baseline_col = f"baseline_{feature}"
        ratio_col = f"{feature}_ratio_to_baseline"
        diff_col = f"{feature}_diff_from_baseline"

        df[baseline_col] = df.groupby(GROUP_COLUMN)[feature].transform(
            lambda x: x.iloc[:baseline_reps].mean()
        )

        df[ratio_col] = df[feature] / (df[baseline_col] + 1e-8)
        df[diff_col] = df[feature] - df[baseline_col]

    return df


def main():
    df = pd.read_csv(INPUT_DATASET)

    print("Original dataset shape:", df.shape)
    print("Original columns:")
    print(df.columns.tolist())

    df = add_raw_timeseries_features(df)
    df = add_baseline_features(df, baseline_reps=3)

    df.to_csv(OUTPUT_DATASET, index=False)

    print("\nSaved enriched dataset to:", OUTPUT_DATASET)
    print("New dataset shape:", df.shape)

    print("\nNew feature columns added:")
    original_cols = pd.read_csv(INPUT_DATASET, nrows=1).columns.tolist()
    new_cols = [col for col in df.columns if col not in original_cols]
    for col in new_cols:
        print("-", col)


if __name__ == "__main__":
    main()