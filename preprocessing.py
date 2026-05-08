#1 LOAD THE DATASETS
import pandas as pd
import numpy as np
import glob
from scipy.signal import find_peaks, butter, filtfilt
from scipy.ndimage import uniform_filter1d

def load_all_csvs(path_pattern):
    files = sorted(glob.glob(path_pattern))
    datasets = []

    for i, f in enumerate(files):
        df = pd.read_csv(f)
        df["set_id"] = i
        df["time_s"] = df["t_us"] / 1e6
        datasets.append(df)
    return datasets, files

#2 EXERCISE CONFIG

EXERCISES = {
    "bicep_curl": {
        "signal_col": "ay_g",
        "method": "peak_valley",
        "cutoff_hz": 3.0,
        "min_rep_duration": 1.0,
        "max_rep_duration": 8.0,
        "min_rep_range": 0.8,
        "prom_fraction": 0.20,
    },
    # "lateral_raise": {
    #     "signal_col": "az_g",
    #     "method": "period",
    #     "cutoff_hz": 5.0,
    #     "min_rep_duration": 1.5,
    #     "max_rep_duration": 8.0,
    # },
    "lateral_raise": {
        "signal_col": "az_g",
        "method": "peak_valley_merge",
        "cutoff_hz": 5.0,
        "min_rep_duration": 1.5,
        "max_rep_duration": 8.0,
        "min_rep_range": 0.3,       # lower — catches weaker reps
        "prom_fraction": 0.5,      # lower — catches smaller peaks
        "merge_gap": 0.8,           # merge segments shorter than this (seconds)
    },
}

#3 REP SEGMENTATION

def segment_reps(df, exercise="bicep_curl"):
    config = EXERCISES[exercise]
    signal_col = config["signal_col"]
    signal = df[signal_col].values
    time   = df["time_s"].values
    dt     = np.median(np.diff(time))
    fs     = 1.0 / dt

    # Smooth
    cutoff_hz   = config["cutoff_hz"]
    nyq         = fs / 2.0
    norm_cutoff = min(cutoff_hz / nyq, 0.99)
    b, a   = butter(N=4, Wn=norm_cutoff, btype='low')
    smooth = filtfilt(b, a, signal)

    # peak_valley for BC / peak_valley_merge for LR
    if config["method"] == "peak_valley":
        reps, peaks, valleys, rep_indices = _segment_peak_valley(df, smooth, time, fs, config)
    elif config["method"] == "peak_valley_merge":
        reps, peaks, valleys, rep_indices = _segment_peak_valley_merge(df, smooth, time, fs, config)

    print(f"  Set {df['set_id'].iloc[0]}: {len(reps)} reps ({exercise})")
    return reps, peaks, valleys, smooth, rep_indices

def _segment_peak_valley(df, smooth, time, fs, config):
    """Peak/valley segmentation — for exercises with one clear peak per rep."""
    min_samples   = int(config["min_rep_duration"] * fs)
    global_range  = np.max(smooth) - np.min(smooth)
    prom_thresh   = config["prom_fraction"] * global_range

    peaks, _   = find_peaks( smooth, distance=min_samples, prominence=prom_thresh)
    valleys, _ = find_peaks(-smooth, distance=min_samples, prominence=prom_thresh)

    if len(valleys) == 0 or (len(peaks) > 0 and valleys[0] > peaks[0]):
        valleys = np.insert(valleys, 0, 0)
    if len(peaks) > 0 and valleys[-1] < peaks[-1]:
        valleys = np.append(valleys, len(smooth) - 1)

    reps = []
    rep_indices = []
    dt = np.median(np.diff(time))

    for i in range(len(valleys) - 1):
        start = valleys[i]
        end   = valleys[i + 1]

        if not np.any((peaks > start) & (peaks < end)):
            continue
        duration = time[end] - time[start]
        if duration < config["min_rep_duration"]:
            continue
        seg_range = np.max(smooth[start:end]) - np.min(smooth[start:end])
        if seg_range < config["min_rep_range"]:
            continue

        reps.append(df.iloc[start:end].copy())
        rep_indices.append((start, end))

    return reps, peaks, valleys, rep_indices

def _segment_peak_valley_merge(df, smooth, time, fs, config):
    """
    Peak/valley + merge short segments.
    Handles exercises where one rep has multiple sub-peaks.
    """
    min_samples   = int(config["min_rep_duration"] * fs)
    global_range  = np.max(smooth) - np.min(smooth)
    prom_thresh   = config["prom_fraction"] * global_range

    peaks, _   = find_peaks( smooth, distance=int(0.5 * fs), prominence=prom_thresh)
    valleys, _ = find_peaks(-smooth, distance=int(0.5 * fs), prominence=prom_thresh)

    if len(valleys) == 0 or (len(peaks) > 0 and valleys[0] > peaks[0]):
        valleys = np.insert(valleys, 0, 0)
    if len(peaks) > 0 and valleys[-1] < peaks[-1]:
        valleys = np.append(valleys, len(smooth) - 1)

    # ── Step 1: Build raw segments (valley to valley) ──
    raw_segments = []
    for i in range(len(valleys) - 1):
        start = valleys[i]
        end   = valleys[i + 1]
        if not np.any((peaks > start) & (peaks < end)):
            continue
        raw_segments.append((start, end))

    # ── Step 2: Merge short segments into their neighbor ──
    # If a segment is too short, it's a sub-peak, merge it with the next one
    merge_samples = int(config["merge_gap"] * fs)
    merged = []
    i = 0

    while i < len(raw_segments):
        start = raw_segments[i][0]
        end   = raw_segments[i][1]

        # Keep merging while the current segment is too short
        while (end - start) < merge_samples and i + 1 < len(raw_segments):
            i += 1
            end = raw_segments[i][1]

        # Also check: if NEXT segment is too short, absorb it
        while i + 1 < len(raw_segments):
            next_start = raw_segments[i + 1][0]
            next_end   = raw_segments[i + 1][1]
            if (next_end - next_start) < merge_samples:
                i += 1
                end = next_end
            else:
                break

        merged.append((start, end))
        i += 1

    # ── Step 3: Filter merged segments ──
    reps = []
    rep_indices = []

    for start, end in merged:
        duration = time[end] - time[start]
        if duration < config["min_rep_duration"]:
            continue
        seg_range = np.max(smooth[start:end]) - np.min(smooth[start:end])
        if seg_range < config["min_rep_range"]:
            continue

        reps.append(df.iloc[start:end].copy())
        rep_indices.append((start, end))

    return reps, peaks, valleys, rep_indices

#4 FEATURE EXTRACTION

def extract_features(rep_df, exercise="bicep_curl"):
    config = EXERCISES[exercise]
    signal_col = config["signal_col"]

    time = rep_df["time_s"].values
    sig  = rep_df[signal_col].values
    dt   = np.median(np.diff(time))

    feat = {}
    # Common features
    feat["duration"]  = time[-1] - time[0]
    feat["max_sig"]   = np.max(sig)
    feat["min_sig"]   = np.min(sig)
    feat["range_sig"] = np.ptp(sig)
    feat["mean_sig"]  = np.mean(sig)
    feat["std_sig"]   = np.std(sig)

    # velocity proxy (gyro)
    feat["gyro_var"] = np.var(rep_df["gx_dps"].values)

    return feat

#5 LABELING

def label_reps(num_reps, near_failure_count=3):
    """
    Labels:
        0 = normal
        1 = near-failure (last N reps of the set)

    Since the set ends at failure, the last few clean reps
    ARE the near-failure reps. No need to find the failure rep.
    """
    labels = []
    for i in range(num_reps):
        if i >= num_reps - near_failure_count:
            labels.append(1)  # near-failure
        else:
            labels.append(0)  # normal

    return labels

#6 BUILDING THE DATASET

def build_dataset(datasets, exercise="bicep_curl"):
    all_features = []
    for df in datasets:
        reps, peaks, valleys, smooth, rep_indices = segment_reps(df, exercise)

        labels = label_reps(len(reps))

        for i, rep in enumerate(reps):
            feat = extract_features(rep, exercise)
            feat["label"]      = labels[i]
            feat["set_id"]     = df["set_id"].iloc[0]
            feat["rep_id"]     = i
            feat["start_time"] = rep["time_s"].iloc[0]
            feat["end_time"]   = rep["time_s"].iloc[-1]

            all_features.append(feat)
    feature_df = pd.DataFrame(all_features)
    return feature_df

#7 VISUALIZING THE DATASET TO VALIDATE THE REP SEGMENTATION
import matplotlib.pyplot as plt

def get_labels(n):
    labels = []
    for i in range(n):
        if i >= n - 3:
            labels.append(1)  # near-failure
        else:
            labels.append(0)  # normal
    return labels

def plot_segmentation(df, peaks, valleys, smooth, rep_indices, exercise="bicep_curl", title=""):
    config = EXERCISES[exercise]
    signal_col = config["signal_col"]

    plt.figure(figsize=(14, 5))
    plt.plot(df["time_s"], smooth, label=f"Smoothed {signal_col}")
    # Peaks
    plt.scatter(df["time_s"].iloc[peaks], smooth[peaks], color="red", label="Peaks")

    # Valleys
    plt.scatter(df["time_s"].iloc[valleys], smooth[valleys], color="green", label="Valleys")
    label_colors = {0: "green", 1: "orange"}
    labels = get_labels(len(rep_indices))

    # Rep boundaries
    for i, (start, end) in enumerate(rep_indices):
        color = label_colors[labels[i]]
        plt.axvspan(df["time_s"].iloc[start], df["time_s"].iloc[end], alpha=0.2, color=color)
        mid = (start + end) // 2
        plt.text(df["time_s"].iloc[mid], smooth[mid], str(i + 1), color="black", fontsize=9, ha="center")

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel(signal_col)
    plt.legend()
    plt.grid()
    plt.show()

#8 USAGE

bicep_curls="bicep_curl"
lateral_raises="lateral_raise"
bicep_curl_dir_path = "./drive/MyDrive/528_Data/bicep_curls/*.csv"
lateral_raise_dir_path = "./drive/MyDrive/528_Data/lateral_raise/*.csv"

# Pass parameter acc to the exercise you want to build the dataset for
datasets, files = load_all_csvs(lateral_raise_dir_path)

# check any dataset by changing index
df = datasets[9]
reps, peaks, valleys, smooth, rep_indices = segment_reps(df, exercise=lateral_raises)
plot_segmentation(df, peaks, valleys, smooth, rep_indices, exercise=lateral_raises, title="Lateral Raise — Rep Segmentation")

# save the dataset
# feature_df.to_csv("./data/preprocessed/rep_dataset_bicep_curls.csv", index=False)