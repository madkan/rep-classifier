#1 LOAD THE DATASETS
import pandas as pd
import numpy as np
import glob

def load_all_csvs(path_pattern):
    files = sorted(glob.glob(path_pattern))
    datasets = []

    for i, f in enumerate(files):
        df = pd.read_csv(f)
        df["set_id"] = i
        df["time_s"] = df["t_us"] / 1e6
        datasets.append(df)
    return datasets, files

#2 REP SEGMENTATION
from scipy.signal import find_peaks, butter, filtfilt

def segment_reps(df, min_rep_duration=1.0, min_rep_range=0.8):
    signal = df["ay_g"].values
    time   = df["time_s"].values

    # Step 1: Estimate sampling rate from the data
    dt = np.median(np.diff(time))
    fs = 1.0 / dt

    # Step 2: Low-pass Butterworth filter (replaces savgol_filter)
    cutoff_hz   = 3.0
    nyq         = fs / 2.0
    norm_cutoff = min(cutoff_hz / nyq, 0.99)
    b, a   = butter(N=4, Wn=norm_cutoff, btype='low')
    smooth = filtfilt(b, a, signal)

    # Step 3: Peak/valley detection with physical constraints
    min_samples   = int(min_rep_duration * fs)
    global_range  = np.max(smooth) - np.min(smooth)
    prom_thresh   = 0.20 * global_range

    peaks, _   = find_peaks( smooth, distance=min_samples, prominence=prom_thresh)
    valleys, _ = find_peaks(-smooth, distance=min_samples, prominence=prom_thresh)

    # Ensure valley before first peak
    if len(valleys) == 0 or (len(peaks) > 0 and valleys[0] > peaks[0]):
        valleys = np.insert(valleys, 0, 0)

    # Ensure valley after last peak
    if len(peaks) > 0 and valleys[-1] < peaks[-1]:
        valleys = np.append(valleys, len(signal) - 1)

    # Step 4: Build rep segments with filtering
    reps        = []
    rep_indices = []

    for i in range(len(valleys) - 1):
        start = valleys[i]
        end   = valleys[i + 1]

        # Must contain at least one peak
        if not np.any((peaks > start) & (peaks < end)):
            continue

        # Must meet minimum duration
        duration = time[end] - time[start]
        if duration < min_rep_duration:
            continue

        # Must meet minimum signal range (kills noise)
        seg_range = np.max(smooth[start:end]) - np.min(smooth[start:end])
        if seg_range < min_rep_range:
            continue

        reps.append(df.iloc[start:end].copy())
        rep_indices.append((start, end))

    print(f"  Set {df['set_id'].iloc[0]}: {len(reps)} valid reps "
          f"(peaks={len(peaks)}, valleys={len(valleys)})")

    return reps, peaks, valleys, smooth, rep_indices

#3 FEATURE EXTRACTION
def extract_features(rep_df):
    ay = rep_df["ay_g"].values
    time = rep_df["time_s"].values

    feat = {}

    feat["duration"] = time[-1] - time[0]
    feat["max_ay"] = np.max(ay)
    feat["min_ay"] = np.min(ay)
    feat["range_ay"] = feat["max_ay"] - feat["min_ay"]
    feat["mean_ay"] = np.mean(ay)
    feat["std_ay"] = np.std(ay)

    # velocity proxy (gyro)
    feat["gyro_var"] = np.var(rep_df["gx_dps"].values)

    return feat

#4 LABELING (binary: normal vs near-failure)
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

#5 BUILDING THE DATASET

def build_dataset(datasets):
    all_features = []
    for df in datasets:
        reps, peaks, valleys, smooth, rep_indices = segment_reps(df)
        labels = label_reps(len(reps))

        for i, rep in enumerate(reps):
            feat = extract_features(rep)
            feat["label"] = labels[i]
            feat["set_id"] = df["set_id"].iloc[0]
            feat["rep_id"] = i
            # adding start and end times of each rep
            feat["start_time"] = rep["time_s"].iloc[0]
            feat["end_time"]   = rep["time_s"].iloc[-1]

            all_features.append(feat)
    feature_df = pd.DataFrame(all_features)
    return feature_df

#6 VISUALIZING THE DATASET TO VALIDATE THE REP SEGMENTATION
import matplotlib.pyplot as plt
# def get_labels(n):
#     labels = []
#     for i in range(n):
#         if i == n - 1:
#             labels.append(2)
#         elif i >= n - 3:
#             labels.append(1)
#         else:
#             labels.append(0)
#     return labels

def get_labels(n):
    labels = []
    for i in range(n):
        if i >= n - 3:
            labels.append(1)  # near-failure
        else:
            labels.append(0)  # normal
    return labels


def plot_segmentation(df, peaks, valleys, smooth, rep_indices, title=""):
    plt.figure(figsize=(12, 6))

    plt.plot(df["time_s"], smooth, label="Smoothed ay_g")

    # Peaks
    plt.scatter(df["time_s"].iloc[peaks], smooth[peaks],
                color="red", label="Peaks")

    # Valleys
    plt.scatter(df["time_s"].iloc[valleys], smooth[valleys],
                color="green", label="Valleys")

    # Label Colors
    label_colors = {
        0: "green",   # normal
        1: "yellow",  # near failure
        # 2: "orange"   # failure
    }
    labels = get_labels(len(rep_indices))

    # Rep boundaries
    for i, (start, end) in enumerate(rep_indices):

        label = labels[i]
        color = label_colors[label]

        # shaded rep region
        plt.axvspan(
            df["time_s"].iloc[start],
            df["time_s"].iloc[end],
            alpha=0.25,
            color=color
        )
        # rep number
        mid = (start + end) // 2
        plt.text(
            df["time_s"].iloc[mid],
            smooth[mid],
            str(i + 1),
            color="black",
            fontsize=9,
            ha="center"
        )
    # for (start, end) in rep_indices:
    #     plt.axvspan(df["time_s"].iloc[start],
    #                 df["time_s"].iloc[end],
    #                 alpha=0.1, color="orange")
    #     for i, (start, end) in enumerate(rep_indices):
    #         mid = (start + end) // 2
    #         plt.text(df["time_s"].iloc[mid],smooth[mid],str(i+1),color="black",fontsize=9,ha='center')
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("ay_g")
    plt.legend()
    plt.grid()
    plt.show()

#7 FUNCTION CALL TO LOAD THE CSV FILES
datasets, files = load_all_csvs("./data/raw/bicep_curls/*.csv")

#8 FUNCTION CALL TO BUILD THE DATASET
feature_df = build_dataset(datasets)

#9 CHECKING REP BOUNDARIES
df = datasets[7] # change index number to check the rep boundaries in a particular set
reps, peaks, valleys, smooth, rep_indices = segment_reps(df)
plot_segmentation(df, peaks, valleys, smooth, rep_indices,title="Segmentation Check")


 #10 EXPORTING DATASET AS CSV FILE
# This dataset can be used for building a simple random forest classifer.
# For LSTM/Deep Learning models, need to do feature extraction differently to feed time series data as input

feature_df.to_csv("./data/preprocessed/rep_dataset_bicep_curls_v1.csv", index=False)