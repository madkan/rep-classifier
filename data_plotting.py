import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Path to your file in the data folder
filename = "data/bicep_curl_set_18.csv"   # or "data/up_00.csv"

df = pd.read_csv(filename)

df["time_s"] = (df["t_us"] - df["t_us"].iloc[0]) / 1_000_000

df["acc_mag_g"] = np.sqrt(df["ax_g"]**2 + df["ay_g"]**2 + df["az_g"]**2)
df["gyr_mag_dps"] = np.sqrt(df["gx_dps"]**2 + df["gy_dps"]**2 + df["gz_dps"]**2)

plt.figure(figsize=(12, 5))
plt.plot(df["time_s"], df["ax_g"], label="ax_g")
plt.plot(df["time_s"], df["ay_g"], label="ay_g")
plt.plot(df["time_s"], df["az_g"], label="az_g")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (g)")
plt.title("Accelerometer Axes")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(df["time_s"], df["acc_mag_g"], label="acc magnitude")
plt.xlabel("Time (s)")
plt.ylabel("Acceleration Magnitude (g)")
plt.title("Accelerometer Magnitude")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(df["time_s"], df["gx_dps"], label="gx_dps")
plt.plot(df["time_s"], df["gy_dps"], label="gy_dps")
plt.plot(df["time_s"], df["gz_dps"], label="gz_dps")
plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (deg/s)")
plt.title("Gyroscope Axes")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 5))
plt.plot(df["time_s"], df["gyr_mag_dps"], label="gyr magnitude")
plt.xlabel("Time (s)")
plt.ylabel("Gyroscope Magnitude (deg/s)")
plt.title("Gyroscope Magnitude")
plt.legend()
plt.grid(True)
plt.show()