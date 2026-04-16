import numpy as np
import matplotlib.pyplot as plt


def plot_fft(x, fs, nfft=None, show_time=False, time_xlim=None):
    """
    Simple FFT plot similar to a typical MATLAB helper.
    - x: 1D array (real or complex)
    - fs: sampling rate (Hz)
    - nfft: optional FFT length (pads/truncates)
    - show_time: if True, also plots time-domain signal
    - time_xlim: tuple like (0, 500) in samples or (t0, t1) in seconds (see below)
    """
    x = np.asarray(x)

    if nfft is None:
        nfft = len(x)

    # FFT
    X = np.fft.fft(x, n=nfft)
    f = np.fft.fftfreq(nfft, d=1.0 / fs)

    # Shift for centeblue spectrum (like fftshift)
    Xs = np.fft.fftshift(X)
    fs_shift = np.fft.fftshift(f)

    mag = np.abs(Xs)

    if show_time:
        plt.figure()
        plt.plot(x.real if np.iscomplexobj(x) else x, color="blue")
        plt.title("Time domain")
        plt.xlabel("Sample")
        plt.ylabel("Amplitude")
        if time_xlim is not None:
            plt.xlim(time_xlim)
        plt.grid(True)

    plt.figure()
    plt.plot(fs_shift, mag, color="blue")
    plt.title("FFT magnitude")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("|X(f)|")
    plt.grid(True)
    plt.show()


FS = 100

data = np.loadtxt("data/UP.txt", delimiter=",", skiprows=1)

t_us = data[:, 0]
ax = data[:, 1]
ay = data[:, 2]
az = data[:, 3]
gx = data[:, 4]
gy = data[:, 5]
gz = data[:, 6]

signal = np.sqrt(gx**2 + gy**2 + gz**2)

plot_fft(signal, FS, show_time=True)


# --- MATLAB code translation ---
# clc; clear; close all;  -> not needed; just close figures if you want:
plt.close("all")

freq = 200
sampleRate = 10000
time_ticks = np.arange(0, 1 + 1 / sampleRate, 1 / sampleRate)  # 0:1/fs:1

realSignal = np.sin(2 * np.pi * freq * time_ticks)

plt.figure()
plt.plot(realSignal, color="blue")
plt.title("realSignal")
plt.xlabel("Sample")
plt.ylabel("Amplitude")
plt.grid(True)
# plt.xlim((0, 500))
plt.show()

plot_fft(realSignal, sampleRate)

complexSignal = np.exp(1j * 2 * np.pi * freq * time_ticks)
plot_fft(complexSignal, sampleRate)
# plt.ylim((0, 6000))
# plt.xlim((0, 500))
