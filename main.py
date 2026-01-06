"""Very cool project :yesyes:"""

import asyncio
from collections import deque
from typing import Any, Dict, Deque, Tuple, TypedDict, cast

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray

import httpx
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks # type: ignore

URL = "http://192.168.20.59:8080"
MAX_POINTS = 250
LOWCUT = 1  # 60 BPM
HIGHCUT = 4  # 240 BPM
POLL_INTERVAL = 0.1
PROMINENCE_MULTIPLIER = 1
MIN_PROMINENCE = 0.0005


async def fetch_data(client: httpx.AsyncClient, last_time: float) -> Dict[str, Any]:
    """Fetch data from phyphox"""
    params: Dict[str, str | float] = {"accZ": f"{last_time}|acc_time", "acc_time": last_time}
    response = await client.get("/get", params=params)
    response.raise_for_status()
    return response.json()


def butter_bandpass_filter(
    data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4
) -> np.ndarray:
    """Apply a Butterworth bandpass filter to the data"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band") # type: ignore
    if data.size <= 27:
        # Not enough samples for filtfilt padding yet; let caller skip metrics until buffer grows.
        return np.array([])
    return filtfilt(b, a, data)

class Metrics(TypedDict):
    filtered: np.ndarray
    peaks: np.ndarray
    bpm_times: np.ndarray
    bpm: np.ndarray
    fs: float | None


def compute_metrics(times: np.ndarray, signal: np.ndarray) -> Metrics:
    metrics: Metrics = {
        "filtered": np.array([]),
        "peaks": np.array([], dtype=int),
        "bpm_times": np.array([]),
        "bpm": np.array([]),
        "fs": None,
    }

    if len(times) < 10:
        return metrics

    dt = np.diff(times)
    if dt.size == 0 or not np.all(dt > 0):
        return metrics

    fs_value = float(1 / np.mean(dt))
    if fs_value <= 0:
        return metrics

    filtered = butter_bandpass_filter(signal, LOWCUT, HIGHCUT, float(fs_value))
    if not filtered.size:
        return metrics
    distance = max(int(fs_value / HIGHCUT), 1)
    baseline = np.median(filtered)
    mad = np.median(np.abs(filtered - baseline))
    noise_level = max(mad * 1.4826, MIN_PROMINENCE)
    min_prominence = PROMINENCE_MULTIPLIER * noise_level
    peaks_raw, _properties = cast(
        Tuple[np.ndarray, dict[str, Any]],
        find_peaks(
            filtered,
            distance=distance,
            height=baseline + min_prominence,
            prominence=min_prominence,
        ),
    )

    peaks_arr = np.asarray(peaks_raw, dtype=int)
    metrics["filtered"] = filtered
    metrics["peaks"] = peaks_arr
    metrics["fs"] = fs_value

    if peaks_arr.size >= 2:
        peak_times = times[peaks_arr]
        rr = np.diff(peak_times)
        valid = rr > 0
        if valid.any():
            rr = rr[valid]
            bpm = 60 / rr
            bpm_times = peak_times[1:][valid]
            metrics["bpm"] = bpm
            metrics["bpm_times"] = bpm_times
            return metrics

    metrics["bpm"] = np.array([0.0])
    metrics["bpm_times"] = np.array([times[-1]])
    return metrics


def init_plots() -> Tuple[Figure, NDArray[np.object_], Line2D, Line2D, Line2D, Line2D]:
    """Init the plots"""
    plt.ion() # type: ignore
    fig, axes = plt.subplots(3, 1, figsize=(12, 10)) # type: ignore

    (raw_line,) = axes[0].plot([], [], label="Raw accZ", color="gray", alpha=0.6)
    axes[0].set_title("Raw Accelerometer Data")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Acceleration (m/s^2)")
    axes[0].grid(True)
    axes[0].legend()

    (filtered_line,) = axes[1].plot([], [], label="Filtered Signal (0.7-3.5 Hz)")
    (peak_marks,) = axes[1].plot([], [], "x", label="Detected Peaks", color="red")
    axes[1].set_title("Filtered Signal & Detected Peaks")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True)
    axes[1].legend()

    (hr_line,) = axes[2].plot(
        [], [], label="Heart Rate", color="red", marker="o", markersize=3
    )
    axes[2].set_title("Calculated Heart Rate")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("BPM")
    axes[2].set_ylim(40, 180)
    axes[2].grid(True)
    axes[2].legend()

    fig.tight_layout()
    plt.show(block=False) # type: ignore

    return fig, axes, raw_line, filtered_line, peak_marks, hr_line


def update_plots(
    fig: Figure,
    axes: NDArray[np.object_],
    raw_line: Line2D,
    filtered_line: Line2D,
    peak_marks: Line2D,
    hr_line: Line2D,
    times: np.ndarray,
    signal: np.ndarray,
    metrics: Metrics,
) -> None:
    """Update the plots with new data"""
    if len(times) < 2:
        return

    raw_line.set_data(times, signal)
    axes[0].set_xlim(times[0], times[-1])
    y_pad_raw = max(np.ptp(signal) * 0.1, 0.1)
    axes[0].set_ylim(signal.min() - y_pad_raw, signal.max() + y_pad_raw)

    filtered = metrics.get("filtered", np.array([]))
    peaks = metrics.get("peaks", np.array([], dtype=int))
    if filtered.size:
        filtered_line.set_data(times, filtered)
        peak_marks.set_data(times[peaks], filtered[peaks])
        axes[1].set_xlim(times[0], times[-1])
        y_pad_filtered = max(np.ptp(filtered) * 0.1, 0.1)
        axes[1].set_ylim(
            filtered.min() - y_pad_filtered, filtered.max() + y_pad_filtered
        )
    else:
        filtered_line.set_data([], [])
        peak_marks.set_data([], [])

    bpm_times = metrics.get("bpm_times", np.array([]))
    bpm = metrics.get("bpm", np.array([]))
    if bpm_times.size and bpm.size:
        hr_line.set_data(bpm_times, bpm)
        axes[2].set_xlim(bpm_times[0], bpm_times[-1])
    else:
        hr_line.set_data([], [])
        axes[2].set_xlim(times[0], times[-1])

    fig.canvas.draw() # type: ignore
    fig.canvas.flush_events()
    plt.pause(0.001)


async def main() -> None:
    """Main entrypoint"""
    time_buffer: Deque[float] = deque(maxlen=MAX_POINTS)
    accel_buffer: Deque[float] = deque(maxlen=MAX_POINTS)

    fig, axes, raw_line, filtered_line, peak_marks, hr_line = init_plots()

    last_time = 0.0

    async with httpx.AsyncClient(base_url=URL, timeout=5.0) as client:
        while True:
            try:
                json_data = await fetch_data(client, last_time)
            except httpx.HTTPError as exc:
                print(f"Request failed: {exc}")
                await asyncio.sleep(POLL_INTERVAL)
                continue

            buffer: Dict[str, Any] = json_data.get("buffer", {})
            acc_times: "list[float]" = buffer.get("acc_time", {}).get("buffer", [])
            acc_z = buffer.get("accZ", {}).get("buffer", [])

            added = 0
            for t, z in zip(acc_times, acc_z):
                if t > last_time:
                    time_buffer.append(float(t))
                    accel_buffer.append(-float(z))
                    last_time = float(t)
                    added += 1

            if added and len(time_buffer) >= 5:
                times_np = np.array(time_buffer, dtype=float)
                signal_np = np.array(accel_buffer, dtype=float)
                metrics = compute_metrics(times_np, signal_np)
                update_plots(
                    fig,
                    axes,
                    raw_line,
                    filtered_line,
                    peak_marks,
                    hr_line,
                    times_np,
                    signal_np,
                    metrics,
                )

            await asyncio.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    asyncio.run(main())
