"""Very cool project :yesyes:"""

import asyncio
from collections import deque
from typing import Any, Deque, Dict, Tuple, TypedDict, cast

import httpx
import matplotlib.pyplot as plt
import numpy as np
import websockets
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt, find_peaks  # type: ignore
from websockets.server import ServerProtocol

send_message_queue: deque[float] = deque()

URL = "http://192.168.20.59:8080"
MAX_POINTS = 250
LOWCUT = 0.8  # 48 BPM - low end of heart rate
HIGHCUT = 4.0  # 240 BPM - high end of heart rate
POLL_INTERVAL = 0.1
PROMINENCE_MULTIPLIER = 0.5
MIN_PROMINENCE = 0.01
MIN_PEAK_SPACING_SEC = 0.4  # ~150 BPM max to avoid double-counting
BPM_WINDOW = 5  # Used for smoothing


async def fetch_data(client: httpx.AsyncClient, last_time: float) -> Dict[str, Any]:
    """Fetch data from phyphox"""
    params: Dict[str, str | float] = {
        "accZ": f"{last_time}|acc_time",
        "acc_time": last_time,
    }
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
    b, a = butter(order, [low, high], btype="band")  # type: ignore
    if data.size <= 27:
        # Skip if not enough data
        return np.array([])
    return filtfilt(b, a, data)


class Metrics(TypedDict):
    """Metrics type"""

    filtered: np.ndarray
    peaks: np.ndarray
    bpm_times: np.ndarray
    bpm: np.ndarray
    instantaneous_bpm: np.ndarray
    fs: float | None


def compute_metrics(times: np.ndarray, signal: np.ndarray) -> Metrics:
    """Get metrics from SCG signal"""
    metrics: Metrics = {
        "filtered": np.array([]),
        "peaks": np.array([], dtype=int),
        "bpm_times": np.array([]),
        "bpm": np.array([]),
        "instantaneous_bpm": np.array([]),
        "fs": None,
    }

    # Early exit if not enough data
    if len(times) < 10:
        return metrics

    # Calculate sampling frequency
    dt = np.diff(times)
    if dt.size == 0 or not np.all(dt > 0):
        return metrics

    # Calculate Hz
    fs_value = float(1 / np.mean(dt))
    if fs_value <= 0:
        return metrics

    # Filter signal
    filtered = butter_bandpass_filter(signal, LOWCUT, HIGHCUT, float(fs_value))
    if not filtered.size:
        return metrics

    # Convert to absolute values
    envelope = np.abs(filtered)

    # Detect peaks
    distance = max(int(MIN_PEAK_SPACING_SEC * fs_value), 1)

    # Dynamic thresholding using percentiles
    baseline = np.percentile(envelope, 25)
    peak_level = np.percentile(envelope, 75)
    min_height = baseline + (peak_level - baseline) * 0.3

    # Calculate prominence relative to noise level
    mad = np.median(np.abs(envelope - np.median(envelope)))
    noise_level = max(mad * 1.4826, MIN_PROMINENCE)
    min_prominence = PROMINENCE_MULTIPLIER * noise_level

    peaks_raw, _properties = cast(
        Tuple[np.ndarray, dict[str, Any]],
        find_peaks(
            envelope,
            distance=distance,
            height=min_height,
            prominence=min_prominence,
        ),
    )

    peaks_arr = np.asarray(peaks_raw, dtype=int)
    metrics["filtered"] = filtered
    metrics["peaks"] = peaks_arr
    metrics["fs"] = fs_value

    # Calculate BPM from inter-beat intervals
    if peaks_arr.size >= 2:
        peak_times = times[peaks_arr]
        ibi = np.diff(peak_times)  # Inter-beat intervals in seconds
        valid = (ibi >= 0.25) & (ibi <= 2.0)  # Filter for valid IBIs (30-240 BPM)

        if valid.any():
            ibi_valid = ibi[valid]
            instantaneous_bpm = 60 / ibi_valid
            bpm_times = peak_times[1:][valid]

            # Apply moving median filter
            if len(instantaneous_bpm) >= BPM_WINDOW:
                bpm_smooth = np.array(
                    [
                        np.median(instantaneous_bpm[max(0, i - BPM_WINDOW + 1) : i + 1])
                        for i in range(len(instantaneous_bpm))
                    ]
                )
            else:
                bpm_smooth = instantaneous_bpm

            metrics["instantaneous_bpm"] = instantaneous_bpm
            metrics["bpm"] = bpm_smooth
            metrics["bpm_times"] = bpm_times
            return metrics

    # Not enough valid peaks
    metrics["bpm"] = np.array([0.0])
    metrics["bpm_times"] = np.array([times[-1]])
    metrics["instantaneous_bpm"] = np.array([0.0])
    return metrics


def init_plots() -> Tuple[Figure, NDArray[np.object_], Line2D, Line2D, Line2D, Line2D]:
    """Init the plots"""
    plt.ion()  # type: ignore
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))  # type: ignore

    (raw_line,) = axes[0].plot([], [], label="Raw accZ", color="gray", alpha=0.6)
    axes[0].set_title("Raw Accelerometer Data")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Acceleration (m/s^2)")
    axes[0].grid(True)
    axes[0].legend()

    (filtered_line,) = axes[1].plot([], [], label="Filtered SCG (0.8-4.0 Hz)")
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
    plt.show(block=False)  # type: ignore

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

    fig.canvas.draw()  # type: ignore
    fig.canvas.flush_events()
    plt.pause(0.001)


async def create_handler(websocket: ServerProtocol) -> None:
    """Create a websocket handler"""
    try:
        while True:
            if len(send_message_queue) > 0:
                message = send_message_queue.popleft()
                await websocket.send(f"{message}")  # type: ignore
            await asyncio.sleep(0.01)
    except websockets.exceptions.ConnectionClosed:
        pass  # Client disconnected


async def main() -> None:
    """Main entrypoint"""
    time_buffer: Deque[float] = deque(maxlen=MAX_POINTS)
    accel_buffer: Deque[float] = deque(maxlen=MAX_POINTS)

    fig, axes, raw_line, filtered_line, peak_marks, hr_line = init_plots()

    server = await websockets.serve(create_handler, "0.0.0.0", 5555)  # type: ignore
    await server.start_serving()

    last_time = 0.0
    async with httpx.AsyncClient(base_url=URL, timeout=5.0) as client:
        while True:
            # Fetch new data from phyphox
            try:
                json_data = await fetch_data(client, last_time)
            except httpx.HTTPError as exc:
                print(f"Request failed: {exc}")
                await asyncio.sleep(POLL_INTERVAL)
                continue

            buffer: Dict[str, Any] = json_data.get("buffer", {})
            acc_times: "list[float]" = buffer.get("acc_time", {}).get("buffer", [])
            acc_z = buffer.get("accZ", {}).get("buffer", [])

            # Parse and append new data
            added = 0
            for t, z in zip(acc_times, acc_z):
                if t > last_time:
                    time_buffer.append(float(t))
                    accel_buffer.append(-float(z))
                    last_time = float(t)
                    added += 1

            # Update plots if there's enough data
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

                # Send current BPM to WebSocket
                bpm = metrics.get("bpm", np.array([]))
                if bpm.size > 0:
                    current_bpm = float(bpm[-1] or 0.0)
                    send_message_queue.append(current_bpm)

            await asyncio.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    asyncio.run(main())
