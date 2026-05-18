from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2


def open_live_capture(source: str | int):
    if _is_realtime_source(source):
        return LatestFrameCapture(source)
    return cv2.VideoCapture(str(source))


def normalize_capture_source(source: str | int) -> str | int:
    if isinstance(source, int):
        return source
    stripped = source.strip()
    if stripped.isdecimal():
        return int(stripped)
    return stripped


def is_probably_stream_url(source: str) -> bool:
    lowered = source.strip().lower()
    return lowered.startswith(
        (
            "http://",
            "https://",
            "rtsp://",
            "rtsps://",
            "rtmp://",
            "udp://",
            "tcp://",
        )
    )


class LatestFrameCapture:
    def __init__(self, source: str | int) -> None:
        self.source = source
        self._capture = _open_capture_with_latency_hints(source)
        self._condition = threading.Condition()
        self._latest_frame = None
        self._latest_seq = -1
        self._consumer_seq = -1
        self._stopped = False
        self._thread = threading.Thread(target=self._reader_loop, name="latest-frame-capture", daemon=True)
        self._thread.start()

    def isOpened(self) -> bool:
        return self._capture.isOpened()

    def read(self, timeout_s: float = 1.0):
        deadline = time.perf_counter() + max(float(timeout_s), 0.01)
        with self._condition:
            while self._latest_seq <= self._consumer_seq and not self._stopped:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return False, None
                self._condition.wait(timeout=remaining)

            if self._latest_seq <= self._consumer_seq:
                return False, None

            self._consumer_seq = self._latest_seq
            return True, self._latest_frame

    def get(self, prop_id: int) -> float:
        return float(self._capture.get(prop_id))

    def release(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        self._thread.join(timeout=1.0)
        self._capture.release()

    def _reader_loop(self) -> None:
        while True:
            with self._condition:
                if self._stopped:
                    return

            ok, frame = self._capture.read()
            with self._condition:
                if self._stopped:
                    return
                if not ok:
                    self._stopped = True
                    self._condition.notify_all()
                    return
                self._latest_frame = frame
                self._latest_seq += 1
                self._condition.notify_all()


def _open_capture_with_latency_hints(source: str | int):
    if isinstance(source, int):
        capture = cv2.VideoCapture(source)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    if is_probably_stream_url(source):
        capture = _open_stream_with_ffmpeg(source)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return capture

    capture = cv2.VideoCapture(str(source))
    if capture.isOpened():
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def _open_stream_with_ffmpeg(source: str):
    capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if capture.isOpened():
        _apply_best_effort_capture_settings(capture)
        return capture
    capture = cv2.VideoCapture(source)
    if capture.isOpened():
        _apply_best_effort_capture_settings(capture)
    return capture


def _apply_best_effort_capture_settings(capture) -> None:
    for prop, value in (
        (cv2.CAP_PROP_BUFFERSIZE, 1),
        (cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000),
        (cv2.CAP_PROP_READ_TIMEOUT_MSEC, 1000),
    ):
        try:
            capture.set(prop, value)
        except Exception:
            continue


def _is_realtime_source(source: str | int) -> bool:
    if isinstance(source, int):
        return True
    if is_probably_stream_url(source):
        return True
    return not Path(source).exists()
