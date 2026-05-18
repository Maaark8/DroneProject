from contextlib import contextmanager
import platform

import cv2


DEFAULT_MAX_CAMERA_INDEX = 10


def _candidate_backends():
    """Prefer a stable live backend, then keep portable fallbacks."""
    backends = []
    if platform.system() == "Windows" and hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    backends.extend(_probe_backends())
    if cv2.CAP_ANY not in backends:
        backends.append(cv2.CAP_ANY)
    return backends


def _probe_backends():
    if platform.system() == "Windows" and hasattr(cv2, "CAP_MSMF"):
        return [cv2.CAP_MSMF]
    return [cv2.CAP_ANY]


@contextmanager
def _suppress_opencv_logs():
    logging_api = getattr(getattr(cv2, "utils", None), "logging", None)
    if logging_api is None:
        yield
        return

    previous_level = logging_api.getLogLevel()
    logging_api.setLogLevel(logging_api.LOG_LEVEL_SILENT)
    try:
        yield
    finally:
        logging_api.setLogLevel(previous_level)


def _open_capture(camera_index, backends=None):
    with _suppress_opencv_logs():
        for backend in backends or _candidate_backends():
            capture = cv2.VideoCapture(camera_index, backend)
            if not capture.isOpened():
                capture.release()
                continue

            ok, frame = capture.read()
            if ok and frame is not None:
                return capture

            capture.release()

    return None


def find_available_cameras(max_index=DEFAULT_MAX_CAMERA_INDEX):
    """Return camera indices that can produce at least one valid frame."""
    available = []
    for camera_index in range(max_index):
        capture = _open_capture(camera_index, backends=_probe_backends())
        if capture is None:
            continue
        available.append(camera_index)
        capture.release()
    return available


def open_camera(camera_index, width=640, height=480, fps=30):
    """Open one camera index and configure preferred capture properties."""
    capture = _open_capture(camera_index)
    if capture is None:
        raise RuntimeError(f"Nu am putut deschide camera cu indexul {camera_index}.")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    return capture
