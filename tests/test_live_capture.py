from __future__ import annotations

from track_detection.live_capture import _is_realtime_source, is_probably_stream_url, normalize_capture_source


def test_normalize_capture_source_parses_camera_index_strings() -> None:
    assert normalize_capture_source("0") == 0
    assert normalize_capture_source(" 12 ") == 12
    assert normalize_capture_source("http://127.0.0.1:8080/video") == "http://127.0.0.1:8080/video"


def test_stream_url_detection_handles_phone_camera_urls() -> None:
    assert is_probably_stream_url("http://192.168.1.2:8080/video")
    assert is_probably_stream_url("rtsp://192.168.1.2/live")
    assert not is_probably_stream_url("tracks_for_drone/wood_track.jpeg")


def test_realtime_source_detection_distinguishes_files_from_streams(tmp_path) -> None:
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"demo")

    assert _is_realtime_source(0)
    assert _is_realtime_source("http://192.168.1.2:8080/video")
    assert not _is_realtime_source(str(video_path))
