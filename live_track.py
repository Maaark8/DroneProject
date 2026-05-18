import argparse

import cv2

from camera_utils import DEFAULT_MAX_CAMERA_INDEX, find_available_cameras, open_camera
from wood_track_detector import WoodTrackDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Detectie live a pistei folosind orice camera disponibila.")
    parser.add_argument("--camera", type=int, help="Indexul camerei OpenCV de folosit.")
    parser.add_argument(
        "--max-cameras",
        type=int,
        default=DEFAULT_MAX_CAMERA_INDEX,
        help="Cate indexuri sa fie testate cand camera nu este aleasa explicit.",
    )
    parser.add_argument("--width", type=int, default=640, help="Latimea dorita a fluxului video.")
    parser.add_argument("--height", type=int, default=480, help="Inaltimea dorita a fluxului video.")
    parser.add_argument("--fps", type=int, default=30, help="FPS-ul dorit al fluxului video.")
    return parser.parse_args()


def main():
    args = parse_args()
    detector = WoodTrackDetector()

    if args.camera is None:
        available_cameras = find_available_cameras(args.max_cameras)
        if not available_cameras:
            raise RuntimeError("Nu am gasit nicio camera disponibila.")
        camera_index = available_cameras[0]
        print(f"Camere detectate: {available_cameras}. Folosesc automat camera {camera_index}.")
    else:
        camera_index = args.camera

    capture = open_camera(camera_index, args.width, args.height, args.fps)
    print(f"Camera {camera_index} a pornit. Apasa 'q' pentru a iesi.")

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                continue

            mask, centerline, offset, heading, debug = detector.process_frame(frame)

            cv2.imshow("Track Detection - Live", debug)
            cv2.imshow("Mask", mask)
            print(f"\rOffset: {offset:+.1f}px | Heading: {heading:+.1f} deg", end="")

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        print("\nCamera a fost oprita.")


if __name__ == "__main__":
    main()
