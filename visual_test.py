import cv2
import os
import numpy as np
from wood_track_detector import WoodTrackDetector

# --- Configurare ---
INPUT_DIR = 'test_set'  # Folderul cu imaginile tale
OUTPUT_DIR = 'results'  # Aici se salvează rezultatele
SAVE_OVERLAY = True
SAVE_MASK = True
SHOW_RESULTS = False  # Dacă True, arată fiecare imagine pe ecran (apasă o tastă)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    detector = WoodTrackDetector()
    extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    for fname in sorted(os.listdir(INPUT_DIR)):
        if not fname.lower().endswith(extensions):
            continue

        img_path = os.path.join(INPUT_DIR, fname)
        print(f"Procesez: {fname} ...")

        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Nu am putut citi {fname}")
            continue

        # Procesare
        mask, centerline, offset, heading, debug = detector.process_frame(frame)

        print(f"  Offset lateral: {offset:.1f} px, Heading: {heading:.1f}°")
        if len(centerline) > 0:
            print(f"  Puncte centerline: {len(centerline)}")
        else:
            print("Nu s-a detectat nicio linie centrală!")

        base_name = os.path.splitext(fname)[0]

        if SAVE_MASK:
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_mask.png"), mask)

        if SAVE_OVERLAY:
            if debug is not None:
                overlay_out = debug
            else:
                overlay_out = frame.copy()
                if len(centerline) > 0:
                    for pt in centerline:
                        cv2.circle(overlay_out, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1)
                h, w = overlay_out.shape[:2]
                cv2.line(overlay_out, (w // 2, h - 1), (w // 2, h // 2), (0, 0, 255), 2)
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_overlay.jpg"), overlay_out)

        # Imagine combinată (original | mască | overlay)
        if SAVE_MASK and SAVE_OVERLAY:
            original = frame.copy()
            mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            if debug is not None:
                overlay_color = debug
            else:
                overlay_color = overlay_out

            # Redimensionează la aceeași înălțime
            h = original.shape[0]
            mask_color = cv2.resize(mask_color, (int(h * mask_color.shape[1] / mask_color.shape[0]), h)) if \
            mask_color.shape[0] != h else mask_color
            overlay_color = cv2.resize(overlay_color, (int(h * overlay_color.shape[1] / overlay_color.shape[0]), h)) if \
            overlay_color.shape[0] != h else overlay_color

            combined = np.hstack([original, mask_color, overlay_color])
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_combined.jpg"), combined)

        if SHOW_RESULTS:
            cv2.imshow(f"Original - {fname}", frame)
            cv2.imshow(f"Masca - {fname}", mask)
            if debug is not None:
                cv2.imshow(f"Overlay - {fname}", debug)
            key = cv2.waitKey(0) & 0xFF
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                break
            cv2.destroyAllWindows()

    print(f"\nToate rezultatele sunt în folderul '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()