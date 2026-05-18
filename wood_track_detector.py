import cv2
import numpy as np

class WoodTrackDetector:
    def __init__(self):
        # Parametri reglabili
        self.hsv_lower = (10, 50, 50)
        self.hsv_upper = (30, 255, 255)
        self.lab_a_thresh = 130          # canalul 'a' peste această valoare
        self.close_kernel = (7, 7)       # pentru closing
        self.open_kernel = (3, 3)        # pentru opening
        self.roi = None                  # <-- FĂRĂ ROI, procesează toată imaginea

    def preprocess(self, frame):
        # Dacă nu e definit ROI, returnează cadrul întreg
        if self.roi is None:
            return frame
        x, y, w, h = self.roi
        return frame[y:y+h, x:x+w]

    def segment_track(self, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        # Masca HSV
        mask_hsv = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        # Masca Lab: canalul a (index 1)
        _, a_channel, _ = cv2.split(lab)
        _, mask_lab = cv2.threshold(a_channel, self.lab_a_thresh, 255, cv2.THRESH_BINARY)
        # Intersecție
        mask = cv2.bitwise_and(mask_hsv, mask_lab)
        return mask

    def apply_morphology(self, mask):
        # Closing (umple găuri)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.close_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        # Opening (șterge zgomote mici)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        return mask

    def keep_largest_connected_component(self, mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return mask  # doar fundal, nimic de făcut
        # stats[0] este fundalul, ignorăm
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
        return mask

    def extract_centerline(self, mask):
        points = []
        rows, cols = mask.shape
        min_width = cols * 0.02  # minim 2% din lățime (ex: 10 px la 640px)
        max_width = cols * 0.90  # maxim 90% (evident, o șină nu ocupă tot cadrul)

        for y in range(rows):
            white_cols = np.where(mask[y, :] > 0)[0]
            if len(white_cols) < min_width or len(white_cols) > max_width:
                continue

            # Verificăm dacă sunt mai multe grupuri (găuri)
            # O metodă simplă: dacă diferența dintre primul și ultimul pixel alb
            # e aproape egală cu numărul total de pixeli albi, nu sunt găuri mari.
            span = white_cols[-1] - white_cols[0] + 1
            if span > len(white_cols) * 1.5:  # sunt goluri, posibil segment dublu
                # găsim cel mai lung subsegment continuu
                edges = np.diff(white_cols)
                gaps = np.where(edges > 1)[0]
                if len(gaps) == 0:
                    # nu sunt goluri, e okay
                    x_center = np.mean(white_cols)
                    points.append((x_center, y))
                else:
                    # ia segmentul cu cel mai mare număr de pixeli consecutivi
                    segments = np.split(white_cols, gaps + 1)
                    best_seg = max(segments, key=len)
                    x_center = np.mean(best_seg)
                    points.append((x_center, y))
            else:
                x_center = np.mean(white_cols)
                points.append((x_center, y))

        return points

    def smooth_centerline(self, points, window=5):
        """Medie mobilă simplă pe coordonatele x"""
        if len(points) < window:
            return points
        smoothed = []
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        half = window // 2
        for i in range(len(xs)):
            start = max(0, i - half)
            end = min(len(xs), i + half + 1)
            avg_x = np.mean(xs[start:end])
            smoothed.append((avg_x, ys[i]))
        return smoothed

    def compute_offset_and_heading(self, centerline, roi_width, roi_height):
        if len(centerline) < 5:
            return 0.0, 0.0

        # Folosim doar punctele din jumătatea inferioară (mai aproape de dronă)
        bottom_half = [p for p in centerline if p[1] > roi_height * 0.5]
        if len(bottom_half) < 2:
            bottom_half = centerline[-5:]  # fallback

        xs = np.array([p[0] for p in bottom_half])
        ys = np.array([p[1] for p in bottom_half])

        # Regresie liniară cu RANSAC pentru robustețe
        from sklearn.linear_model import RANSACRegressor
        ransac = RANSACRegressor(min_samples=0.5, residual_threshold=5)
        ransac.fit(ys.reshape(-1, 1), xs)
        m = ransac.estimator_.coef_[0]
        c = ransac.estimator_.intercept_

        heading_deg = np.degrees(np.arctan(m))
        image_center_x = roi_width / 2
        bottom_y = roi_height - 1
        x_at_bottom = m * bottom_y + c
        lateral_offset = x_at_bottom - image_center_x

        return lateral_offset, heading_deg

    def process_frame(self, frame):
        roi = self.preprocess(frame)              # acum e tot cadrul
        mask = self.segment_track(roi)
        mask = self.apply_morphology(mask)
        mask = self.keep_largest_connected_component(mask)
        raw_points = self.extract_centerline(mask)
        if len(raw_points) == 0:
            # Șina nu e detectată – returnează valori sigure
            h, w = roi.shape[:2]
            return mask, [], 0.0, 0.0, roi
        smoothed = self.smooth_centerline_poly(raw_points, degree=3)
        h, w = roi.shape[:2]
        offset, heading = self.compute_offset_and_heading(smoothed, w, h)
        # Debug overlay: desenează linia centrală și punctele
        debug = roi.copy()
        for pt in smoothed:
            cx, cy = int(pt[0]), int(pt[1])
            cv2.circle(debug, (cx, cy), 2, (0, 255, 0), -1)
        cv2.line(debug, (w//2, h-1), (w//2, 0), (0, 0, 255), 1)  # linia centrală a camerei
        return mask, smoothed, offset, heading, debug

    def smooth_centerline_poly(self, points, degree=2):
        #Aproximează punctele cu un polinom x = f(y) de gradul dat.
        if len(points) < degree + 1:
            return points

        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])

        # Potrivire polinomială: x = poly(y)
        coeffs = np.polyfit(ys, xs, degree)
        poly = np.poly1d(coeffs)

        smoothed = []
        for y in ys:
            x_smooth = poly(y)
            smoothed.append((x_smooth, y))
        return smoothed