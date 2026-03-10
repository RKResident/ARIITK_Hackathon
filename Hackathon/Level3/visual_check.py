"""
visual_check.py — Project MIRAGE
Team Aerial Robotics IITK | Y25 Recruitment Hackathon

Measures the ghosting score of a hybrid marker:
how much ArUco signal bleeds into the decoy view at 30m altitude.

Algorithm:
  1. Simulate 30m drone view of hybrid  (shrink → atm blur → upscale)
  2. Blur that view with σ=10           (integrates residual HF content)
  3. Apply same blur to the pure decoy  (the ideal 30m reference)
  4. Ghosting score = mean absolute difference between (2) and (3)

Target: score < 5.
  A score of 0 means the hybrid is indistinguishable from the raw decoy
  at 30m. A score above 5 means ArUco structure is visibly bleeding through.

Usage:
    python visual_check.py <hybrid.png> <decoy.png>

Example:
    python visual_check.py hybrid_marker.png far_target_decoy.png
"""

import sys
import cv2
import numpy as np


# ── Pinhole (RPi Cam V2) ──────────────────────────────────────────────────────
FOCAL_LENGTH_MM = 3.04
SENSOR_WIDTH_MM = 3.68
RESOLUTION_PX   = 1920
MARKER_SIZE_M   = 1.0
CHECK_ALTITUDE  = 30.0      # metres — altitude at which ghosting is measured
GHOSTING_SIGMA  = 10.0      # σ for integration blur (simulates eye/optics)
GHOSTING_LIMIT  = 5.0       # pass threshold

def altitude_to_pixels(altitude_m: float) -> int:
    px = (FOCAL_LENGTH_MM * MARKER_SIZE_M * RESOLUTION_PX) / \
         (altitude_m * SENSOR_WIDTH_MM)
    return max(int(round(px)), 10)


def simulate_view(img: np.ndarray, altitude_m: float) -> np.ndarray:
    """Shrink to sensor pixel count, apply atmospheric blur, upscale to 512."""
    px    = altitude_to_pixels(altitude_m)
    small = cv2.resize(img, (px, px), interpolation=cv2.INTER_AREA)
    if altitude_m >= 15:
        small = cv2.GaussianBlur(small, (3, 3), sigmaX=0.8)
    return cv2.resize(small, (512, 512), interpolation=cv2.INTER_NEAREST)


def ghosting_score(hybrid: np.ndarray, decoy: np.ndarray) -> tuple:
    """
    Returns (score, hybrid_view, decoy_view, diff_map).

    hybrid_view : 30m simulation of hybrid, integration-blurred
    decoy_view  : 30m simulation of pure decoy, integration-blurred
    diff_map    : absolute difference (uint8, scaled for visualisation)
    """
    # Step 1 — simulate 30m view for both images
    hybrid_sim = simulate_view(hybrid, CHECK_ALTITUDE).astype(np.float32)
    decoy_sim  = simulate_view(decoy,  CHECK_ALTITUDE).astype(np.float32)

    # Step 2 — integration blur (what an observer integrates at distance)
    hybrid_blurred = cv2.GaussianBlur(hybrid_sim, (0, 0), sigmaX=GHOSTING_SIGMA)
    decoy_blurred  = cv2.GaussianBlur(decoy_sim,  (0, 0), sigmaX=GHOSTING_SIGMA)

    # Step 3 — absolute difference
    diff  = np.abs(hybrid_blurred - decoy_blurred)
    score = float(diff.mean())

    # Scale diff map for visibility (×10 so subtle bleed shows clearly)
    diff_vis = np.clip(diff * 10, 0, 255).astype(np.uint8)
    diff_vis = cv2.applyColorMap(diff_vis, cv2.COLORMAP_INFERNO)

    return (
        score,
        hybrid_blurred.astype(np.uint8),
        decoy_blurred.astype(np.uint8),
        diff_vis,
    )


def build_report_image(hybrid_view, decoy_view, diff_map,
                        score, hybrid_path, decoy_path) -> np.ndarray:
    """Compose a side-by-side visual report as a single image."""
    PANEL  = 512
    PAD    = 12
    BAR    = 52
    TOTAL_W = PANEL * 3 + PAD * 4
    TOTAL_H = BAR + PANEL + BAR

    canvas = np.zeros((TOTAL_H, TOTAL_W, 3), dtype=np.uint8)
    canvas[:] = (22, 22, 30)

    # ── Top bar ───────────────────────────────────────────────────────────────
    passed     = score < GHOSTING_LIMIT
    bar_color  = (0, 130, 0) if passed else (0, 0, 180)
    verdict    = f"PASS  (score={score:.2f} < {GHOSTING_LIMIT})" if passed else \
                 f"FAIL  (score={score:.2f} >= {GHOSTING_LIMIT})"
    canvas[:BAR] = bar_color
    cv2.putText(canvas,
                f"MIRAGE Ghosting Check @ {CHECK_ALTITUDE}m  —  {verdict}",
                (PAD, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2,
                cv2.LINE_AA)

    # ── Three panels ──────────────────────────────────────────────────────────
    panels = [
        (cv2.cvtColor(hybrid_view, cv2.COLOR_GRAY2BGR), "Hybrid @ 30m (blurred)"),
        (cv2.cvtColor(decoy_view,  cv2.COLOR_GRAY2BGR), "Decoy  @ 30m (blurred)"),
        (diff_map,                                       f"Abs diff  ×10  (mean={score:.2f})"),
    ]

    for i, (panel_img, label) in enumerate(panels):
        x = PAD + i * (PANEL + PAD)
        y = BAR
        canvas[y:y+PANEL, x:x+PANEL] = cv2.resize(panel_img, (PANEL, PANEL))
        cv2.putText(canvas, label, (x + 8, y + PANEL + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1, cv2.LINE_AA)

    # ── Bottom bar ────────────────────────────────────────────────────────────
    y_bot = BAR + PANEL
    cv2.putText(canvas,
                f"hybrid: {hybrid_path}   |   decoy: {decoy_path}   "
                f"|   sigma_integration={GHOSTING_SIGMA}",
                (PAD, y_bot + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (130, 130, 150), 1, cv2.LINE_AA)

    return canvas


def main():
    if len(sys.argv) != 3:
        print("Usage:   python visual_check.py <hybrid.png> <decoy.png>")
        print("Example: python visual_check.py hybrid_marker.png far_target_decoy.png")
        sys.exit(1)

    hybrid_path = sys.argv[1]
    decoy_path  = sys.argv[2]

    hybrid = cv2.imread(hybrid_path, cv2.IMREAD_GRAYSCALE)
    decoy  = cv2.imread(decoy_path,  cv2.IMREAD_GRAYSCALE)

    if hybrid is None:
        print(f"[ERROR] Could not load hybrid: {hybrid_path}")
        sys.exit(1)
    if decoy is None:
        print(f"[ERROR] Could not load decoy: {decoy_path}")
        sys.exit(1)

    if hybrid.shape != (512, 512):
        hybrid = cv2.resize(hybrid, (512, 512), interpolation=cv2.INTER_AREA)
    if decoy.shape != (512, 512):
        decoy  = cv2.resize(decoy,  (512, 512), interpolation=cv2.INTER_AREA)

    score, hybrid_view, decoy_view, diff_map = ghosting_score(hybrid, decoy)

    passed  = score < GHOSTING_LIMIT
    verdict = "PASS" if passed else "FAIL"

    print(f"\n{'='*52}")
    print(f"  MIRAGE Ghosting Score @ {CHECK_ALTITUDE}m")
    print(f"{'='*52}")
    print(f"  Hybrid  : {hybrid_path}")
    print(f"  Decoy   : {decoy_path}")
    print(f"  Score   : {score:.4f}  (target < {GHOSTING_LIMIT})")
    print(f"  Verdict : {verdict}")
    print(f"{'='*52}\n")

    report = build_report_image(hybrid_view, decoy_view, diff_map,
                                 score, hybrid_path, decoy_path)

    save_path = "ghosting_report.png"
    cv2.imwrite(save_path, report)
    print(f"  Report saved → {save_path}")

    cv2.imshow("MIRAGE — Ghosting Report", report)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()