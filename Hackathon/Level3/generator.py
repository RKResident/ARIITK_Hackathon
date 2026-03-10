"""
generator.py — Project MIRAGE
Team Aerial Robotics IITK | Y25 Recruitment Hackathon

Produces a single 512x512 grayscale Hybrid Marker that:
  - Shows a decoy helipad "H" at high altitude (≥30m)
  - Reveals a hidden ArUco marker at low altitude (≤5m)

Usage:
    python generator.py decoy.png aruco_markers/aruco_id_X.png X output.png

Example:
    python generator.py far_target_decoy.png aruco_markers/aruco_id_07.png 7 hybrid_marker.png
"""

import sys
import cv2
import numpy as np
from aruco_detect import detect_aruco


# ─────────────────────────────────────────────
#  PINHOLE CAMERA MODEL CONSTANTS (RPi Cam V2)
# ─────────────────────────────────────────────
FOCAL_LENGTH_MM   = 3.04    # mm
SENSOR_WIDTH_MM   = 3.68    # mm
RESOLUTION_PX     = 1920    # pixels (horizontal)
MARKER_SIZE_M     = 1.0     # physical marker size in metres

def altitude_to_pixels(altitude_m: float) -> int:
    """
    Pinhole Camera Model:
        pixels = (f × S_real × Resolution) / (Z × sensor_width)

    Returns how many pixels the 1m marker occupies on the sensor
    at the given altitude.
    """
    pixels = (FOCAL_LENGTH_MM * MARKER_SIZE_M * RESOLUTION_PX) / \
             (altitude_m * SENSOR_WIDTH_MM)
    return int(round(pixels))


# ─────────────────────────────────────────────
#  SIGMA SELECTION — backed by GSD / Nyquist
# ─────────────────────────────────────────────
#
#  GSD (Ground Sampling Distance) at altitude Z:
#      GSD = (Z × sensor_width) / (f × resolution)
#
#  At 30m  → GSD ≈ 0.0192 m/px  → marker is ~52px wide
#  At  5m  → GSD ≈ 0.0032 m/px  → marker is ~317px wide
#
#  ArUco 4×4 grid on a 512px image → each cell ≈ 512/6 ≈ 85px
#  Nyquist frequency of ArUco grid ≈ 1/(2×85) cycles/px ≈ 0.006 cyc/px
#
#  We want σ_low  to kill all frequencies above ~0.004 cyc/px
#      → σ_low  ≈ 1/(2π × 0.004) ≈ 40  (aggressive low-pass on decoy)
#
#  We want σ_high to only remove DC / very slow drift from ArUco
#      → σ_high ≈ 2  (gentle blur — keeps all ArUco edge detail)
#
#  Gap ratio: σ_low / σ_high = 40 / 2 = 20  ✓  (well above required 3×)
#
SIGMA_LOW  = 40   # Low-pass σ  for decoy  (keeps only broad shapes)
SIGMA_HIGH =  2   # High-pass σ for ArUco  (removes only DC offset)


def load_grayscale_512(path: str, label: str) -> np.ndarray:
    """Load an image, convert to grayscale, resize to 512×512."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[ERROR] Could not load {label} from: {path}")
        sys.exit(1)
    if img.shape != (512, 512):
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
        print(f"[INFO]  {label} resized to 512×512")
    return img.astype(np.float32)


def low_pass(image: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian low-pass filter — keeps smooth/broad content."""
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)


def high_pass(image: np.ndarray, sigma: float) -> np.ndarray:
    """
    High-pass filter = original − low-pass.
    Mean is subtracted so the signal has zero DC offset,
    preventing ghosting on the decoy layer.
    """
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    hp = image - blurred
    hp -= hp.mean()          # ← zero-mean: eliminates DC ghosting
    return hp


def blend_and_normalise(lp: np.ndarray, hp: np.ndarray,
                         alpha: float = 1.0, beta: float = 1.0) -> np.ndarray:
    """
    Combine low-pass decoy + high-pass ArUco.
    alpha / beta let you tune relative contribution of each layer.
    Clips and converts to uint8 [0, 255].
    """
    combined = alpha * lp + beta * hp
    combined = np.clip(combined, 0, 255)
    return combined.astype(np.uint8)


def verify_aruco(image: np.ndarray, aruco_id: int, altitude_m: float) -> bool:
    """
    Simulate drone view at given altitude and attempt ArUco detection
    using the edge-integration pipeline from aruco_detect.py.
    """
    px    = max(altitude_to_pixels(altitude_m), 10)
    small = cv2.resize(image, (px, px), interpolation=cv2.INTER_AREA)
    if altitude_m >= 15:
        small = cv2.GaussianBlur(small, (3, 3), sigmaX=0.8)
    view        = cv2.resize(small, (512, 512), interpolation=cv2.INTER_NEAREST)
    found, _    = detect_aruco(view, aruco_id)
    return found


def main():
    # ── Argument Parsing ──────────────────────────────────────────────
    if len(sys.argv) != 5:
        print("Usage: python generator.py <decoy.png> <aruco.png> <aruco_id> <output.png>")
        print("Example: python generator.py far_target_decoy.png aruco_markers/aruco_id_07.png 7 hybrid_marker.png")
        sys.exit(1)

    decoy_path  = sys.argv[1]
    aruco_path  = sys.argv[2]
    aruco_id    = int(sys.argv[3])
    output_path = sys.argv[4]

    print(f"\n{'='*55}")
    print(f"  Project MIRAGE — Hybrid Marker Generator")
    print(f"{'='*55}")
    print(f"  Decoy  : {decoy_path}")
    print(f"  ArUco  : {aruco_path}  (ID = {aruco_id})")
    print(f"  Output : {output_path}")
    print(f"  σ_low  = {SIGMA_LOW}   |   σ_high = {SIGMA_HIGH}")
    print(f"  Gap ratio: {SIGMA_LOW/SIGMA_HIGH:.1f}×  (must be ≥ 3×) ✓")
    print(f"{'='*55}\n")

    # ── Load Images ───────────────────────────────────────────────────
    decoy = load_grayscale_512(decoy_path, "Decoy")
    aruco = load_grayscale_512(aruco_path, "ArUco")

    # ── Frequency Separation ──────────────────────────────────────────
    print("[1/4]  Applying low-pass filter to decoy  (σ_low  = {})".format(SIGMA_LOW))
    lp = low_pass(decoy, SIGMA_LOW)

    print("[2/4]  Applying high-pass filter to ArUco (σ_high = {})".format(SIGMA_HIGH))
    hp = high_pass(aruco, SIGMA_HIGH)

    # ── Blend ─────────────────────────────────────────────────────────
    print("[3/4]  Blending layers → hybrid_marker")
    # beta=1.5 boosts ArUco signal strength for reliable detection
    # without pushing ghosting score above 5 at 30m
    hybrid = blend_and_normalise(lp, hp, alpha=1.0, beta=1.5)

    # ── Save Output ───────────────────────────────────────────────────
    cv2.imwrite(output_path, hybrid)
    print(f"[4/4]  Saved → {output_path}  ({hybrid.shape[1]}×{hybrid.shape[0]} px, grayscale)\n")

    # ── Self-Verification ─────────────────────────────────────────────
    print("Running self-verification across altitudes...")
    print(f"{'─'*45}")

    test_altitudes = [2, 5, 15, 30, 100]
    for alt in test_altitudes:
        px      = altitude_to_pixels(alt)
        found   = verify_aruco(hybrid, aruco_id, alt)
        status  = "DETECTED ✓" if found else "NOT DETECTED ✗"
        expected = "DETECTED ✓" if alt <= 5 else ("NOT DETECTED ✗" if alt >= 30 else "transition")
        match   = "✓" if (found == (alt <= 5)) or alt in [15] else "⚠"
        print(f"  Alt {alt:>4}m  →  {px:>4}px  →  ArUco {status:<18}  {match}")

    print(f"{'─'*45}")
    print("\n  Hint: Run viewer.py to generate drone_view PNGs.")
    print("  Hint: Run visual_check.py to measure ghosting score.")
    print(f"\n{'='*55}\n")


if __name__ == "__main__":
    main()