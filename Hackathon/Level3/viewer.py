"""
viewer.py — Project MIRAGE
Team Aerial Robotics IITK | Y25 Recruitment Hackathon

Simulates what a drone camera sees at a given altitude.
Displays the view, prints detection result, and saves the output PNG.

Usage:
    python viewer.py <image_path> <altitude_m> <aruco_id>

Examples:
    python viewer.py hybrid_marker.png 30 7
    python viewer.py hybrid_marker.png 2 7
"""

import sys
import cv2
import numpy as np
from aruco_detect import detect_aruco


# ─────────────────────────────────────────────
#  PINHOLE CAMERA MODEL CONSTANTS (RPi Cam V2)
# ─────────────────────────────────────────────
FOCAL_LENGTH_MM = 3.04
SENSOR_WIDTH_MM = 3.68
RESOLUTION_PX   = 1920
MARKER_SIZE_M   = 1.0

def altitude_to_pixels(altitude_m: float) -> int:
    px = (FOCAL_LENGTH_MM * MARKER_SIZE_M * RESOLUTION_PX) / \
         (altitude_m * SENSOR_WIDTH_MM)
    return max(int(round(px)), 10)



def simulate_drone_view(hybrid: np.ndarray, altitude_m: float):
    """
    Shrink image to simulate sensor pixel count at given altitude.
    Apply atmospheric blur at high altitudes.
    Returns (drone_view_upscaled, px_size, blur_applied)
    """
    px = altitude_to_pixels(altitude_m)

    # Shrink to sensor pixel size
    small = cv2.resize(hybrid, (px, px), interpolation=cv2.INTER_AREA)

    # Atmospheric blur — only at altitude ≥ 15m
    blur_applied = False
    if altitude_m >= 15:
        small = cv2.GaussianBlur(small, (3, 3), sigmaX=0.8)
        blur_applied = True

    # Upscale back to 512 for display (nearest neighbour keeps pixelated look)
    view = cv2.resize(small, (512, 512), interpolation=cv2.INTER_NEAREST)
    return view, px, blur_applied


# ─────────────────────────────────────────────
#  BUILD OUTPUT IMAGE WITH UI OVERLAYS
# ─────────────────────────────────────────────
def build_output_image(view: np.ndarray, altitude_m: float, px: int,
                        blur_applied: bool, found: bool,
                        aruco_id: int, bit_grid) -> np.ndarray:
    """
    Compose final output image:
      - Top bar:    altitude + sensor resolution info
      - Centre:     drone POV view (with bit grid overlay if detected)
      - Bottom bar: green (detected) or red (not detected)
    """
    VIEW_SIZE  = 512
    BAR_HEIGHT = 48

    # Convert grayscale view → BGR for coloured overlays
    canvas = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)

    # Draw reconstructed bit grid overlay on the view if detected
    # if found and bit_grid is not None:
    #     SIZE   = VIEW_SIZE
    #     cell   = SIZE / 6.0
    #     h_grid = [round(i * cell) for i in range(7)]
    #     v_grid = [round(i * cell) for i in range(7)]
    #     for row in range(4):
    #         for col in range(4):
    #             bit = bit_grid[row, col]
    #             if bit < 0:
    #                 continue
    #             x0 = v_grid[col+1]+2; x1 = v_grid[col+2]-2
    #             y0 = h_grid[row+1]+2; y1 = h_grid[row+2]-2
    #             color = (0,230,118) if bit == 1 else (0,120,60)
    #             cv2.rectangle(canvas, (x0,y0), (x1,y1), color, 1)
    #             cv2.putText(canvas, str(bit),
    #                         ((x0+x1)//2-4, (y0+y1)//2+5),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
    #     cv2.putText(canvas, f"ID {aruco_id}",
    #                 (v_grid[1], h_grid[1]-4),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,230,118), 1)

    # ── Top bar ───────────────────────────────────────────────────────
    top_bar = np.zeros((BAR_HEIGHT, VIEW_SIZE, 3), dtype=np.uint8)
    top_bar[:] = (40, 40, 40)   # dark grey

    blur_str = "Applied" if blur_applied else "None"
    top_text = f"Altitude: {altitude_m}m   |   Sensor: {px}x{px} px   |   Atm. Blur: {blur_str}"
    cv2.putText(top_bar, top_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

    # ── Bottom banner ─────────────────────────────────────────────────
    bot_bar = np.zeros((BAR_HEIGHT, VIEW_SIZE, 3), dtype=np.uint8)

    if found:
        bot_bar[:] = (0, 140, 0)    # green
        bot_text   = f"ArUco ID_{aruco_id} DETECTED -> Precision landing zone identified"
        txt_color  = (255, 255, 255)
    else:
        bot_bar[:] = (0, 0, 180)    # red
        bot_text   = f"ArUco ID_{aruco_id} NOT DETECTED -> Continue cruising"
        txt_color  = (255, 255, 255)

    cv2.putText(bot_bar, bot_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, txt_color, 1, cv2.LINE_AA)

    # ── Stack vertically ──────────────────────────────────────────────
    output = np.vstack([top_bar, canvas, bot_bar])
    return output


# ─────────────────────────────────────────────
#  PRINT TERMINAL OUTPUT
# ─────────────────────────────────────────────
def print_terminal(altitude_m, px, blur_applied, found, aruco_id, save_path):
    print(f"\n{'='*58}")
    print(f"  Drone at {altitude_m}m -> Marker occupies {px}x{px} pixels on sensor")
    print(f"  Atmospheric blur: {'Applied' if blur_applied else 'None'}")

    if altitude_m <= 5:
        print(f"  Drone sees: ARUCO MARKER dominant")
    elif altitude_m >= 30:
        print(f"  Drone sees: DECOY IMAGE dominant")
    else:
        print(f"  Drone sees: TRANSITION ZONE (both blending)")

    if found:
        print(f"  ArUco ID_{aruco_id} DETECTED -> Precision landing zone identified")
    else:
        print(f"  ArUco ID_{aruco_id} NOT DETECTED -> Continue cruising")

    print(f"  Image saved -> {save_path}")
    print(f"{'='*58}\n")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    if len(sys.argv) != 4:
        print("Usage: python viewer.py <hybrid_marker.png> <altitude_m> <aruco_id>")
        print("Example: python viewer.py hybrid_marker.png 30 7")
        sys.exit(1)

    image_path = sys.argv[1]
    altitude_m = float(sys.argv[2])
    aruco_id   = int(sys.argv[3])

    # Load hybrid marker
    hybrid = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if hybrid is None:
        print(f"[ERROR] Could not load image: {image_path}")
        sys.exit(1)

    # Simulate drone view
    view, px, blur_applied = simulate_drone_view(hybrid, altitude_m)

    # Detect ArUco on the simulated view
    found, bit_grid = detect_aruco(view, aruco_id)

    # Build output image with overlays
    output = build_output_image(view, altitude_m, px, blur_applied,
                                 found, aruco_id, bit_grid)

    # Save
    save_path = f"drone_view_{altitude_m}m.png"
    cv2.imwrite(save_path, output)

    # Print terminal output
    print_terminal(altitude_m, px, blur_applied, found, aruco_id, save_path)

    # Display
    cv2.imshow(f"Drone View @ {altitude_m}m — ArUco ID {aruco_id}", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
#  SAMPLE TESTING — run all key altitudes
# ─────────────────────────────────────────────
def run_sample_tests(image_path: str, aruco_id: int):
    """
    Automatically runs viewer at all key test altitudes and
    displays them in a single tiled window for quick visual comparison.
    """
    TEST_ALTITUDES = [2, 5, 15, 30, 100]

    hybrid = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if hybrid is None:
        print(f"[ERROR] Could not load image: {image_path}")
        sys.exit(1)

    print(f"\n{'='*58}")
    print(f"  SAMPLE TEST — running all key altitudes")
    print(f"  Image: {image_path}   |   ArUco ID: {aruco_id}")
    print(f"{'='*58}")

    output_frames = []

    for alt in TEST_ALTITUDES:
        view, px, blur_applied = simulate_drone_view(hybrid, alt)
        found, bit_grid        = detect_aruco(view, aruco_id)
        frame                  = build_output_image(view, alt, px, blur_applied,
                                                     found, aruco_id, bit_grid)
        save_path              = f"drone_view_{alt}.0m.png"
        cv2.imwrite(save_path, frame)
        print_terminal(alt, px, blur_applied, found, aruco_id, save_path)
        output_frames.append(frame)

    # ── Tile all frames into one display window ───────────────────────
    # Each frame is 512 × (48+512+48) = 512 × 608
    # Arrange as a 1-row strip: [2m | 5m | 15m | 30m | 100m]
    FRAME_W = output_frames[0].shape[1]   # 512
    FRAME_H = output_frames[0].shape[0]   # 608

    # Add altitude label above each frame
    labelled = []
    for i, (alt, frame) in enumerate(zip(TEST_ALTITUDES, output_frames)):
        label_bar = np.zeros((30, FRAME_W, 3), dtype=np.uint8)
        label_bar[:] = (60, 60, 60)
        cv2.putText(label_bar, f"  {alt}m", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        labelled.append(np.vstack([label_bar, frame]))

    tiled = np.hstack(labelled)

    # Resize tiled image to fit screen if too wide
    screen_w = 1800
    if tiled.shape[1] > screen_w:
        scale  = screen_w / tiled.shape[1]
        tiled  = cv2.resize(tiled, (screen_w, int(tiled.shape[0] * scale)))

    print("\n  Displaying tiled sample test window...")
    print("  Press any key to close.\n")
    cv2.imshow("MIRAGE Sample Test — All Altitudes (2m | 5m | 15m | 30m | 100m)", tiled)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    #  python viewer.py hybrid_marker.png 30 7        → single altitude
    #  python viewer.py hybrid_marker.png --test 7    → all sample altitudes
    #  python viewer.py hybrid_marker.png 7           → shorthand for --test

    if len(sys.argv) == 4 and sys.argv[2] == "--test":
        # python viewer.py hybrid_marker.png --test 7
        run_sample_tests(sys.argv[1], int(sys.argv[3]))

    elif len(sys.argv) == 4:
        # python viewer.py hybrid_marker.png 30 7
        main()

    elif len(sys.argv) == 3:
        # python viewer.py hybrid_marker.png 7  → run all sample altitudes
        run_sample_tests(sys.argv[1], int(sys.argv[2]))

    else:
        print("Usage:")
        print("  Single altitude : python viewer.py hybrid_marker.png 30 7")
        print("  Sample test all : python viewer.py hybrid_marker.png --test 7")
        sys.exit(1)