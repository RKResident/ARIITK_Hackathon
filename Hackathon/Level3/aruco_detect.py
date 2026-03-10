"""
aruco_detect.py — Project MIRAGE
Team Aerial Robotics IITK | Y25 Recruitment Hackathon

Shared ArUco detection module using the edge-integration pipeline.
Replaces OpenCV's standard centre-sampling detector with an approach
that works on hybrid markers, where only cell boundaries carry signal.

Public API
----------
detect_aruco(img_gray, aruco_id) -> (found: bool, bit_grid: np.ndarray)

    img_gray  : uint8 grayscale image, marker must fill the frame exactly,
                axis-aligned (drone nadir view). Any square size works.
    aruco_id  : int in 0..49  (DICT_4X4_50)

    Returns:
        found     : True if the reconstructed bit grid matches aruco_id
        bit_grid  : 4×4 int array of 0/1 (-1 = unknown cell)

Pipeline
--------
  1. Contrast stretch + unsharp mask
  2. Sobel edge map
  3. 6×6 grid by arithmetic  (line i = round(i × SIZE / 6))
  4. Sample Sobel magnitude at each of the 28 cell boundaries
  5. 1-D k-means (k=2) → threshold → transition map
  6. Border anchor  (border ring = always 0, ArUco invariant)
  7. Flip-on-crossing integration → 4×4 bit grid
  8. Match bit grid against DICT_4X4_50 with Hamming distance
"""

import cv2
import numpy as np


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sample_strip(mag: np.ndarray, x0, y0, x1, y1, half_w: int) -> float:
    """Mean Sobel magnitude in a strip of width 2×half_w along (x0,y0)→(x1,y1)."""
    H, W   = mag.shape
    length = max(abs(x1 - x0), abs(y1 - y0), 1)
    vals   = []
    for t in np.linspace(0, 1, max(length, 2)):
        cx = x0 + t * (x1 - x0)
        cy = y0 + t * (y1 - y0)
        dx, dy = (y1 - y0) / length, -(x1 - x0) / length   # perpendicular
        for d in np.linspace(-half_w, half_w, max(half_w * 2, 2)):
            px = int(round(cx + d * dx))
            py = int(round(cy + d * dy))
            if 0 <= px < W and 0 <= py < H:
                vals.append(mag[py, px])
    return float(np.mean(vals)) if vals else 0.0


def _kmeans_threshold(values: np.ndarray) -> float:
    """
    1-D k-means with k=2, initialised at min and max.
    Returns midpoint between the two converged cluster centres.
    """
    c_low  = float(values.min())
    c_high = float(values.max())
    for _ in range(100):
        mid      = (c_low + c_high) / 2.0
        labels   = values > mid
        new_low  = float(values[~labels].mean()) if (~labels).any() else c_low
        new_high = float(values[ labels].mean()) if  labels.any()  else c_high
        if abs(new_low - c_low) < 1e-4 and abs(new_high - c_high) < 1e-4:
            break
        c_low, c_high = new_low, new_high
    return (c_low + c_high) / 2.0


def _bits_to_id(bit_grid: np.ndarray, aruco_id: int) -> bool:
    """
    Compare a 4×4 bit grid against all 4 rotations of aruco_id
    in DICT_4X4_50 using Hamming distance.
    Returns True if any rotation matches within errorCorrectionRate=1 bits.
    """
    if (bit_grid < 0).any():
        return False   # unknown cells → can't match

    adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # Extract the reference 4×4 bit pattern for aruco_id.
    # generateImageMarker gives a (size×size) image; we read its inner grid.
    ref_img  = cv2.aruco.generateImageMarker(adict, aruco_id, 60)
    ref_cell = 60 // 6
    ref_bits = np.zeros((4, 4), dtype=int)
    for r in range(4):
        for c in range(4):
            x0 = (c + 1) * ref_cell + ref_cell // 4
            y0 = (r + 1) * ref_cell + ref_cell // 4
            x1 = (c + 2) * ref_cell - ref_cell // 4
            y1 = (r + 2) * ref_cell - ref_cell // 4
            patch = ref_img[y0:y1, x0:x1]
            ref_bits[r, c] = 1 if patch.mean() > 128 else 0

    # Try all 4 rotations of the detected grid
    grid = bit_grid.copy()
    for _ in range(4):
        hamming = int(np.sum(grid != ref_bits))
        if hamming <= 1:          # errorCorrectionRate = 1 bit
            return True
        grid = np.rot90(grid)

    return False


# ── Public API ────────────────────────────────────────────────────────────────

def detect_aruco(img_gray: np.ndarray, aruco_id: int):
    """
    Detect ArUco marker using edge-integration on a hybrid marker image.

    Parameters
    ----------
    img_gray  : uint8 grayscale, marker fills frame exactly, no tilt.
    aruco_id  : target ID to match (0–49).

    Returns
    -------
    found     : bool
    bit_grid  : 4×4 int ndarray (0/1, or -1 for unknown cells)
    """
    SIZE = img_gray.shape[0]

    # ── 1: Contrast stretch + unsharp mask ───────────────────────────────────
    stretched = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX)
    blurred   = cv2.GaussianBlur(stretched, (0, 0), sigmaX=2)
    sharpened = cv2.addWeighted(stretched, 2.0, blurred, -1.0, 0)

    # ── 2: Sobel edge map ─────────────────────────────────────────────────────
    sx  = cv2.Sobel(sharpened, cv2.CV_32F, 1, 0, ksize=3)
    sy  = cv2.Sobel(sharpened, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2)

    # ── 3: Axis-aligned 6×6 grid by arithmetic ───────────────────────────────
    cell   = SIZE / 6.0
    h_grid = [round(i * cell) for i in range(7)]
    v_grid = [round(i * cell) for i in range(7)]
    cell_h = h_grid[1] - h_grid[0]
    cell_w = v_grid[1] - v_grid[0]
    HALF_W = max(2, min(cell_w, cell_h) // 5)

    # ── 4: Sample boundary strengths ─────────────────────────────────────────
    # h_strength[r, c] : vertical boundary between data col c and c+1, row r
    # v_strength[r, c] : horizontal boundary between data row r and r+1, col c
    h_strength = np.zeros((4, 3), dtype=np.float32)
    for row in range(4):
        for bnd in range(3):
            h_strength[row, bnd] = _sample_strip(
                mag,
                v_grid[bnd + 2], h_grid[row + 1],
                v_grid[bnd + 2], h_grid[row + 2],
                HALF_W)

    v_strength = np.zeros((3, 4), dtype=np.float32)
    for col in range(4):
        for bnd in range(3):
            v_strength[bnd, col] = _sample_strip(
                mag,
                v_grid[col + 1], h_grid[bnd + 2],
                v_grid[col + 2], h_grid[bnd + 2],
                HALF_W)

    # Seed row: boundary between top border ring and data row 0
    seed_raw = np.array([
        _sample_strip(mag, v_grid[col + 1], h_grid[1],
                           v_grid[col + 2], h_grid[1], HALF_W)
        for col in range(4)
    ], dtype=np.float32)

    # ── 5: K-means threshold ─────────────────────────────────────────────────
    all_strengths = np.concatenate([
        h_strength.flatten(),
        v_strength.flatten(),
        seed_raw
    ])
    thresh = _kmeans_threshold(all_strengths)

    h_edge     = h_strength > thresh          # (4, 3) bool
    v_edge     = v_strength > thresh          # (3, 4) bool
    seed_edges = seed_raw > thresh            # (4,)   bool

    # ── 6 + 7: Border anchor → flip-on-crossing integration ─────────────────
    # Border ring = always 0 (ArUco invariant).
    # Each crossing flips the running bit; no crossing keeps it.
    bit_grid = np.full((4, 4), -1, dtype=int)

    # Seed first data row from border→data boundary
    for col in range(4):
        bit_grid[0, col] = 1 if seed_edges[col] else 0

    # Propagate downward column by column
    for row in range(3):
        for col in range(4):
            if bit_grid[row, col] >= 0:
                bit_grid[row + 1, col] = bit_grid[row, col] ^ int(v_edge[row, col])

    # Cross-fill horizontally to recover any missed cells
    for row in range(4):
        for col in range(3):
            if bit_grid[row, col] >= 0 and bit_grid[row, col + 1] < 0:
                bit_grid[row, col + 1] = bit_grid[row, col] ^ int(h_edge[row, col])

    # ── 8: Match against dictionary ──────────────────────────────────────────
    found = _bits_to_id(bit_grid, aruco_id)

    return found, bit_grid