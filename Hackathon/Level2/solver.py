"""


YOUR MISSION
─────────────
A drone is descending automatically toward the ground.
A landing platform oscillates sinusoidally (SHM) — it never stops moving.
You must write an autonomous controller that:

  PHASE 1 — SEARCH:
    The drone starts near the top-left of the arena, far from the platform.
    Design a search pattern to sweep the arena until the ArUco marker on
    the platform enters the camera's field of view.

  PHASE 2 — TRACK & LAND:
    Once detected, use a PID controller to keep the drone aligned with
    the moving platform as it descends.
    Success = final horizontal error <= 5 cm at touchdown.

HOW IT WORKS
─────────────
  This solver directly imports the simulator environment from simulator.py.
  Each frame you call sim.step_env(vx, vy) to send velocity commands and
  receive the camera image back. No files, no keyboard.

  Run with a SINGLE command:
      python solver.py

  The simulation window opens automatically.

ENVIRONMENT API  (this is your only interface — read carefully)
────────────────────────────────────────────────────────────────
  sim = make_sim()
        Creates the simulator. Do NOT touch the sim object directly.

  pixels, done = sim.step_env(vx, vy)
        Advances physics by one frame and returns:
          pixels  list[int]  Flat 10 000-element grayscale image (100x100).
                             Row-major order: index = row * 100 + col.
                             Values 0-255.
          done    bool       True when the drone has reached the ground.

        Your inputs:
          vx  float   Horizontal velocity command (m/s). Positive = RIGHT.
          vy  float   Vertical   velocity command (m/s). Positive = DOWN.
          Speed is capped at 5 m/s by the simulator.

  sim.drone_altitude  float   Current altitude in metres (10.0 -> 0.0).
  sim.fov_m           float   Camera ground footprint in metres at this altitude.
                              fov_m = 0.30 x altitude.
                              Use this to convert image offsets -> real metres.

CAMERA IMAGE FORMAT
────────────────────
  pixels is a flat list of 10 000 integers (0-255), row-major.

  What you will see:
    Grass / background  ->  dark gray    (~45-90)
    Platform surface    ->  bright gray  (~200-230)
    Inner ArUco square  ->  near black   (~0-20)  at the centre of the platform

  The platform appears as a bright rectangular cluster with a dark centre.
  It may be partially visible (clipped at the image edge) or absent entirely.

YOUR TASKS
───────────
  TODO 1 — Tune SEARCH_SPEED
  TODO 2 — Tune PID gains  (Kp, Ki, Kd)
  TODO 3 — Implement detect_platform()   [ArUco / bright-blob detection]
  TODO 4 — Implement PID.update()        [P + I + D terms]
  TODO 5 — Implement search_velocity()   [your search pattern]
  TODO 6 — Wire up the error -> PID -> velocity in main()

GRADING
────────
  Final error <= 0.05 m  (SUCCESS)        
  Bonus: error <= 0.02 m                     

RULES
──────
  OK   Only edit THIS file.
  OK   Tune any constant marked TODO.
  NO   Do NOT modify simulator_level2.py.
"""

import math
import sys

# ════════════════════════════════════════════════════════════════════
#  ENVIRONMENT BRIDGE
#  Imports the simulator and exposes a clean autonomous API.
#  ── DO NOT MODIFY THIS SECTION ──
# ════════════════════════════════════════════════════════════════════

try:
    import simulator_level2 as _sim_module
    import pygame
except ImportError as e:
    print(f"[ERROR] Could not import simulator: {e}")
    print("Ensure simulator.py is in the same directory.")
    sys.exit(1)


def make_sim():
    """Create the simulation environment. Keyboard control is disabled."""
    return _SimEnv()


class _SimEnv:
    """
    Wraps DroneSim in autonomous mode.
    Keyboard overrides are suppressed every frame — only your vx/vy commands
    control the drone.
    """

    CAM_RESOLUTION       = 100
    CAM_GROUND_SIZE_BASE = 0.30
    INITIAL_ALTITUDE     = 10.0
    SIM_TIME_SEC         = 35
    PIXELS_PER_METER     = 100
    FPS                  = 30

    def __init__(self):
        self._sim = _sim_module.DroneSim(mode=_sim_module.DroneSim.MODE_EXTERNAL)

    @property
    def drone_altitude(self):
        """Current altitude in metres."""
        return self._sim.drone_altitude

    @property
    def fov_m(self):
        """Camera ground footprint in metres (shrinks as drone descends)."""
        return self.CAM_GROUND_SIZE_BASE * max(0.1, self._sim.drone_altitude)

    def step_env(self, vx: float, vy: float):
        """
        Send a velocity command and advance the simulation by one frame.

        Returns:
            pixels  list[int]  Flat 100x100 grayscale image (10 000 values).
            done    bool       True when altitude reaches zero.
        """
        s  = self._sim
        dt = 1.0 / self.FPS

        # Keep pygame window responsive; ESC exits cleanly
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit(0)

        # Suppress keyboard — autonomous only
        s._kb_vx = 0.0
        s._kb_vy = 0.0

        # Physics
        s.update_platform(dt)
        s.drone_altitude -= s.descent_rate * dt
        s.generate_camera_feed()

        spd = math.hypot(vx, vy)
        if spd > 5.0:
            vx *= 5.0 / spd
            vy *= 5.0 / spd

        s.drone_x += vx * self.PIXELS_PER_METER * dt
        s.drone_y += vy * self.PIXELS_PER_METER * dt
        s.drone_x  = max(20, min(s.W - 20, s.drone_x))
        s.drone_y  = max(20, min(s.H - 20, s.drone_y))

        err = math.hypot(s.drone_x - s.plat_x,
                         s.drone_y - s.plat_y) / self.PIXELS_PER_METER
        s.error_history.append(err)
        if len(s.error_history) > s.max_history:
            s.error_history.pop(0)

        s.trail.append((s.drone_x, s.drone_y))
        if len(s.trail) > s.trail_max_len:
            s.trail.pop(0)

        s.step += 1
        s._render()
        pygame.display.flip()
        s.clock.tick(self.FPS)

        # Convert camera surface -> flat pixel list
        pixels = []
        if s._last_cam_surface is not None:
            res = self.CAM_RESOLUTION
            for row in range(res):
                for col in range(res):
                    r, g, b, *_ = s._last_cam_surface.get_at((col, row))
                    pixels.append(int(0.299 * r + 0.587 * g + 0.114 * b))

        done = s.drone_altitude <= 0
        if done:
            final_err = math.hypot(s.drone_x - s.plat_x,
                                   s.drone_y - s.plat_y) / self.PIXELS_PER_METER
            success = final_err <= _sim_module.LANDING_SUCCESS_M
            s._render_result(success, final_err)
            pygame.display.flip()
            pygame.time.wait(4000)
            pygame.quit()

        return pixels, done


# ════════════════════════════════════════════════════════════════════
#  SECTION 1 — CONFIGURATION
# ════════════════════════════════════════════════════════════════════

SEARCH_SPEED = 3.5      # m/s during search — fast beeline then spiral

# Brightness cut-off for platform detection — pre-calibrated, do not change.
BRIGHT_THRESHOLD = 150

# PID gains — same for both axes (symmetric dynamics)
# Kp high enough to chase 1.8 m/s platform at small error margins.
# Ki = 0: platform never stops moving so integral only winds up.
# Kd damps overshoot; D term now does the rate-of-change work EMA was doing.
KP_X = 20
KI_X = 0.0
KD_X = 0.4

KP_Y = 20
KI_Y = 0.0
KD_Y = 0.4

OUTPUT_LIMIT = 4.0      # Max velocity the PID may output (m/s). Do not change.

# Platform velocity EMA smoothing factor (0=no memory, 1=no update)
# ── REMOVED: EMA velocity prediction eliminated (see tracking section) ──

# Max plausible frame-to-frame error delta before spike rejection discards it.
# Used to protect the D term from single-frame detection glitches.
MAX_VEL_SPIKE = 8.0   # m/s

# Overflow detection hysteresis thresholds (in fraction of resolution)
# Enter overflow mode when plat_px >= OVERFLOW_ENTER * resolution,
# exit only when plat_px <  OVERFLOW_EXIT  * resolution.
# Gap between thresholds prevents flickering at the boundary.
OVERFLOW_ENTER = 0.85
OVERFLOW_EXIT  = 0.75

# Hard cap on half_plat used in corner-offset calculation.
# Prevents a 1-px corner error from producing a huge centre estimate jump.
HALF_PLAT_MAX_PX = 40.0

# Arena center in pixel space (used for beeline target)
# Arena width ~830px (1100 - 268 panel - margins), height ~820px
# Drone starts at (80, 180). Center of arena is approximately (470, 410).
ARENA_CENTER_X_PX = 470.0
ARENA_CENTER_Y_PX = 410.0
PIXELS_PER_METER  = 100.0

# Spiral parameters
SPIRAL_ANGULAR_SPEED = 1.2   # rad/s — how fast the spiral rotates
SPIRAL_EXPAND_RATE   = 0.18  # m per radian — how fast the radius grows


# ════════════════════════════════════════════════════════════════════
#  SECTION 2 — PID CONTROLLER
# ════════════════════════════════════════════════════════════════════

class PID:
    """Single-axis PID controller."""

    def __init__(self, kp, ki, kd, limit=OUTPUT_LIMIT):
        self.kp    = kp
        self.ki    = ki
        self.kd    = kd
        self.limit = limit

        self._integral   = 0.0
        self._prev_error = 0.0

    def reset(self):
        """Zero internal state. Call once when the platform is first detected."""
        self._integral   = 0.0
        self._prev_error = 0.0

    def update(self, error, dt):
        """
        Compute PID output for the given error and time-step.

        Args:
            error (float): Signed position error in metres.
                           Positive -> platform is to the RIGHT / BELOW centre.
            dt    (float): Seconds since the last call.

        Returns:
            float: Velocity command in m/s, clamped to +/- self.limit.
        """
        if dt <= 0:
            return 0.0

        proportional_term = self.kp * error

        self._integral   += error * dt
        integral_term     = self.ki * self._integral

        derivative        = (error - self._prev_error) / dt
        derivative_term   = self.kd * derivative

        self._prev_error  = error          # ← bug-fix: must update prev_error

        output = proportional_term + integral_term + derivative_term
        return max(-self.limit, min(self.limit, output))


# ════════════════════════════════════════════════════════════════════
#  SECTION 3 — PLATFORM DETECTION
# ════════════════════════════════════════════════════════════════════

def detect_platform(pixels, fov_m, in_overflow=False,
                    resolution=100, threshold=BRIGHT_THRESHOLD):
    """
    Locate the landing platform in the camera image using corner detection.

    Strategy:
      1. Collect all bright pixels (above threshold).
      2. Compute plat_px — expected platform size in pixels at this altitude.
      3. Overflow check (with hysteresis):
           - Already in overflow mode  → stay until plat_px < OVERFLOW_EXIT * res
           - Not in overflow mode      → enter  when plat_px >= OVERFLOW_ENTER * res
         In overflow mode use the dark ArUco centroid directly.
      4. Fully-visible check: blob spans >= 70% of plat_px → ArUco centroid.
      5. Partial visibility: find the extreme corner furthest from image centre,
         offset by half_plat (capped at HALF_PLAT_MAX_PX) to recover true centre.

    Args:
        pixels      list[int]   Flat 10 000-element grayscale image.
        fov_m       float       Camera ground footprint in metres at this altitude.
        in_overflow bool        Whether we were in overflow mode last frame
                                (caller must pass and store this flag).
        resolution  int         Image width = height = 100.
        threshold   int         Brightness cut-off.

    Returns:
        (found, cx_norm, cy_norm, new_overflow, raw_px)
          found        bool             True if platform detected.
          cx_norm      float            Horizontal offset from image centre [-1, +1].
          cy_norm      float            Vertical   offset from image centre [-1, +1].
          new_overflow bool             Updated overflow state — pass back in next call.
          raw_px       (float,float)|None
                                        Raw ArUco centroid in pixel coords (cx_px, cy_px)
                                        when new_overflow=True, else None.
                                        Use pixel_to_m = fov_m/resolution to convert
                                        directly to metres — avoids half_fov compression.
    """
    half = resolution / 2.0

    # ── Step 1: collect bright pixel coordinates ──────────────────────────
    bright_cols = []
    bright_rows = []
    for row in range(resolution):
        for col in range(resolution):
            if pixels[row * resolution + col] > threshold:
                bright_cols.append(col)
                bright_rows.append(row)

    if len(bright_cols) < 5:
        return False, 0.0, 0.0, in_overflow, None

    min_col = min(bright_cols)
    max_col = max(bright_cols)
    min_row = min(bright_rows)
    max_row = max(bright_rows)

    # ── Step 2: platform and ArUco pixel sizes at current altitude ───────────
    plat_px   = (1.0 / fov_m) * resolution
    # Cap half_plat so a 1-px corner error can't produce a huge jump.
    half_plat = min(plat_px / 2.0, HALF_PLAT_MAX_PX)

    # ArUco inner square is exactly half the platform size (0.5m × 0.5m).
    # Its centre == platform centre.
    # half_aruco_px = pixels spanning half the ArUco square side.
    # Capped at 40px for the same reason as half_plat.
    half_aruco_px = min((0.5 / fov_m) * resolution / 2.0, HALF_PLAT_MAX_PX)

    # ── Step 3: overflow check with hysteresis ────────────────────────────
    if in_overflow:
        new_overflow = plat_px >= OVERFLOW_EXIT * resolution
    else:
        new_overflow = plat_px >= OVERFLOW_ENTER * resolution

    if new_overflow:
        # ── ArUco corner detection ────────────────────────────────────────
        # At overflow the platform fills the frame and the ArUco square
        # itself may also overflow.  The centroid is unreliable when the
        # ArUco is partially off-screen (it gets pulled toward whichever
        # side has more dark pixels).  Instead use the same corner approach
        # as for the bright platform: find the extreme dark corner furthest
        # from image centre and offset inward by half_aruco_px.
        dark_cols, dark_rows = [], []
        for row in range(resolution):
            for col in range(resolution):
                if pixels[row * resolution + col] < 25:
                    dark_cols.append(col)
                    dark_rows.append(row)

        if len(dark_cols) < 3:
            # No ArUco visible at all — hold last command
            return False, 0.0, 0.0, new_overflow, None

        dark_min_col = min(dark_cols)
        dark_max_col = max(dark_cols)
        dark_min_row = min(dark_rows)
        dark_max_row = max(dark_rows)

        # Four extreme corners of the dark blob
        aruco_candidates = [
            (dark_min_col, dark_min_row),   # top-left
            (dark_max_col, dark_min_row),   # top-right
            (dark_min_col, dark_max_row),   # bottom-left
            (dark_max_col, dark_max_row),   # bottom-right
        ]

        # Pick corner furthest from image centre — most likely a true
        # geometric corner of the ArUco square, not an interior dark pixel.
        def dist_from_centre(pt):
            return math.hypot(pt[0] - half, pt[1] - half)

        aruco_corner_col, aruco_corner_row = max(aruco_candidates,
                                                  key=dist_from_centre)

        # Offset toward image centre by half_aruco_px to recover ArUco centre
        # (== platform centre).
        if aruco_corner_col <= half:
            cx_px = aruco_corner_col + half_aruco_px
        else:
            cx_px = aruco_corner_col - half_aruco_px

        if aruco_corner_row <= half:
            cy_px = aruco_corner_row + half_aruco_px
        else:
            cy_px = aruco_corner_row - half_aruco_px

        cx_px = max(0.0, min(resolution - 1.0, cx_px))
        cy_px = max(0.0, min(resolution - 1.0, cy_px))

        cx_norm = (cx_px - half) / half
        cy_norm = (cy_px - half) / half

        # Return raw pixel coords for uncompressed metre conversion in main()
        return True, cx_norm, cy_norm, new_overflow, (cx_px, cy_px)

    # ── Step 4: fully-visible check ───────────────────────────────────────
    blob_w = max_col - min_col
    blob_h = max_row - min_row
    fully_visible = (blob_w >= 0.7 * plat_px) and (blob_h >= 0.7 * plat_px)

    if fully_visible:
        dark_cols, dark_rows = [], []
        for row in range(resolution):
            for col in range(resolution):
                if pixels[row * resolution + col] < 25:
                    dark_cols.append(col)
                    dark_rows.append(row)

        if len(dark_cols) >= 3:
            cx_px = sum(dark_cols) / len(dark_cols)
            cy_px = sum(dark_rows) / len(dark_rows)
        else:
            # ArUco not distinct yet — use bright blob centre
            cx_px = (min_col + max_col) / 2.0
            cy_px = (min_row + max_row) / 2.0

        cx_norm = (cx_px - half) / half
        cy_norm = (cy_px - half) / half
        return True, cx_norm, cy_norm, new_overflow, None

    # ── Step 5: partial visibility — corner detection ─────────────────────
    candidates = [
        (min_col, min_row),   # top-left  of blob
        (max_col, min_row),   # top-right
        (min_col, max_row),   # bottom-left
        (max_col, max_row),   # bottom-right
    ]

    def dist_from_centre(pt):
        return math.hypot(pt[0] - half, pt[1] - half)

    corner_col, corner_row = max(candidates, key=dist_from_centre)

    # Offset toward image centre by half_plat to recover true platform centre.
    if corner_col <= half:
        cx_centre = corner_col + half_plat
    else:
        cx_centre = corner_col - half_plat

    if corner_row <= half:
        cy_centre = corner_row + half_plat
    else:
        cy_centre = corner_row - half_plat

    cx_norm = (cx_centre - half) / half
    cy_norm = (cy_centre - half) / half

    cx_norm = max(-1.0, min(1.0, cx_norm))
    cy_norm = max(-1.0, min(1.0, cy_norm))

    return True, cx_norm, cy_norm, new_overflow, None


# ════════════════════════════════════════════════════════════════════
#  SECTION 4 — SEARCH STRATEGY
# ════════════════════════════════════════════════════════════════════

# Drone starts at pixel (80, 180).
# Arena centre is at approximately pixel (470, 410).
# Convert the displacement to metres for the beeline.
_BEELINE_DX_M = (ARENA_CENTER_X_PX - 80.0)  / PIXELS_PER_METER   #  3.9 m right
_BEELINE_DY_M = (ARENA_CENTER_Y_PX - 180.0) / PIXELS_PER_METER   #  2.3 m down
_BEELINE_DIST = math.hypot(_BEELINE_DX_M, _BEELINE_DY_M)         # total distance
_BEELINE_TIME = _BEELINE_DIST / SEARCH_SPEED                      # seconds to arrive


def search_velocity(search_timer, search_angle, search_radius, dt):
    """
    Two-phase search:
      Phase A (beeline): fly straight from start to arena centre.
      Phase B (spiral):  expand an Archimedean spiral around the centre.

    Args:
        search_timer  float   Total seconds spent in search phase so far.
        search_angle  float   Persistent angle state (radians).
        search_radius float   Persistent radius state (metres).
        dt            float   Frame time (~0.033 s).

    Returns:
        (vx, vy, new_angle, new_radius)
    """

    if search_timer < _BEELINE_TIME:
        # ── Phase A: beeline to centre ────────────────────────────────────
        vx = SEARCH_SPEED * (_BEELINE_DX_M / _BEELINE_DIST)
        vy = SEARCH_SPEED * (_BEELINE_DY_M / _BEELINE_DIST)
        return vx, vy, search_angle, search_radius

    else:
        # ── Phase B: Archimedean spiral ───────────────────────────────────
        # Angle advances at a fixed angular speed.
        # Radius grows proportionally to angle so loops stay spaced.
        new_angle  = search_angle  + SPIRAL_ANGULAR_SPEED * dt
        new_radius = new_angle * SPIRAL_EXPAND_RATE        # r = k * theta

        # Velocity = tangential (perpendicular to radius) + radial (outward).
        # For a spiral r = k*theta:  dr/dt = k * omega,  v_tan = r * omega.
        r_dot   = SPIRAL_EXPAND_RATE * SPIRAL_ANGULAR_SPEED          # radial speed
        v_tan   = new_radius          * SPIRAL_ANGULAR_SPEED          # tangential speed

        vx = v_tan * (-math.sin(new_angle)) + r_dot * math.cos(new_angle)
        vy = v_tan * ( math.cos(new_angle)) + r_dot * math.sin(new_angle)

        # Normalise to SEARCH_SPEED so the drone moves at a consistent pace.
        spd = math.hypot(vx, vy)
        if spd > 0.01:
            vx = vx / spd * SEARCH_SPEED
            vy = vy / spd * SEARCH_SPEED

        return vx, vy, new_angle, new_radius


# ════════════════════════════════════════════════════════════════════
#  SECTION 5 — MAIN CONTROL LOOP
# ════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Operation Touchdown — ART IITK")
    print("  Initialising simulation...")
    print("=" * 60)

    sim   = make_sim()
    pid_x = PID(KP_X, KI_X, KD_X)
    pid_y = PID(KP_Y, KI_Y, KD_Y)

    aruco_found   = False
    search_phase  = True
    search_angle  = 0.0
    search_radius = 0.0
    search_timer  = 0.0

    # Last good velocity command — held when platform is temporarily lost
    last_vx = 0.0
    last_vy = 0.0

    # Overflow state — persisted across frames for hysteresis
    in_overflow = False

    dt   = 1.0 / sim.FPS
    step = 0

    # First frame — hover to initialise the camera image
    pixels, done = sim.step_env(0.0, 0.0)

    while not done:

        # ── Detect platform ────────────────────────────────────────────────
        found, cx_norm, cy_norm, in_overflow, raw_px = detect_platform(
            pixels, sim.fov_m, in_overflow
        )

        # ── Switch phases on first detection ──────────────────────────────
        if found and not aruco_found:
            print(f"  [DETECT] Platform acquired at step {step} "
                  f"({step / sim.FPS:.1f} s)  ->  PID tracking engaged.")
            aruco_found  = True
            search_phase = False
            pid_x.reset()
            pid_y.reset()

        # ── PHASE 1: Search ────────────────────────────────────────────────
        if search_phase:
            search_timer += dt
            vx, vy, search_angle, search_radius = search_velocity(
                search_timer, search_angle, search_radius, dt
            )
            if step % sim.FPS == 0:
                print(f"  [SEARCH]  step={step:4d}  t={search_timer:.1f}s  "
                      f"alt={sim.drone_altitude:.2f} m")

        # ── PHASE 2: PID Track & Land ──────────────────────────────────────
        elif aruco_found:
            if not found:
                # Platform temporarily out of sight — hold last good command.
                # Do NOT update prev_err or velocity estimate (would corrupt EMA).
                vx, vy = last_vx, last_vy
                if step % sim.FPS == 0:
                    print(f"  [LOST]    step={step:4d}  holding cmd=({vx:+.3f}, {vy:+.3f})  "
                          f"alt={sim.drone_altitude:.2f} m")
            else:
                # ── Convert detection to real-world metres ────────────────────
                if in_overflow and raw_px is not None:
                    # Overflow mode: platform exceeds FOV.
                    # Use raw ArUco pixel position directly — avoids half_fov
                    # compression that caps error at ±half_fov regardless of
                    # true offset.  1 px = fov_m / resolution metres.
                    pixel_to_m = sim.fov_m / 100.0
                    err_x_m    = (raw_px[0] - 50.0) * pixel_to_m
                    err_y_m    = (raw_px[1] - 50.0) * pixel_to_m
                else:
                    # Normal mode: normalised offset is valid.
                    half_fov = sim.fov_m / 2.0
                    err_x_m  = cx_norm * half_fov
                    err_y_m  = cy_norm * half_fov

                # ── PID commands ───────────────────────────────────────────────
                # No velocity prediction — Kd handles rate-of-change, and EMA
                # was contaminated by drone's own motion anyway.
                # Spike rejection: if the error jumped implausibly large in one
                # frame (detection glitch), skip this PID update and hold last cmd.
                skip = False
                if pid_x._prev_error != 0.0 or pid_y._prev_error != 0.0:
                    delta_x = abs(err_x_m - pid_x._prev_error)
                    delta_y = abs(err_y_m - pid_y._prev_error)
                    if delta_x > MAX_VEL_SPIKE * dt or delta_y > MAX_VEL_SPIKE * dt:
                        skip = True
                        if step % sim.FPS == 0:
                            print(f"  [SPIKE]   step={step:4d}  "
                                  f"delta=({delta_x:+.3f}, {delta_y:+.3f}) m  skipped")

                if not skip:
                    vx = pid_x.update(err_x_m, dt)
                    vy = pid_y.update(err_y_m, dt)
                    last_vx = vx
                    last_vy = vy
                else:
                    vx, vy = last_vx, last_vy

                if step % sim.FPS == 0:
                    print(f"  [TRACK]   step={step:4d}  "
                          f"err=({err_x_m:+.3f}, {err_y_m:+.3f}) m  "
                          f"cmd=({vx:+.3f}, {vy:+.3f}) m/s  "
                          f"alt={sim.drone_altitude:.2f} m  "
                          f"{'[OVF]' if in_overflow else ''}")

        # ── Platform not yet found — hover ─────────────────────────────────
        else:
            vx, vy = 0.0, 0.0

        # ── Send command and receive next frame ────────────────────────────
        pixels, done = sim.step_env(vx, vy)
        step += 1

    print(f"\n  Simulation ended after {step} steps.")
    print("  Check the window for your final score.")


if __name__ == "__main__":
    main()