# Level 2 — Operation Touchdown: Precision Landing

> **Team Aerial Robotics | Y25 Recruitment Hackathon**

---

## 1. Context

Welcome to **Operation Touchdown** — the next escalation in the SKYE-X series. Level 1 tested your ability to *find and pursue* a target. Level 2 tests something far more precise: **landing your drone on a moving platform using only a downward-facing camera**.

The drone is already descending. You cannot stop it. You must steer it laterally so that it touches down on the platform center — while the platform itself is moving beneath you.

---

## 2. The Environment

### Arena & Physics

| Parameter | Value |
|-----------|-------|
| World Size | `800 × 800` pixels (1 px = 1 cm, scale: 100 px/m) |
| Simulation Duration | `20 seconds` (fixed descent time) |
| Frame Rate | `30 FPS` |
| Landing Success Radius | `0.35 m` from platform center |

### The Drone

| Parameter | Value |
|-----------|-------|
| Starting Position | `(475, 300)` — offset from platform |
| Starting Altitude | `10.0 m` |
| Descent Rate | Fixed: `0.5 m/s` (auto-descending, you cannot control altitude) |
| Max Lateral Speed | `~1.2 m/s` (controller-limited) |

### The Landing Platform

| Parameter | Value |
|-----------|-------|
| Size | `1m × 1m` square |
| Initial Position | `(400, 400)` — center of arena |
| Movement | Horizontal oscillation ±1m at `0.5 m/s` |

> The platform moves left and right in a sinusoidal pattern. You must **predict** its position at the moment of touchdown, not just react to it.

---

## 3. Sensor Suite — What Your Drone Sees

Unlike Level 1 (LiDAR + fog of war), Level 2 gives you a **simulated downward-facing camera**. There is no direct position readout — you must infer everything from pixels.

### Camera Feed

Every frame, the simulator writes a **100×100 grayscale pixel grid** to `camera_pixels.txt`:

```
// camera_pixels.txt format:
// 100 rows × 100 columns of integer grayscale values (0–255)
// Separated by spaces, one row per line

45 45 45 45 ... (100 values)
45 45 255 255... 
...
```

### Understanding the Camera

- The camera looks **straight down** from the drone
- **Field of View scales with altitude**: at high altitude, you see a large ground area; at low altitude, you see a small area in detail
  - Ground coverage: `2.5 × altitude` meters (e.g., at 5m altitude → 12.5m × 12.5m visible)
- The platform appears as a **bright white square** (grayscale ≈ 250) on a **dark green background** (≈ 45)
- The platform has a **black inner marker** in its center (H-landing-pad style) to help with precise centering

### What You Must Compute

From the pixel data, you must determine:
1. **Is the platform visible?** (is there a large white region in the frame?)
2. **Where is the platform center relative to the drone** (in pixels → convert to meters)
3. **How fast is it moving?** (track across frames)

---

## 4. Your Task

Your controller reads `camera_pixels.txt` and writes velocity commands to `commands.txt`:

```
// commands.txt format:
vx vy
// Example: "-0.5 0.3"
// Units: m/s
```

The simulator reads `commands.txt` every frame and applies lateral velocity to the drone accordingly.

### In Python (if using Python controller)

```python
# Read camera
with open("camera_pixels.txt", "r") as f:
    rows = [[int(v) for v in line.split()] for line in f]

# Compute vx, vy (your logic here)
vx, vy = your_controller(rows)

# Write commands
with open("commands.txt", "w") as f:
    f.write(f"{vx:.4f} {vy:.4f}")
```

### In C (optional, for extra challenge)

```c
// Read camera_pixels.txt into a 100x100 int array
// Compute vx, vy
// Write to commands.txt
// Run: python simulator.py --c
```

---

## 5. Running the Simulation

```bash
cd Hackathon/Level2

# Demo mode (uses built-in proportional controller — your baseline reference)
python simulator.py

# Your controller mode (reads from commands.txt)
python simulator.py --c
```

> **In demo mode**, the simulator uses a simple proportional controller that already has direct access to platform position. Your job is to replicate this performance using **only camera pixels** — no direct coordinates.

---

## 6. Scoring

| Outcome | Score |
|---------|-------|
| Landed within `0.35 m` of center | **SUCCESS** |
| Landed outside `0.35 m` | **FAILED** |
| Final error `< 0.1 m` | **Precision Bonus** |

The simulation prints your final distance from the platform center at touchdown.

---

## 7. Key Challenges

> **Perspective Distortion.** At high altitude, the entire platform appears small. At low altitude, only part of the platform may be visible. Your pixel-to-meter conversion must account for current altitude.

> **Platform Prediction.** The platform moves at `0.5 m/s`. Over 20 seconds of descent, it oscillates multiple times. Steer toward where it *will be*, not where it is now.

> **No Direct Position Data.** Unlike Level 1, there is no `player_x / player_y`. You must estimate position purely from the camera image.

---

## 8. Tips

- Start by detecting the centroid of the white blob in the camera image using basic image processing (threshold → find center of mass)
- Track centroid across frames to estimate platform velocity
- Convert pixel offset to real-world meters: `error_m = (pixel_offset / 100) * ground_coverage_m`
- A proportional controller is sufficient for a passing grade; a predictive controller earns bonus points

---


