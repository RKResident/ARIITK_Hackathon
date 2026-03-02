# Operation SKYE-X — Autonomous Aerial Robotics Hackathon

> **Aerial Robotics Club | Y25 Recruitment Hackathon**

Welcome to **Operation SKYE-X**, a multi-level autonomous drone simulation challenge. Your mission is to program an intelligent flight controller for a quadcopter drone navigating a contested 2D aerial environment.

---

## Challenge Structure

| Level | Title | Description |
|-------|-------|-------------|
| [Level 1](./Hackathon/Level1/PROBLEM_STATEMENT.md) | **Operation SKYE-X: Search & Pursuit** | Explore the map, find an evasive target, and track it under sensor limitations |
| [Level 2](./Hackathon/Level2/PROBLEM_STATEMENT.md) | **Operation Touchdown: Precision Landing** | A harder variant with new constraints — unlocked after Level 1 |

---

## Environment Setup

### Requirements
```bash
pip install pygame
```

### Running Simulation
```bash
# Level 1 — Search & Pursuit
cd Hackathon/Level1
python skye_controller.py

# Level 2 — Precision Landing
cd Hackathon/Level2
python simulator.py          # Demo mode (built-in controller)
python simulator.py --c      # Your controller (reads commands.txt)
```

---

## Submission Guidelines

### Level 1
- Only modify `skye_controller.py` — **do not modify `skye_env.py` or `config.py`**
- Submit your final `skye_controller.py` with a short write-up explaining your approach

### Level 2
- Write a controller (Python or C++) that reads `camera_pixels.txt` and writes velocity commands to `commands.txt`
- Submit your controller source file with a short write-up explaining your approach

---

## Scoring

### Level 1 — Search & Pursuit

| Metric | How It's Measured |
|--------|-------------------|
| **Tracking Score** | +1 per timestep the target is within `TRACKING_RADIUS` (70px) |
| **Survival** | Score is forfeited on crash |
| **Mission Duration** | Max `3000` timesteps |

### Level 2 — Precision Landing

| Metric | How It's Measured |
|--------|-------------------|
| **Landing Accuracy** | Distance (m) from platform center at touchdown |
| **Success Threshold** | ≤ `0.35 m` from center = **SUCCESS** |
| **Precision Bonus** | `< 0.1 m` = exceptional |
