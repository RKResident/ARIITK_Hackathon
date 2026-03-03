# Aerial Robotics IITK Hackathon 2026

> **Team Aerial Robotics IITK | Y25 Recruitment Hackathon 2026**

Hey Y25s! Hope you are enjoying your time on the campus. These are the problem statements of the **Aerial Robotics Hackathon 2026**. They are divided into three levels and a **Bonus level**. The submission guidelines for each level are mentioned clearly — please follow them correctly. Any doubts can be raised on GitHub in the **Discussions** tab of this repository. You can find the problem statements and required starter kits in this repo.

---

## Hackathon Structure

| Level | Title | Description |
|-------|-------|-------------|
| [Level 1](./Hackathon/Level1/PROBLEM_STATEMENT.md) | **Operation SKYE-X: Search & Pursuit** | Explore the map, find an evasive target, and track it under sensor limitations |
| [Level 2](./Hackathon/Level2/PROBLEM_STATEMENT.md) | **Operation Touchdown: Precision Landing** | A harder variant with new constraints — unlocked after Level 1 |
| [Level 3](./Hackathon/Level3/PROBLEM_STATEMENT.md) | **Project MIRAGE: Hybrid Marker Steganography** | Create a single image that shows a decoy at high altitude and a secret ArUco marker at low altitude |
| [Bonus](./Hackathon/Bonus/PROBLEM_STATEMENT.md) | **Operation FREEFALL: Autonomous Precision Landing** | Use a 3D physics simulator, ArUco detection, and PID control to land a drone on static and moving targets |

---

## Environment Setup

### Quick Install (All Levels)
```bash
pip install -r requirements.txt
```

### Per-Level Dependencies
| Level | Packages |
|-------|----------|
| Level 1 & 2 | `pygame` |
| Level 3 | `opencv-contrib-python`, `numpy` |
| Bonus | `opencv-contrib-python`, `numpy`, `requests`, `simple-pid`, `pyautogui` |

---

## Submission Guidelines

### Level 1
- Only modify `skye_controller.py` — **do not modify `skye_env.py` or `config.py`**
- Submit your final `skye_controller.py` with a short write-up explaining your approach

### Level 2
- Write a controller (Python or C) that reads `camera_pixels.txt` and writes velocity commands to `commands.txt`
- Submit your controller source file with a short write-up explaining your approach

### Level 3
- Submit a GitHub repository with: `generator.py`, `hybrid_marker.png`, `viewer.py`, `visual_check.py`, `drone_view_2.0m.png`, `drone_view_30.0m.png`, and `report.pdf`
- `viewer.py` and `visual_check.py` must be **built from scratch** — they are not provided
- `report.pdf` (max 2 pages) must include σ values, mathematical reasoning, and GSD calculation

### Bonus Level
- Submit `auto_land.py`, `landing_video.mp4`, and `report.pdf`
- `auto_land.py` must connect to the simulator (ports 8080/8081) and use ArUco detection — **no hardcoded coordinates**
- `report.pdf` (max 2 pages) must include PID tuning approach, control loop diagram, and ArUco pipeline explanation

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

### Level 3 — Project MIRAGE

| Criterion | Weightage | How It's Measured |
|-----------|-----------|-------------------|
| **ArUco ID_X detected at ≤ 5m** | 40% | Image shrunk to 5m pixel size, ArUco detection run |
| **ArUco ID_X hidden at ≥ 30m** | 30% | Image shrunk to 30m pixel size, ArUco detection run |
| **Visual quality — no ghosting** | 15% | Judging script at 30m; visual inspection for ArUco bleed-through |
| **Math justification in report** | 15% | σ values backed by GSD/Nyquist calculations, not trial and error |

### Bonus — Operation FREEFALL

| Metric | Points |
|--------|--------|
| **Phase 1 Completion** (static landing) | +300 |
| **Phase 2 Completion** (moving platform) | +500 |
| **Landing Smoothness** | Up to +200 bonus (lower touchdown velocity → higher score) |
| **Time to Land** | Fastest landing breaks ties |
