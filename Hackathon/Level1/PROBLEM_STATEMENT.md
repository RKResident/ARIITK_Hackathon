![Level 1 Demo](Level1.gif)

# Level 1 — Operation SKYE-X: Search and Pursuit

> **Team Aerial Robotics IITK| Y25 Recruitment Hackathon**

---

## 1. Context

A rogue adversarial drone — designation **SKYE-X** — has been detected inside a classified industrial facility. Intel suggests it is equipped with onboard evasion algorithms and is actively trying to break line-of-sight, hide behind obstacles, and escape your tracking radius.

You are the operator of an autonomous interceptor quadcopter. You have no direct control — your drone acts on the instructions of the AI controller you write. Your sensors are limited. The map is mostly hidden behind Fog of War. The target knows you are coming.

Your mission is to navigate the facility, locate SKYE-X, and shadow it persistently within a **70-pixel tracking radius** for as long as possible before the mission clock runs out.

**SKYE-X will not make this easy.**

---

## 2. The Enemy — SKYE-X

SKYE-X is not a passive target. It has its own onboard AI with three distinct behavioural layers that activate depending on how close you are:

| Behaviour | Trigger | What SKYE-X Does |
|-----------|---------|-------------------|
| **Gaussian Random Walk** | Always active | Moves with smooth, unpredictable momentum — never fully still, never fully predictable |
| **Seek Cover** | You within `80 px` | Identifies the nearest obstacle and steers toward it to break your line-of-sight. **This is its most dangerous capability.** |
| **Direct Escape** | You within `80 px`, no cover nearby | Flees directly away from you at full speed |
| **Wall Avoidance** | Near arena edges | Soft repulsion keeps it from cornering itself — it will not trap itself for you |

> The cover-seeking behaviour is what separates good controllers from great ones. SKYE-X actively places rectangular obstacles between itself and you, causing `target_visible` to drop to `False` and forcing you back into blind exploration. If your pursuit logic cannot recover from a lost target, your score will stall.

---

## 3. The Environment

### Arena
| Parameter | Value |
|-----------|-------|
| Width × Height | `1280 × 900` pixels |
| Obstacles | `22` randomized rectangles (crates, pillars, barriers) |
| Player Spawn | `(80, 80)` |
| Target Spawn | `(WIDTH − 150, HEIGHT − 150)` ≈ `(1130, 750)` |
| Mission Duration | `3000` timesteps (default — configurable via `config.py`) |
| Frame Rate | `60 FPS` |

### Drone Physics
| Parameter | Value |
|-----------|-------|
| Max Player Speed | `3.5` px/timestep |
| Player Collision Radius | `10` px |
| Max Target Speed (Normal) | `3.5` px/timestep |

> **Note:** Speed is enforced — if your output velocity vector exceeds `MAX_PLAYER_SPEED`, it is automatically clamped to that magnitude.

---

## 4. Sensor Suite — What You Can See

Your only window into the world is the `get_sensor_data()` return dictionary:

```python
sensors = {
    "player_x":        float,   # Your X coordinate
    "player_y":        float,   # Your Y coordinate
    "lidar_distances": list,    # 36 floats — one per ray (every 10°)
    "target_visible":  bool,    # True if target is within VISIBILITY_RADIUS (150px)
    "target_pos":      tuple    # (tx, ty) if visible, else None
}
```

### LiDAR System
- **36 rays**, equally spaced every **10°** (full 360° sweep)
- Each reading is the distance to the nearest obstacle/wall in that direction
- Maximum range: **150 px** (Visibility Radius)
- Beyond that range: the map is hidden behind **Fog of War**

### Visibility
- You can only detect the target if it is within **150 px** of your position
- `target_visible` becomes `True` the moment the target enters this radius
- Once discovered (`target_discovered = True`), scoring begins

---

## 5. Collision Rules

Your drone must avoid:
1. **Arena boundaries** — flying outside `[0, WIDTH] × [0, HEIGHT]` instantly crashes the drone.
2. **Rectangular obstacles** — collision is detected using circle-AABB: if your drone's center comes within **10 px** (DRONE_RADIUS) of any obstacle edge, it crashes.

**A crash immediately ends the mission with your current score — no partial credit.**

---

## 6. Objectives & Scoring

### Score Accumulation
- **+1 point per timestep** that the following conditions are **both** met:
  1. The target has been discovered at least once (`target_discovered == True`)
  2. Your drone is within **70 px** (TRACKING_RADIUS) of the target's current position

### Mission Termination
The mission ends when **any** of these occur:
- `timesteps >= MAX_TIMESTEPS` (default `3000`, configurable in `config.py`)
- Player drone crashes into a wall or obstacle

### What Determines a Good Score?
- **Fast exploration**: Find the target quickly to start accumulating points early
- **Persistent pursuit**: Stay within 70 px despite the target actively fleeing
- **Obstacle awareness**: Cut through the maze efficiently without crashing

---

## 7. Your Task

Open `skye_controller.py` and implement the `compute_velocity(sensors)` function.

```python
def compute_velocity(sensors) -> (float, float):
    """
    Input : sensors dict (see Section 3)
    Output: (vx, vy) — your velocity command for this timestep
            ||(vx, vy)|| will be clamped to MAX_PLAYER_SPEED = 3.5
    """
    ...
```

A starter implementation with a **dual-state template** is already provided:
- **State 1 (Exploration):** When `target_visible == False` — explore the map
- **State 2 (Pursuit):** When `target_visible == True` — chase and track the target

You are free to use any approach: hand-tuned heuristics, potential fields, A* path planning, Reinforcement Learning, or anything else.

---

## 8. Deliverables

Teams must submit a single GitHub repository containing all of the following:

| # | File | Description |
|---|------|-------------|
| 1 | `skye_controller.py` | Your implementation of `compute_velocity(sensors)`. This is the only file you modify. Must run without errors on the unmodified `skye_env.py` and `config.py` provided. |
| 2 | `score_proof.png` | A screenshot of your final simulation run showing the score on screen. Taken at mission end (timestep 3000 or crash). |

---

## 9. Constraints

| Rule | Detail |
|------|--------|
| Modify | `skye_controller.py` (your main solution) |
| May modify | `config.py` — you may adjust parameters like `MAX_TIMESTEPS` for tuning/testing |
| Do not modify | `skye_env.py` |
| Do not | Hard-code target coordinates or exploit rendering internals |
| Allowed | Any Python library (`numpy`, `scipy`, etc.) |

---

## 10. Scoring Summary

| Outcome | Score |
|---------|-------|
| Crashed immediately | `0` |
| Found target but failed to track | Low (< 50) |
| Found & tracked for most of mission | Medium (200–800) |
| Perfect pursuit + survival | High (> 1000) |
