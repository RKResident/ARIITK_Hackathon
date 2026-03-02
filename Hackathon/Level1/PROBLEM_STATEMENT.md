# Level 1 — Operation SKYE-X: Search and Pursuit

> **Team Aerial Robotics | Y25 Recruitment Hackathon**

---

## 1. Context

You are the operator of an autonomous quadcopter drone in a classified industrial facility — **Operation SKYE-X**. The arena is a bounded 2D warehouse space with a tile-grid floor, scattered with rectangular physical obstacles simulating crates, pillars, and wide barriers.

Your drone — the **Player** — starts at the top-left corner of the facility `(80, 80)`. Somewhere in the vastness of the map, an **adversarial Target drone** has been deployed at the opposite corner. Your mission: find it, and never let it go.

---

## 2. The Environment

### Arena
| Parameter | Value |
|-----------|-------|
| Width × Height | `1280 × 900` pixels |
| Obstacles | `22` randomized rectangles (crates, pillars, barriers) |
| Player Spawn | `(80, 80)` |
| Target Spawn | `(WIDTH − 150, HEIGHT − 150)` ≈ `(1130, 750)` |
| Mission Duration | `3000` timesteps |
| Frame Rate | `60 FPS` |

### Drone Physics
| Parameter | Value |
|-----------|-------|
| Max Player Speed | `3.5` px/timestep |
| Player Collision Radius | `10` px |
| Max Target Speed (Normal) | `3.5` px/timestep |

> **Note:** Speed is enforced — if your output velocity vector exceeds `MAX_PLAYER_SPEED`, it is automatically clamped to that magnitude.

---

## 3. Sensor Suite — What You Can See

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

## 4. The Adversarial Target

The target drone is not passive. It has its own onboard AI with three behavioural layers:

| Behaviour | Trigger | Description |
|-----------|---------|-------------|
| **Gaussian Random Walk** | Always | Base momentum: smoothly random trajectory |
| **Seek Cover (LoS Breaking)** | Player within `80 px` | Identifies the best obstacle to hide behind and steers toward it |
| **Direct Escape** | Player within `80 px`, no cover nearby | Flees directly away from the player |
| **Wall Avoidance** | Near arena edges | Soft repulsion from boundaries |

> The target's **cover-seeking** is its most dangerous behaviour: it actively places obstacles between itself and you, breaking your line-of-sight and causing `target_visible` to drop to `False`.

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
- `timesteps >= 3000` (normal completion)
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

## 8. Constraints

| Rule | Detail |
|------|--------|
| Modify | `skye_controller.py` only |
| Do not modify | `skye_env.py`, `config.py` |
| Do not | Hard-code target coordinates or exploit rendering internals |
| Allowed | Any Python library (`numpy`, `scipy`, etc.) |

---

## 9. Tips & Hints

> **Exploration is the bottleneck.** The target spawns ~1500 px away. A naive random walk may never find it within 3000 steps.

> **The LiDAR is your map.** Rays pointing toward open space (high reading) indicate navigable directions. Use them to do wall-following or potential field navigation.

> **Predictive pursuit beats reactive pursuit.** The target moves up to 3.5 px/step. If you know its position, aim slightly ahead of it based on its movement direction.

> **Cover Awareness.** After losing sight of the target, it likely tried to hide behind the nearest obstacle relative to you. Check nearby obstacle positions.

---

## 10. Scoring Summary

| Outcome | Score |
|---------|-------|
| Crashed immediately | `0` |
| Found target but failed to track | Low (< 50) |
| Found & tracked for most of mission | Medium (200–800) |
| Perfect pursuit + survival | High (> 1000) |

