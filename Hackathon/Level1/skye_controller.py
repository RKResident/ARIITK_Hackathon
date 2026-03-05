"""
skye_controller.py  –  LiDAR Ray-Cast Mapping + Frontier Exploration + Wall-Sliding
=====================================================================================
MAPPING:
  - Every frame, 36 LiDAR rays are ray-marched through a 20px occupancy grid.
    Cells along each ray → FREE.  Terminal cell (ray hit) → OCCUPIED.
    Everything else stays UNKNOWN.  States: 0=UNKNOWN, 1=FREE, 2=OCCUPIED.

EXPLORATION:
  - Frontier cells: FREE cells 8-adjacent to at least one UNKNOWN cell.
  - Frontiers are greedy-clustered; clusters below MIN_FRONTIER_SIZE cells
    are discarded so the drone ignores tiny slivers at obstacle edges.
  - A* on the occupancy grid plans a collision-free path to the chosen
    frontier centroid. The drone follows the path cell-by-cell, replanning
    when the goal changes or the path is exhausted.
  - Gaussian random walk fallback when no frontier goal is active.
  - Wall-sliding collision response (normal component zeroed, tangential kept).

VISUALIZATION:
  - Matplotlib runs in a BACKGROUND THREAD using the non-interactive Agg
    backend (no Tk/Qt event loop → no GIL conflict with pygame).
  - Main thread pushes snapshots via a Queue every 60 frames.
  - Each rendered frame is held in memory as a PIL Image.
  - On mission end the thread stitches all frames into  map_coverage.gif.

PURSUIT: predictive intercept with per-frame velocity estimation.
"""

import math
import random
import queue
import heapq
from collections import deque
import threading
import pygame
import matplotlib
matplotlib.use("Agg")          # ← non-interactive, no Tk, safe from any thread
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from skye_env_corrected import SkyeEnv
from config import (
    WIDTH, HEIGHT, FPS,
    MAX_PLAYER_SPEED, MAX_TARGET_SPEED,
    VISIBILITY_RADIUS, TRACKING_RADIUS,
    MAX_TIMESTEPS, SPAWN_X, SPAWN_Y,
    NUM_RAYS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Grid
# ─────────────────────────────────────────────────────────────────────────────
_CELL     = 20
_COLS     = WIDTH  // _CELL    # 64
_ROWS     = HEIGHT // _CELL    # 45
_UNKNOWN  = 0
_FREE     = 1
_OCCUPIED = 2

# ─────────────────────────────────────────────────────────────────────────────
# Motion constants
# ─────────────────────────────────────────────────────────────────────────────
_WALL_MARGIN       = 25    # drone can fly close to arena edges (DRONE_RADIUS=10)
_OBS_HARD_D        = 10   # px beyond obstacle face: always deflect (just above DRONE_RADIUS)
_OBS_SOFT_D        = 26    # px beyond obstacle face: partial deflect (proportional)
_WANDER_SIGMA      = 0.18
_FRONTIER_CLUSTER  = 80    # px — merge radius for frontier clustering
_FRONTIER_REACH    = 25    # px — goal considered reached within this distance
_MIN_FRONTIER_SIZE = 6     # minimum cells in a cluster to be navigation-worthy
_MIN_CLEARANCE     = 2     # cells of clearance required to enter a cell in A*
                           # (2 cells * 20px = 40px, > DRONE_RADIUS=10 with margin)
_CLEARANCE_PENALTY = 6.0   # extra cost per cell for low-clearance cells in A*

# Levy flight parameters
_LEVY_BETA         = 1.5   # power-law exponent in (1, 2); lower = heavier tail
_LEVY_SCALE        = 40.0  # base px scale -- median short step length
_LEVY_MAX_STEP     = max(WIDTH, HEIGHT) * 0.85  # cap ballistic jumps at 85% arena

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(vx, vy, speed=1.0):
    mag = math.hypot(vx, vy)
    if mag < 1e-6:
        return 0.0, 0.0
    return (vx / mag) * speed, (vy / mag) * speed



def _levy_step(beta=_LEVY_BETA, scale=_LEVY_SCALE):
    """Sample a step length from a Levy distribution via the Mantegna algorithm."""
    import math as _m
    sigma_num = _m.gamma(1 + beta) * _m.sin(_m.pi * beta / 2)
    sigma_den = _m.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2))
    sigma = (sigma_num / sigma_den) ** (1 / beta)
    u = random.gauss(0, sigma)
    v = abs(random.gauss(0, 1)) or 1e-9
    raw = abs(u / (v ** (1 / beta)))
    return min(raw * scale, _LEVY_MAX_STEP)


def _levy_waypoint(px, py, visit_grid):
    """
    Sample a Levy step length, then pick the direction (among 16 candidates)
    pointing toward the least-visited cell at approximately that distance.
    Returns a (wx, wy) world-coordinate waypoint, clamped inside the arena.
    """
    step = _levy_step()
    best_angle = random.uniform(0, 2 * math.pi)   # fallback: random direction
    best_score = float("inf")
    for k in range(16):
        a  = (k / 16) * 2 * math.pi
        tx = px + math.cos(a) * step
        ty = py + math.sin(a) * step
        tx = max(_WALL_MARGIN, min(WIDTH  - _WALL_MARGIN, tx))
        ty = max(_WALL_MARGIN, min(HEIGHT - _WALL_MARGIN, ty))
        gc, gr = _world_to_grid(tx, ty)
        # Skip destinations that land in low-clearance cells
        clr_map = _state.get("clearance_map")
        if clr_map is not None and clr_map[gr][gc] < _MIN_CLEARANCE:
            continue
        score  = visit_grid[gr][gc]
        if score < best_score:
            best_score = score
            best_angle = a
    wx = px + math.cos(best_angle) * step
    wy = py + math.sin(best_angle) * step
    wx = max(_WALL_MARGIN, min(WIDTH  - _WALL_MARGIN, wx))
    wy = max(_WALL_MARGIN, min(HEIGHT - _WALL_MARGIN, wy))
    return wx, wy

def _world_to_grid(x, y):
    return (
        int(max(0, min(_COLS - 1, x / _CELL))),
        int(max(0, min(_ROWS - 1, y / _CELL))),
    )


def _grid_to_world(col, row):
    return (col + 0.5) * _CELL, (row + 0.5) * _CELL


# ─────────────────────────────────────────────────────────────────────────────
# LiDAR ray-cast mapping
# ─────────────────────────────────────────────────────────────────────────────

def _update_map(grid, px, py, lidar):
    for i, dist in enumerate(lidar):
        angle = (i / NUM_RAYS) * 2 * math.pi
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        hit   = dist < VISIBILITY_RADIUS - 1.0

        step  = _CELL * 0.5
        steps = int(dist / step)
        prev_c, prev_r = -1, -1

        for s in range(steps + 1):
            t  = s * step
            c, r = _world_to_grid(px + cos_a * t, py + sin_a * t)
            if c == prev_c and r == prev_r:
                continue
            prev_c, prev_r = c, r
            if grid[r][c] != _OCCUPIED:
                grid[r][c] = _FREE

        if hit:
            ec, er = _world_to_grid(px + cos_a * dist, py + sin_a * dist)
            grid[er][ec] = _OCCUPIED


# ─────────────────────────────────────────────────────────────────────────────
# Frontier extraction + clustering
# ─────────────────────────────────────────────────────────────────────────────
_NBRS_8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def _get_frontier_cells(grid):
    frontiers = []
    for r in range(_ROWS):
        for c in range(_COLS):
            if grid[r][c] != _FREE:
                continue
            for dr, dc in _NBRS_8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < _ROWS and 0 <= nc < _COLS and grid[nr][nc] == _UNKNOWN:
                    frontiers.append(_grid_to_world(c, r))
                    break
    return frontiers


def _cluster_frontiers(pts, radius=_FRONTIER_CLUSTER):
    """
    Greedy clustering of frontier points.
    Returns list of (centroid_world_xy, cluster_size) — only clusters with
    at least _MIN_FRONTIER_SIZE raw cells are included.
    """
    if not pts:
        return []
    assigned = [False] * len(pts)
    results   = []
    for i, (fx, fy) in enumerate(pts):
        if assigned[i]:
            continue
        cluster = [(fx, fy)]
        assigned[i] = True
        for j in range(i + 1, len(pts)):
            if not assigned[j] and math.hypot(pts[j][0]-fx, pts[j][1]-fy) < radius:
                cluster.append(pts[j])
                assigned[j] = True
        if len(cluster) < _MIN_FRONTIER_SIZE:
            continue                          # ← size filter
        centroid = (
            sum(p[0] for p in cluster) / len(cluster),
            sum(p[1] for p in cluster) / len(cluster),
        )
        results.append((centroid, len(cluster)))
    return results   # [(world_xy, size), ...]


def _nearest_frontier(px, py, clusters):
    """
    Pick the nearest qualifying cluster centroid.
    `clusters` is the list returned by _cluster_frontiers.
    """
    best, best_d = None, float("inf")
    for (cx, cy), _size in clusters:
        d = math.hypot(cx - px, cy - py)
        if d < best_d:
            best_d, best = d, (cx, cy)
    return best   # world (x, y) or None


def _nearest_free_cell(grid, gc, gr):
    """
    If (gc, gr) is FREE, return it unchanged.
    Otherwise BFS outward to find the nearest FREE cell.
    Returns (col, row) in grid coordinates.
    """
    if grid[gr][gc] == _FREE:
        return gc, gr
    visited = [[False] * _COLS for _ in range(_ROWS)]
    q = deque([(gc, gr)])
    visited[gr][gc] = True
    while q:
        c, r = q.popleft()
        if grid[r][c] == _FREE:
            return c, r
        for dr, dc in _NBRS_8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < _ROWS and 0 <= nc < _COLS and not visited[nr][nc]:
                visited[nr][nc] = True
                q.append((nc, nr))
    return gc, gr   # fallback


# ─────────────────────────────────────────────────────────────────────────────
# Brushfire clearance map
# ─────────────────────────────────────────────────────────────────────────────

def _build_clearance_map(grid):
    """
    BFS brushfire from every OCCUPIED/UNKNOWN cell outward.
    Returns a 2-D array (same shape as grid) where each FREE cell's value
    is its distance in cells to the nearest non-FREE cell.
    OCCUPIED/UNKNOWN cells get clearance 0.
    """
    clearance = [[0] * _COLS for _ in range(_ROWS)]
    q = deque()

    # Seed: every non-FREE cell has clearance 0 and is the wavefront
    for r in range(_ROWS):
        for c in range(_COLS):
            if grid[r][c] != _FREE:
                clearance[r][c] = 0
                q.append((r, c))
            else:
                clearance[r][c] = 999  # large sentinel

    # BFS outward — each FREE cell gets distance from nearest obstacle
    while q:
        r, c = q.popleft()
        for dr, dc in _NBRS_8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < _ROWS and 0 <= nc < _COLS):
                continue
            if grid[nr][nc] != _FREE:
                continue
            new_dist = clearance[r][c] + 1
            if new_dist < clearance[nr][nc]:
                clearance[nr][nc] = new_dist
                q.append((nr, nc))

    return clearance


# ─────────────────────────────────────────────────────────────────────────────
# A* path planner on the occupancy grid
# ─────────────────────────────────────────────────────────────────────────────
# Diagonal move cost
_SQRT2 = math.sqrt(2)

def _astar(grid, start_world, goal_world, clearance=None):
    """
    A* from start_world to goal_world on the occupancy grid.
    Treats OCCUPIED and UNKNOWN cells as impassable.
    If clearance map is provided, cells with clearance < _MIN_CLEARANCE
    are penalised heavily so A* prefers wide corridors over narrow gaps.
    Returns a list of world-coordinate waypoints (excluding the start cell),
    or [] if no path exists.
    """
    sc, sr = _world_to_grid(*start_world)
    gc, gr = _world_to_grid(*goal_world)

    if (sc, sr) == (gc, gr):
        return []

    # If goal cell is not FREE, find the nearest FREE cell to it
    if grid[gr][gc] != _FREE:
        best, best_d = None, float("inf")
        for r in range(_ROWS):
            for c in range(_COLS):
                if grid[r][c] == _FREE:
                    d = math.hypot(c - gc, r - gr)
                    if d < best_d:
                        best_d, best = d, (c, r)
        if best is None:
            return []
        gc, gr = best

    # g_cost[r][c] = best known cost to reach (c, r)
    g_cost = [[float("inf")] * _COLS for _ in range(_ROWS)]
    g_cost[sr][sc] = 0.0

    # parent[r][c] = (pc, pr) for path reconstruction
    parent = [[None] * _COLS for _ in range(_ROWS)]

    def h(c, r):
        return math.hypot(c - gc, r - gr)

    # Priority queue: (f, g, col, row)
    open_heap = [(h(sc, sr), 0.0, sc, sr)]

    while open_heap:
        f, g, c, r = heapq.heappop(open_heap)

        if (c, r) == (gc, gr):
            # Reconstruct path
            path = []
            node = (gc, gr)
            while node != (sc, sr):
                path.append(_grid_to_world(*node))
                node = parent[node[1]][node[0]]
            path.reverse()
            return path

        if g > g_cost[r][c]:
            continue   # stale entry

        for dr, dc in _NBRS_8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < _ROWS and 0 <= nc < _COLS):
                continue
            if grid[nr][nc] != _FREE:
                continue
            move_cost = _SQRT2 if (dr != 0 and dc != 0) else 1.0
            # Clearance penalty: heavily penalise cells too close to walls
            if clearance is not None:
                cell_clr = clearance[nr][nc]
                if cell_clr < _MIN_CLEARANCE:
                    move_cost += _CLEARANCE_PENALTY * (_MIN_CLEARANCE - cell_clr)
            ng = g + move_cost
            if ng < g_cost[nr][nc]:
                g_cost[nr][nc] = ng
                parent[nr][nc] = (c, r)
                heapq.heappush(open_heap, (ng + h(nc, nr), ng, nc, nr))

    return []   # no path found


# ─────────────────────────────────────────────────────────────────────────────
# Wall-sliding collision response
# ─────────────────────────────────────────────────────────────────────────────

def _slide_velocity(px, py, vx, vy, obstacles):
    """
    Two-zone wall sliding:
      Hard zone (within OBS_HARD_D of face): fully zero the into-surface
        velocity component — drone cannot advance closer.
      Soft zone (OBS_HARD_D..OBS_SOFT_D): scale the into-surface component
        down proportionally — drone slows its approach but isn't stopped,
        allowing it to squeeze into narrow gaps while still being deflected
        away from direct collisions.
    Arena walls use the same two-zone logic.
    """
    # ── Arena walls ───────────────────────────────────────────────────────────
    for wall_dist, axis, into_neg in [
        (px,              "x", True),   # left wall
        (WIDTH  - px,     "x", False),  # right wall
        (py,              "y", True),   # top wall
        (HEIGHT - py,     "y", False),  # bottom wall
    ]:
        if wall_dist < _WALL_MARGIN:
            t = max(0.0, wall_dist / _WALL_MARGIN)   # 0=at wall, 1=at margin
            if axis == "x":
                component = -vx if into_neg else vx
                if component > 0:
                    if into_neg: vx = vx * t
                    else:        vx = vx * t
            else:
                component = -vy if into_neg else vy
                if component > 0:
                    if into_neg: vy = vy * t
                    else:        vy = vy * t

    # ── Obstacle faces ────────────────────────────────────────────────────────
    for cx, cy, hw, hh in obstacles:
        dx = px - cx
        dy = py - cy

        # Skip if outside the soft zone entirely
        if abs(dx) >= hw + _OBS_SOFT_D or abs(dy) >= hh + _OBS_SOFT_D:
            continue

        # Identify nearest face normal (minimum-overlap axis)
        overlap_x = (hw + _OBS_SOFT_D) - abs(dx)
        overlap_y = (hh + _OBS_SOFT_D) - abs(dy)
        if overlap_x < overlap_y:
            nx, ny   = math.copysign(1.0, dx), 0.0
            face_gap = abs(dx) - hw      # actual distance from obstacle face
        else:
            nx, ny   = 0.0, math.copysign(1.0, dy)
            face_gap = abs(dy) - hh

        # How much of the into-surface velocity to cancel
        into = vx * (-nx) + vy * (-ny)   # positive = moving into surface
        if into <= 0:
            continue   # already moving away — don't interfere

        if face_gap <= _OBS_HARD_D:
            # Hard zone: cancel fully
            cancel = 1.0
        else:
            # Soft zone: linear ramp from 0 (at soft boundary) to 1 (at hard)
            cancel = 1.0 - (face_gap - _OBS_HARD_D) / (_OBS_SOFT_D - _OBS_HARD_D)

        vx -= cancel * into * (-nx)
        vy -= cancel * into * (-ny)

    return vx, vy


# ─────────────────────────────────────────────────────────────────────────────
# Background plot thread  (Agg — no GUI event loop, no GIL clash)
# ─────────────────────────────────────────────────────────────────────────────

import io
from PIL import Image

_plot_queue   = queue.Queue(maxsize=2)   # holds latest snapshot dicts
_SENTINEL     = object()                 # signals thread to exit
_GIF_PATH     = "map_coverage.gif"

def _plot_thread_fn(obstacles):
    """
    Runs in a daemon thread.  Pulls snapshots from _plot_queue, renders with
    Agg, and saves to map_live.png.  Exits when it receives _SENTINEL.
    """
    # ── Build the figure once ─────────────────────────────────────────────────
    cmap  = matplotlib.colors.ListedColormap(["#1a1a2e", "#2d6a2d", "#8b4513"])
    norm  = matplotlib.colors.BoundaryNorm([0, 1, 2, 3], cmap.N)

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor("#111120")
    ax.set_facecolor("#1a1a2e")

    data = np.zeros((_ROWS, _COLS), dtype=np.uint8)
    im   = ax.imshow(data, origin="upper",
                     extent=[0, WIDTH, HEIGHT, 0],
                     cmap=cmap, norm=norm,
                     interpolation="nearest", zorder=1)

    for cx, cy, hw, hh in obstacles:
        ax.add_patch(plt.Rectangle(
            (cx - hw, cy - hh), hw * 2, hh * 2,
            lw=1, edgecolor="#c87840", facecolor="none",
            linestyle="--", alpha=0.6, zorder=2,
        ))

    frontier_sc = ax.scatter([], [], c="#00ffff", s=10, zorder=4,
                             label="Frontier", alpha=0.8)
    drone_dot,  = ax.plot([], [], "o", color="#40a0ff", ms=10,
                          label="Player", zorder=6)
    target_dot, = ax.plot([], [], "o", color="#ff4040", ms=10,
                          label="Target",  zorder=6)
    goal_dot,   = ax.plot([], [], "*", color="#ffe040", ms=14,
                          label="Goal",    zorder=6)
    path_line,    = ax.plot([], [], "-", color="#a0ff80", lw=1.5,
                            alpha=0.7, zorder=5, label="A* explore")
    pursuit_line, = ax.plot([], [], "-", color="#ff80ff", lw=1.5,
                            alpha=0.8, zorder=5, label="A* pursue")
    arrow_line,   = ax.plot([], [], "-", color="#ffe080", lw=2, zorder=7)

    legend_handles = [
        mpatches.Patch(color="#1a1a2e", label="Unknown"),
        mpatches.Patch(color="#2d6a2d", label="Free"),
        mpatches.Patch(color="#8b4513", label="Occupied"),
        frontier_sc, drone_dot, target_dot, goal_dot, path_line, pursuit_line,
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              facecolor="#22223a", labelcolor="white",
              fontsize=8, framealpha=0.85)
    ax.set_xlim(0, WIDTH);  ax.set_ylim(HEIGHT, 0)
    ax.set_title("SKYE-X  ·  LiDAR Occupancy Map", color="white", fontsize=13)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#404060")
    fig.tight_layout()

    gif_frames = []   # list of PIL Images accumulated across the mission

    # ── Render loop ───────────────────────────────────────────────────────────
    while True:
        snap = _plot_queue.get()          # blocks until data arrives
        if snap is _SENTINEL:
            break

        # Unpack snapshot
        grid         = snap["grid"]
        px, py       = snap["px"], snap["py"]
        wander_angle = snap["wander_angle"]
        f_pts        = snap["frontier_pts"]
        goal         = snap["goal"]
        astar_path   = snap["astar_path"]
        pursuit_path = snap["pursuit_path"]
        target_pos   = snap["target_pos"]

        # Update map image
        im.set_data(np.array(grid, dtype=np.uint8))

        # Frontier scatter
        if f_pts:
            frontier_sc.set_offsets(np.array(f_pts))
        else:
            frontier_sc.set_offsets(np.empty((0, 2)))

        # Drone / target / goal markers
        drone_dot.set_data([px], [py])
        target_dot.set_data(
            [target_pos[0]] if target_pos else [],
            [target_pos[1]] if target_pos else [],
        )
        goal_dot.set_data(
            [goal[0]] if goal else [],
            [goal[1]] if goal else [],
        )

        # A* exploration path (green)
        if astar_path:
            path_xs = [px] + [wp[0] for wp in astar_path]
            path_ys = [py] + [wp[1] for wp in astar_path]
            path_line.set_data(path_xs, path_ys)
        else:
            path_line.set_data([], [])

        # A* pursuit path (magenta)
        if pursuit_path:
            pur_xs = [px] + [wp[0] for wp in pursuit_path]
            pur_ys = [py] + [wp[1] for wp in pursuit_path]
            pursuit_line.set_data(pur_xs, pur_ys)
        else:
            pursuit_line.set_data([], [])

        # Wander heading arrow
        ax_len = 55
        arrow_line.set_data(
            [px, px + math.cos(wander_angle) * ax_len],
            [py, py + math.sin(wander_angle) * ax_len],
        )

        # Render figure to an in-memory PNG buffer, then load as PIL image
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=72, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        gif_frames.append(Image.open(buf).copy())   # .copy() detaches from buf
        buf.close()


    # ── Mission over: stitch all frames into an animated GIF ─────────────────
    if gif_frames:
        gif_frames[0].save(
            _GIF_PATH,
            save_all=True,
            append_images=gif_frames[1:],
            optimize=False,
            duration=300,        # ms per frame  (~3 fps playback)
            loop=0,              # loop forever
        )


# ─────────────────────────────────────────────────────────────────────────────
# Persistent state
# ─────────────────────────────────────────────────────────────────────────────
_state = {
    "grid":            [[_UNKNOWN] * _COLS for _ in range(_ROWS)],
    "wander_angle":    math.atan2(
                           HEIGHT - _WALL_MARGIN - SPAWN_Y,
                           WIDTH  - _WALL_MARGIN - SPAWN_X,
                       ),
    "frontier_goal":   None,   # current target frontier centroid (world xy)
    "frontier_pts":    [],     # raw frontier cells for visualisation
    "astar_path":      [],     # remaining A* waypoints — exploration
    "astar_goal":      None,   # frontier goal A* was last run for
    "pursuit_path":        [],     # remaining A* waypoints — pursuit
    "pursuit_path_target": None,   # target position when pursuit path was planned
    "last_target":     None,
    "last_target_vel": (0.0, 0.0),
    "prev_target_pos": None,
    "lost_timer":      0,
    "obstacles":       [],
    "visit_grid":      [[0] * _COLS for _ in range(_ROWS)],
    "frame":           0,
    "levy_waypoint":   None,   # current Levy flight destination (world xy)
    "clearance_map":   None,   # brushfire clearance distances (rebuilt with frontiers)
}


# ─────────────────────────────────────────────────────────────────────────────
# Main controller
# ─────────────────────────────────────────────────────────────────────────────

def compute_velocity(sensors):
    px    = sensors["player_x"]
    py    = sensors["player_y"]
    lidar = sensors.get("lidar_distances", [])
    obs   = _state["obstacles"]
    grid  = _state["grid"]
    _state["frame"] += 1

    # 1. Update occupancy map
    _update_map(grid, px, py, lidar)
    gc, gr = _world_to_grid(px, py)
    _state["visit_grid"][gr][gc] += 1

    # 2. Recompute frontiers + clearance map every 30 frames
    if _state["frame"] % 30 == 0:
        raw      = _get_frontier_cells(grid)
        clusters = _cluster_frontiers(raw)          # [(world_xy, size), ...]
        _state["frontier_pts"]  = raw
        _state["clearance_map"] = _build_clearance_map(grid)
        goal = _nearest_frontier(px, py, clusters)
        if goal is not None and goal != _state["frontier_goal"]:
            _state["frontier_goal"] = goal
            _state["astar_path"]    = []   # invalidate old path → replan

    # 3. Push snapshot to plot thread every 30 frames (non-blocking) — 2x gif density
    if _state["frame"] % 30 == 0:
        snap = {
            "frame":        _state["frame"],
            "grid":         [row[:] for row in grid],
            "px": px, "py": py,
            "wander_angle": _state["wander_angle"],
            "frontier_pts": list(_state["frontier_pts"]),
            "goal":         _state["frontier_goal"],
            "astar_path":   list(_state["astar_path"]),
            "pursuit_path": list(_state["pursuit_path"]),
            "target_pos":   sensors.get("target_pos"),
        }
        try:
            _plot_queue.put_nowait(snap)
        except queue.Full:
            pass   # drop frame if thread is still busy — never block the game

    # =========================================================================
    # STATE 2: PURSUIT
    # =========================================================================
    if sensors["target_visible"]:
        tx, ty = sensors["target_pos"]

        # Velocity estimation
        if _state["prev_target_pos"] is not None:
            ptx, pty = _state["prev_target_pos"]
            _state["last_target_vel"] = (tx - ptx, ty - pty)
        _state["prev_target_pos"] = (tx, ty)
        _state["last_target"]     = (tx, ty)
        _state["lost_timer"]      = 0

        dist     = math.hypot(tx - px, ty - py)
        tvx, tvy = _state["last_target_vel"]

        # ── Shadow phase: already inside scoring radius ───────────────────────
        # Skip A* — just match speed to stay close without pushing the target.
        if dist <= TRACKING_RADIUS * 0.5:
            vx, vy = _normalize(tx - px, ty - py, MAX_TARGET_SPEED * 0.9)
            _state["pursuit_path"]        = []
            _state["pursuit_path_target"] = None

        # ── A* routed pursuit ─────────────────────────────────────────────────
        else:
            # Replan if: no path, or target has moved more than one cell (20px)
            # from where we last planned.
            ppt = _state["pursuit_path_target"]
            target_moved = (
                ppt is None or
                math.hypot(tx - ppt[0], ty - ppt[1]) > _CELL
            )

            if target_moved or not _state["pursuit_path"]:
                # For far phase, plan to a lead point ahead of the target.
                # For closing phase, plan directly to the target.
                if dist > TRACKING_RADIUS * 1.4:
                    ttr        = dist / MAX_PLAYER_SPEED
                    look_ahead = min(ttr * 0.6, 25)
                    goal_x = max(_WALL_MARGIN, min(WIDTH  - _WALL_MARGIN,
                                                   tx + tvx * look_ahead))
                    goal_y = max(_WALL_MARGIN, min(HEIGHT - _WALL_MARGIN,
                                                   ty + tvy * look_ahead))
                else:
                    goal_x, goal_y = tx, ty

                # Sanitise: if target is inside an obstacle (env bug), snap
                # the A* goal to the nearest FREE cell instead.
                raw_gc, raw_gr = _world_to_grid(goal_x, goal_y)
                safe_gc, safe_gr = _nearest_free_cell(grid, raw_gc, raw_gr)
                if (safe_gc, safe_gr) != (raw_gc, raw_gr):
                    goal_x, goal_y = _grid_to_world(safe_gc, safe_gr)

                _state["pursuit_path"]        = _astar(grid, (px, py),
                                                       (goal_x, goal_y),
                                                       _state["clearance_map"])
                _state["pursuit_path_target"] = (tx, ty)

            # Follow the next waypoint
            path = _state["pursuit_path"]
            if path:
                wx, wy = path[0]
                if math.hypot(wx - px, wy - py) < _FRONTIER_REACH:
                    path.pop(0)
                if path:
                    wx, wy = path[0]
                    vx, vy = _normalize(wx - px, wy - py, MAX_PLAYER_SPEED)
                else:
                    # Path exhausted — fly direct for this frame
                    vx, vy = _normalize(tx - px, ty - py, MAX_PLAYER_SPEED * 0.85)
            else:
                # A* found no path (target behind unmapped cells) — fly direct
                vx, vy = _normalize(tx - px, ty - py, MAX_PLAYER_SPEED * 0.85)

        vx, vy = _slide_velocity(px, py, vx, vy, obs)
        return vx, vy

    # =========================================================================
    # STATE 1: EXPLORATION
    # =========================================================================
    _state["prev_target_pos"] = None
    _state["lost_timer"]     += 1

    # Sub-state A: dash to projected last-known position
    if _state["last_target"] is not None and _state["lost_timer"] < 90:
        lx, ly   = _state["last_target"]
        tvx, tvy = _state["last_target_vel"]
        proj     = min(_state["lost_timer"] * 0.8, 60)
        sx = max(_WALL_MARGIN, min(WIDTH  - _WALL_MARGIN, lx + tvx * proj))
        sy = max(_WALL_MARGIN, min(HEIGHT - _WALL_MARGIN, ly + tvy * proj))
        if math.hypot(sx - px, sy - py) > 15:
            vx, vy = _normalize(sx - px, sy - py, MAX_PLAYER_SPEED)
            vx, vy = _slide_velocity(px, py, vx, vy, obs)
            return vx, vy
        else:
            _state["last_target"] = None

    # Sub-state B: A* path following toward frontier goal
    goal = _state["frontier_goal"]

    if goal is not None:
        gx, gy = goal

        # Arrived at frontier goal
        if math.hypot(gx - px, gy - py) < _FRONTIER_REACH:
            _state["frontier_goal"] = None
            _state["astar_path"]    = []
            goal = None

        else:
            # Replan A* if we have no path or the goal changed
            if not _state["astar_path"]:
                _state["astar_path"] = _astar(grid, (px, py), (gx, gy), _state["clearance_map"])

            # Follow the next waypoint in the A* path
            if _state["astar_path"]:
                wx, wy = _state["astar_path"][0]
                if math.hypot(wx - px, wy - py) < _FRONTIER_REACH:
                    _state["astar_path"].pop(0)   # waypoint reached, advance

                if _state["astar_path"]:
                    wx, wy = _state["astar_path"][0]
                    _state["wander_angle"]  = math.atan2(wy - py, wx - px)
                    _state["wander_angle"] += random.gauss(0, _WANDER_SIGMA * 0.3)
                else:
                    # Path exhausted but not at goal yet — replan next frame
                    _state["wander_angle"] = math.atan2(gy - py, gx - px)
            else:
                # A* found no path (goal unreachable) — random walk toward goal
                _state["wander_angle"]  = math.atan2(gy - py, gx - px)
                _state["wander_angle"] += random.gauss(0, _WANDER_SIGMA)

    if goal is None:
        # ── Levy flight fallback ──────────────────────────────────────────────
        # When no frontier goal is active, use a Levy flight instead of a
        # pure Gaussian drift.  Pick a new Levy waypoint when:
        #   - we have none yet, OR
        #   - we have arrived within FRONTIER_REACH of the current one.
        lw = _state["levy_waypoint"]
        if lw is None or math.hypot(lw[0] - px, lw[1] - py) < _FRONTIER_REACH:
            _state["levy_waypoint"] = _levy_waypoint(px, py, _state["visit_grid"])
            lw = _state["levy_waypoint"]

        lwx, lwy = lw
        _state["wander_angle"] = math.atan2(lwy - py, lwx - px)
        # Small Gaussian jitter keeps the path from being perfectly straight,
        # which helps with obstacle avoidance while preserving ballistic intent.
        _state["wander_angle"] += random.gauss(0, _WANDER_SIGMA * 0.4)

    vx = math.cos(_state["wander_angle"]) * MAX_PLAYER_SPEED
    vy = math.sin(_state["wander_angle"]) * MAX_PLAYER_SPEED
    vx, vy = _slide_velocity(px, py, vx, vy, obs)

    # Stuck recovery
    if math.hypot(vx, vy) < MAX_PLAYER_SPEED * 0.25:
        _state["wander_angle"]  += math.pi + random.gauss(0, 0.5)
        _state["frontier_goal"]  = None
        _state["astar_path"]     = []
        _state["levy_waypoint"]  = None   # resample Levy destination after recovery
        vx = math.cos(_state["wander_angle"]) * MAX_PLAYER_SPEED
        vy = math.sin(_state["wander_angle"]) * MAX_PLAYER_SPEED
        vx, vy = _slide_velocity(px, py, vx, vy, obs)

    spd = math.hypot(vx, vy)
    if spd > MAX_PLAYER_SPEED:
        vx = (vx / spd) * MAX_PLAYER_SPEED
        vy = (vy / spd) * MAX_PLAYER_SPEED

    return vx, vy


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("--- SKYE-X Booting: LiDAR Mapping + Frontier Exploration ---")
    print(f"[MAP] Coverage GIF will be saved to '{_GIF_PATH}' on mission end.")
    env = SkyeEnv()
    _state["obstacles"] = env.obstacles

    # Start background plot thread (daemon so it dies if main crashes)
    t = threading.Thread(target=_plot_thread_fn, args=(env.obstacles,),
                         daemon=True)
    t.start()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        sensors = env.get_sensor_data()
        vx, vy  = compute_velocity(sensors)
        env.step(vx, vy)

        if env.crashed:
            print(f"MISSION FAILED: Drone Crashed! Final Score: {env.score}")
            break
        elif env.mission_over:
            print(f"SIMULATION COMPLETE. Final Score: {env.score} timesteps")
            break

    pygame.quit()

    # Signal plot thread to render final frame and show it
    _plot_queue.put(_SENTINEL)
    t.join(timeout=10)


if __name__ == "__main__":
    main()