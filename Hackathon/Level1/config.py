"""
Operation SKYE-X Configuration
Edit MAX_TIMESTEPS to adjust mission duration, and NUM_OBSTACLES_DEFAULT for obstacle density.
"""

# --- Arena ---
WIDTH = 1280
HEIGHT = 900
FPS = 60

# --- Speeds ---
MAX_PLAYER_SPEED = 3.5
MAX_TARGET_SPEED = 2.5 # Target's normal max speed
MAX_TARGET_SPEED_FLEE = 2.5  # Target speed when fleeing

# --- Distances & Radii ---
VISIBILITY_RADIUS = 150   # Lidar/Fog of War radius
TRACKING_RADIUS = 70      # Radius required to score points
DRONE_RADIUS = 10         # Player drone collision radius
TARGET_VISION = 80        # Range at which target detects player and seeks cover/flees

# --- Mission ---
MAX_TIMESTEPS = 3000      # Mission ends after this many frames

# --- Spawn ---
SPAWN_X = 80.0
SPAWN_Y = 80.0
TARGET_SPAWN_OFFSET = 150  # Target spawns at (WIDTH - offset, HEIGHT - offset)
OBSTACLE_EDGE_MARGIN = 140  # Obstacles spawned at least this far from arena edges

# --- LiDAR ---
NUM_RAYS = 36             # 360° / NUM_RAYS = degrees per ray

# --- Obstacles ---
NUM_OBSTACLES_DEFAULT = 22

# --- Target AI ---
WALL_AVOIDANCE_MARGIN = 60   # Soft boundary for target
WALL_AVOIDANCE_FORCE = 5.0   # Repulsion strength near walls
TARGET_BOUNDARY_PAD = 16     # Hard clamp: target kept pad pixels from edges (DRONE_RADIUS + 6)
COVER_SEARCH_RADIUS = 250    # Only consider obstacles within this for hiding
HIDE_SPOT_OFFSET = 45        # Distance beyond obstacle edge for hide point
