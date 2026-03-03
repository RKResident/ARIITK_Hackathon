import math
import pygame
from skye_env import SkyeEnv

def _lidar_repulsion(lidar_distances, num_rays=36, danger_dist=80):
    """Returns (rx, ry) repulsion vector from LiDAR readings."""
    rx, ry = 0.0, 0.0
    if not lidar_distances:
        return rx, ry
    for i, dist in enumerate(lidar_distances):
        if dist < danger_dist and dist > 1e-6:
            angle = (i / num_rays) * 2 * math.pi
            strength = (danger_dist - dist) / danger_dist
            rx -= strength * math.cos(angle)
            ry -= strength * math.sin(angle)
    return rx, ry

def compute_velocity(sensors):
    """
    TASK: Write your dual-state flight controller!

    You receive 'sensors', a dictionary containing:
    - player_x, player_y: Your current coordinates (float).
    - lidar_distances: Array of 36 floats (distances in each direction).
    - target_visible: Boolean. True if the target is within your sensor radius.
    - target_pos: (x, y) tuple of the target. (Is 'None' if target_visible is False).
    """
    px = sensors["player_x"]
    py = sensors["player_y"]
    lidar = sensors.get("lidar_distances", [])

    rx, ry = _lidar_repulsion(lidar)

    vx, vy = 0.0, 0.0

    if not sensors["target_visible"]:
        # ==========================================================
        # STATE 1: EXPLORATION (INTO THE UNKNOWN)
        # ==========================================================
        # The target is hidden in the Fog of War.
        # You must systematically explore the map without hitting obstacles.
        # If you just fly randomly, you will waste time. 
        
        base_vx, base_vy = 0.8, 0.4
    else:
        # ==========================================================
        # STATE 2: THE PURSUIT
        # ==========================================================
        # You found the target! 
        # However, the target's max speed is 2.5, and yours is 3.5
        # You must anticipate its trajectory, cut corners around obstacles,
        # and stay within the 70px tracking radius to maximize your score.
        
        tx, ty = sensors["target_pos"]
        dx, dy = tx - px, ty - py
        dist = math.hypot(dx, dy)
        if dist > 1e-6:
            base_vx = 1.0 * (dx / dist)
            base_vy = 1.0 * (dy / dist)
        else:
            base_vx, base_vy = 0.0, 0.0

    vx = base_vx + rx
    vy = base_vy + ry
    speed = math.hypot(vx, vy)
    if speed > 1.0 and speed > 1e-6:
        vx = (vx / speed) * 1.0
        vy = (vy / speed) * 1.0

    return vx, vy

def main():
    print("--- SKYE-X Booting: Search & Pursuit ---")
    
    # Initialize the simulation environment
    env = SkyeEnv()
    
    running = True
    while running:
        # 1. Handle window closing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        # 2. Read the drone's sensors (Subject to Fog of War)
        sensors = env.get_sensor_data()
        
        # 3. Compute physics-based velocity commands (YOUR CODE)
        vx, vy = compute_velocity(sensors)

        env.step(vx, vy)
        
        # 5. Check for Mission Status
        if env.crashed:
            print(f"MISSION FAILED: Drone Crashed! Final Score: {env.score}")
            break
        elif env.mission_over:
            print(f"SIMULATION COMPLETE. Final Escort Score: {env.score} timesteps")
            break

    pygame.quit()

if __name__ == "__main__":
    main()