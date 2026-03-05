import pygame
import math
import random

from config import (
    WIDTH, HEIGHT, FPS,
    MAX_PLAYER_SPEED, MAX_TARGET_SPEED, MAX_TARGET_SPEED_FLEE,
    VISIBILITY_RADIUS, TRACKING_RADIUS, DRONE_RADIUS, TARGET_VISION,
    MAX_TIMESTEPS, SPAWN_X, SPAWN_Y, TARGET_SPAWN_OFFSET,
    OBSTACLE_EDGE_MARGIN, NUM_RAYS, NUM_OBSTACLES_DEFAULT,
    WALL_AVOIDANCE_MARGIN, WALL_AVOIDANCE_FORCE, TARGET_BOUNDARY_PAD,
    COVER_SEARCH_RADIUS, HIDE_SPOT_OFFSET,
)

def _draw_drone(screen, x, y, body_color, accent_color, enemy=False):
    """Draw a quadcopter drone: central body + 4 rotor arms."""
    ix, iy = int(x), int(y)
    arm_len = 14
    rotor_r = 5
    pygame.draw.line(screen, accent_color, (ix - arm_len, iy - arm_len), (ix + arm_len, iy + arm_len), 2)
    pygame.draw.line(screen, accent_color, (ix - arm_len, iy + arm_len), (ix + arm_len, iy - arm_len), 2)
    for dx, dy in [(-1, -1), (1, -1), (-1, 1), (1, 1)]:
        rx = ix + dx * arm_len
        ry = iy + dy * arm_len
        pygame.draw.circle(screen, (90, 95, 100), (rx, ry), rotor_r)
        pygame.draw.circle(screen, (140, 145, 150), (rx, ry), rotor_r - 1)
    pygame.draw.circle(screen, body_color, (ix, iy), 8)
    pygame.draw.circle(screen, accent_color, (ix, iy), 8, 1)
    if enemy:
        pygame.draw.circle(screen, (255, 80, 80), (ix, iy), 4)


def _ray_aabb_t(px, py, dx, dy, cx, cy, hw, hh):
    """Ray-AABB intersection. Returns smallest positive t, or None."""
    eps = 1e-9
    inv_dx = 1.0 / (dx + eps) if abs(dx) > eps else 1e9
    inv_dy = 1.0 / (dy + eps) if abs(dy) > eps else 1e9
    t0_x = ((cx - hw) - px) * inv_dx
    t1_x = ((cx + hw) - px) * inv_dx
    t0_y = ((cy - hh) - py) * inv_dy
    t1_y = ((cy + hh) - py) * inv_dy
    t_near = max(min(t0_x, t1_x), min(t0_y, t1_y))
    t_far = min(max(t0_x, t1_x), max(t0_y, t1_y))
    if t_near <= t_far and t_far > 0:
        return t_near if t_near > 0 else t_far
    return None


class SkyeEnv:
    def __init__(self, num_obstacles=None):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Operation SKYE-X: Search and Pursuit")
        self.clock = pygame.time.Clock()

        if num_obstacles is None:
            num_obstacles = NUM_OBSTACLES_DEFAULT

        self.player_pos = [SPAWN_X, SPAWN_Y]
        self.crashed = False

        self.target_pos = [WIDTH - TARGET_SPAWN_OFFSET, HEIGHT - TARGET_SPAWN_OFFSET]
        self.target_vel = [0.0, 0.0]
        self.target_wander_angle = random.uniform(0, 2 * math.pi)
        self.target_discovered = False

        self.score = 0
        self.timesteps = 0
        self.max_timesteps = MAX_TIMESTEPS
        self.mission_over = False

        self.obstacles = []
        shapes = [
            (28, 28), (22, 40), (40, 22), (35, 35), (18, 50), (50, 18),
            (25, 32), (32, 25), (30, 38), (38, 30)
        ]
        for _ in range(num_obstacles):
            for attempt in range(50):
                cx = random.randint(OBSTACLE_EDGE_MARGIN, WIDTH - OBSTACLE_EDGE_MARGIN)
                cy = random.randint(OBSTACLE_EDGE_MARGIN, HEIGHT - OBSTACLE_EDGE_MARGIN)
                hw, hh = random.choice(shapes)
                if math.hypot(cx - SPAWN_X, cy - SPAWN_Y) > OBSTACLE_EDGE_MARGIN + max(hw, hh):
                    break
            self.obstacles.append((cx, cy, hw, hh))

        self.lidar_readings = []
        self.num_rays = NUM_RAYS
        self._update_lidar()

    def _update_lidar(self):
        px, py = self.player_pos
        self.num_rays = NUM_RAYS
        self.lidar_readings = []

        for i in range(self.num_rays):
            angle = (i / self.num_rays) * 2 * math.pi
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            min_dist = VISIBILITY_RADIUS

            if abs(cos_a) > 1e-9:
                tw_x = (WIDTH - px) / cos_a if cos_a > 0 else -px / cos_a
                if 0 < tw_x < min_dist:
                    min_dist = tw_x
            if abs(sin_a) > 1e-9:
                tw_y = (HEIGHT - py) / sin_a if sin_a > 0 else -py / sin_a
                if 0 < tw_y < min_dist:
                    min_dist = tw_y

            for cx, cy, hw, hh in self.obstacles:
                t = _ray_aabb_t(px, py, cos_a, sin_a, cx, cy, hw, hh)
                if t is not None and 0 < t < min_dist:
                    min_dist = t

            self.lidar_readings.append(min_dist)

    def _update_target_ai(self):
        """Adversarial AI: Gaussian Random Walk + LoS Breaking + Wall Sliding."""
        tx, ty = self.target_pos
        px, py = self.player_pos

        # 1. Base Gaussian Random Walk
        self.target_wander_angle += random.gauss(0, 0.15)
        desired_vx = math.cos(self.target_wander_angle) * MAX_TARGET_SPEED
        desired_vy = math.sin(self.target_wander_angle) * MAX_TARGET_SPEED

        # 2. Flee and Seek Cover
        dist_to_player = math.hypot(tx - px, ty - py)
        if dist_to_player < TARGET_VISION:
            best_cover_dist = float('inf')
            cover_x, cover_y = None, None

            for cx, cy, hw, hh in self.obstacles:
                r_eff = max(hw, hh)
                dist_to_obs = math.hypot(tx - cx, ty - cy)

                if dist_to_obs < COVER_SEARCH_RADIUS:
                    vec_po_x = cx - px
                    vec_po_y = cy - py
                    po_mag = math.hypot(vec_po_x, vec_po_y)

                    if po_mag > 0:
                        hide_x = cx + (vec_po_x / po_mag) * (r_eff + HIDE_SPOT_OFFSET)
                        hide_y = cy + (vec_po_y / po_mag) * (r_eff + HIDE_SPOT_OFFSET)

                        dist_to_hide = math.hypot(tx - hide_x, ty - hide_y)
                        if dist_to_hide < best_cover_dist:
                            best_cover_dist = dist_to_hide
                            cover_x, cover_y = hide_x, hide_y

            if cover_x is not None:
                seek_x = cover_x - tx
                seek_y = cover_y - ty
                seek_mag = math.hypot(seek_x, seek_y)
                if seek_mag > 0:
                    desired_vx += (seek_x / seek_mag) * 3.5
                    desired_vy += (seek_y / seek_mag) * 3.5
            else:
                escape_x = tx - px
                escape_y = ty - py
                mag = math.hypot(escape_x, escape_y)
                if mag > 0:
                    desired_vx += (escape_x / mag) * 3.0
                    desired_vy += (escape_y / mag) * 3.0

        # 3. Wall Avoidance
        if tx < WALL_AVOIDANCE_MARGIN:
            desired_vx += WALL_AVOIDANCE_FORCE
        if tx > WIDTH - WALL_AVOIDANCE_MARGIN:
            desired_vx -= WALL_AVOIDANCE_FORCE
        if ty < WALL_AVOIDANCE_MARGIN:
            desired_vy += WALL_AVOIDANCE_FORCE
        if ty > HEIGHT - WALL_AVOIDANCE_MARGIN:
            desired_vy -= WALL_AVOIDANCE_FORCE

        # 4. Normalise to MAX_TARGET_SPEED
        speed = math.hypot(desired_vx, desired_vy)
        if speed > 0:
            tvx = (desired_vx / speed) * MAX_TARGET_SPEED
            tvy = (desired_vy / speed) * MAX_TARGET_SPEED
        else:
            tvx, tvy = 0.0, 0.0

        # 5. CORRECTED WALL SLIDING
        #
        # Original bugs:
        #   a) `break` after first obstacle — corrected velocity could still
        #      enter a second obstacle unchecked.
        #   b) The tangential axis was forced to MAX_TARGET_SPEED regardless
        #      of its actual value — giving free speed boosts into other walls.
        #   c) next_x/next_y were never updated between iterations, so each
        #      obstacle tested the original pre-correction position.
        #
        # Fix: iterate ALL obstacles without break, recompute next_x/next_y
        # after each correction, and PRESERVE the existing tangential speed
        # instead of overwriting it.

        for cx, cy, hw, hh in self.obstacles:
            # Recompute candidate next position with current (possibly already
            # corrected) velocity — this is the key fix for bug (c).
            next_x = tx + tvx
            next_y = ty + tvy

            bx = hw + 13
            by = hh + 13

            if abs(next_x - cx) >= bx or abs(next_y - cy) >= by:
                continue   # not overlapping this obstacle

            overlap_x = bx - abs(next_x - cx)
            overlap_y = by - abs(next_y - cy)

            if overlap_x < overlap_y:
                # Nearest face is left or right — zero the x component only.
                # PRESERVE tvy rather than forcing it to MAX_TARGET_SPEED (bug b fix).
                tvx = 0.0
                # If tangential speed is also zero, nudge along y to escape.
                if tvy == 0.0:
                    tvy = random.choice([-1.0, 1.0]) * MAX_TARGET_SPEED
            else:
                # Nearest face is top or bottom — zero the y component only.
                tvy = 0.0
                if tvx == 0.0:
                    tvx = random.choice([-1.0, 1.0]) * MAX_TARGET_SPEED
            # No break — continue checking remaining obstacles with updated tvx/tvy

        # Re-normalise after slide corrections to keep speed consistent
        slide_speed = math.hypot(tvx, tvy)
        if slide_speed > MAX_TARGET_SPEED:
            tvx = (tvx / slide_speed) * MAX_TARGET_SPEED
            tvy = (tvy / slide_speed) * MAX_TARGET_SPEED

        if tvx != 0 or tvy != 0:
            self.target_wander_angle = math.atan2(tvy, tvx)

        # 6. Apply Final Movement
        self.target_vel = [tvx, tvy]
        self.target_pos[0] += tvx
        self.target_pos[1] += tvy

        # 7. Hard boundary clamp
        pad = TARGET_BOUNDARY_PAD
        self.target_pos[0] = max(pad, min(WIDTH  - pad, self.target_pos[0]))
        self.target_pos[1] = max(pad, min(HEIGHT - pad, self.target_pos[1]))

    def get_sensor_data(self):
        px, py = self.player_pos
        sensors = {
            "player_x": px,
            "player_y": py,
            "lidar_distances": self.lidar_readings,
            "target_visible": False,
            "target_pos": None
        }

        dist_to_target = math.hypot(px - self.target_pos[0], py - self.target_pos[1])
        if dist_to_target <= VISIBILITY_RADIUS:
            self.target_discovered = True
            sensors["target_visible"] = True
            sensors["target_pos"] = (self.target_pos[0], self.target_pos[1])

        return sensors

    def step(self, vx, vy):
        if self.crashed or self.mission_over: return

        self.timesteps += 1
        if self.timesteps >= self.max_timesteps:
            self.mission_over = True

        speed = math.hypot(vx, vy)
        if speed > MAX_PLAYER_SPEED:
            vx = (vx / speed) * MAX_PLAYER_SPEED
            vy = (vy / speed) * MAX_PLAYER_SPEED

        self.player_pos[0] += vx
        self.player_pos[1] += vy
        self._update_target_ai()

        px, py = self.player_pos
        if px < 0 or px > WIDTH or py < 0 or py > HEIGHT:
            self.crashed = True

        for cx, cy, hw, hh in self.obstacles:
            closest_x = max(cx - hw, min(px, cx + hw))
            closest_y = max(cy - hh, min(py, cy + hh))
            if math.hypot(px - closest_x, py - closest_y) < DRONE_RADIUS:
                self.crashed = True

        self._update_lidar()

        if self.target_discovered:
            dist = math.hypot(px - self.target_pos[0], py - self.target_pos[1])
            if dist <= TRACKING_RADIUS:
                self.score += 1

        self._render()

    def _render(self):
        border = 4
        pygame.draw.rect(self.screen, (35, 38, 42), (0, 0, WIDTH, HEIGHT), border * 2)
        pygame.draw.rect(self.screen, (70, 75, 82), (border, border, WIDTH - 2 * border, HEIGHT - 2 * border), 1)

        tile_size = 48
        floor_dark = (52, 55, 58)
        floor_light = (62, 66, 70)
        for y in range(0, HEIGHT + tile_size, tile_size):
            for x in range(0, WIDTH + tile_size, tile_size):
                c = floor_light if (x // tile_size + y // tile_size) % 2 == 0 else floor_dark
                pygame.draw.rect(self.screen, c, (x, y, tile_size, tile_size))

        for cx, cy, hw, hh in self.obstacles:
            x1, y1 = int(cx - hw), int(cy - hh)
            w, h = int(hw * 2), int(hh * 2)
            pygame.draw.ellipse(self.screen, (20, 22, 25), (x1, int(cy + hh * 0.4), w, max(6, h // 4)))
            pygame.draw.rect(self.screen, (160, 95, 55), (x1, y1, w, h))
            pygame.draw.rect(self.screen, (120, 70, 40), (x1, y1, w, h), 2)
            top_h = min(8, h // 4)
            pygame.draw.rect(self.screen, (195, 120, 70), (x1 + 2, y1 + 2, w - 4, top_h))
            if h > w:
                pygame.draw.line(self.screen, (230, 180, 40), (cx - 4, cy - hh // 2), (cx + 4, cy - hh // 2), 2)
                pygame.draw.line(self.screen, (230, 180, 40), (cx - 4, cy + hh // 2), (cx + 4, cy + hh // 2), 2)

        fog = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        fog.fill((25, 30, 35, 150))
        px, py = int(self.player_pos[0]), int(self.player_pos[1])
        pygame.draw.circle(fog, (0, 0, 0, 0), (px, py), VISIBILITY_RADIUS)
        pygame.draw.circle(fog, (40, 120, 180, 35), (px, py), TRACKING_RADIUS, 2)
        self.screen.blit(fog, (0, 0))

        _draw_drone(self.screen, px, py, (70, 160, 255), (120, 200, 255))

        tx, ty = int(self.target_pos[0]), int(self.target_pos[1])
        _draw_drone(self.screen, tx, ty, (220, 60, 60), (255, 100, 100), enemy=True)

        font = pygame.font.SysFont(None, 26)
        panel = pygame.Surface((320, 48))
        panel.set_alpha(200)
        panel.fill((30, 33, 36))
        self.screen.blit(panel, (10, 10))
        score_text = font.render(f"Tracking: {self.score}  |  Time: {self.timesteps}/{self.max_timesteps}", True, (220, 225, 230))
        self.screen.blit(score_text, (18, 18))

        pygame.display.flip()
        self.clock.tick(FPS)

    def run_human(self):
        vx, vy = 0, 0
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
            keys = pygame.key.get_pressed()
            vx = vy = 0
            if keys[pygame.K_w] or keys[pygame.K_UP]:    vy = -MAX_PLAYER_SPEED
            if keys[pygame.K_s] or keys[pygame.K_DOWN]:  vy =  MAX_PLAYER_SPEED
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:  vx = -MAX_PLAYER_SPEED
            if keys[pygame.K_d] or keys[pygame.K_RIGHT]: vx =  MAX_PLAYER_SPEED
            self.get_sensor_data()
            self.step(vx, vy)
            if self.crashed or self.mission_over:
                running = False
        print(self.score)
        pygame.quit()


if __name__ == "__main__":
    env = SkyeEnv()
    env.run_human()