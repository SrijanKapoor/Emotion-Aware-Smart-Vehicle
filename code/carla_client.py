"""
CARLA 0.9.15 Client & Basic Operations
HUD panel rendered as 2D overlay on pygame window (top-left corner).
No more floating world-space debug text.
"""

import carla
import math
import numpy as np
import time
import sys

try:
    import pygame
    PYGAME_AVAILABLE = True
except Exception as e:
    print(f"Warning: pygame import failed: {e}")
    PYGAME_AVAILABLE = False


# ---------------------------------------------------------------------------
# Colours (RGB)
# ---------------------------------------------------------------------------
COL_BG_PANEL   = (10,  15,  40)   # dark navy box background
COL_BG_HEADER  = (18,  24,  60)   # slightly lighter header bar
COL_BORDER     = (50,  80, 180)   # blue border
COL_HEADER_TXT = (255, 225,  40)  # yellow header text
COL_GREEN      = (30,  210,  90)
COL_ORANGE     = (255, 170,   0)
COL_RED        = (255,  50,  50)
COL_GREY       = (170, 180, 200)
COL_DIM        = (90, 100, 130)


class HUDPanel:
    """
    2-D safety-alert panel drawn directly onto a pygame Surface.
    Sits in the top-left corner with a dark-blue rounded box.
    """

    PANEL_X      = 18     # px from left
    PANEL_Y      = 18     # px from top
    PANEL_W      = 310    # panel width
    PADDING      = 14     # inner padding
    ROW_H        = 34     # height of each data row
    HEADER_H     = 36     # height of the header bar
    CORNER_R     = 8      # border-radius
    BORDER_W     = 2      # border thickness

    SPEED_LIMIT      = 80   # km/h
    OBSTACLE_DANGER  = 10   # m
    OBSTACLE_CAUTION = 15   # m

    def __init__(self):
        pygame.font.init()
        self.font_header = pygame.font.SysFont("consolas",  15, bold=True)
        self.font_label  = pygame.font.SysFont("consolas",  13, bold=False)
        self.font_value  = pygame.font.SysFont("consolas",  13, bold=True)
        self.font_ctrl   = pygame.font.SysFont("consolas",  12, bold=False)

    # ------------------------------------------------------------------
    def _draw_rounded_rect(self, surf, color, rect, radius):
        """Fill a rounded rectangle."""
        x, y, w, h = rect
        pygame.draw.rect(surf, color, (x + radius, y,         w - 2*radius, h))
        pygame.draw.rect(surf, color, (x,           y + radius, w,           h - 2*radius))
        for cx, cy in [(x+radius, y+radius), (x+w-radius, y+radius),
                       (x+radius, y+h-radius), (x+w-radius, y+h-radius)]:
            pygame.draw.circle(surf, color, (cx, cy), radius)

    def _draw_rounded_rect_border(self, surf, color, rect, radius, width):
        """Draw only the border of a rounded rectangle."""
        x, y, w, h = rect
        # top / bottom / left / right edges
        pygame.draw.line(surf, color, (x+radius, y),     (x+w-radius, y),     width)
        pygame.draw.line(surf, color, (x+radius, y+h-1), (x+w-radius, y+h-1), width)
        pygame.draw.line(surf, color, (x,     y+radius), (x,     y+h-radius), width)
        pygame.draw.line(surf, color, (x+w-1, y+radius), (x+w-1, y+h-radius), width)
        # corners
        for (cx, cy, sa, ea) in [
            (x+radius,   y+radius,   180, 270),
            (x+w-radius, y+radius,   270, 360),
            (x+radius,   y+h-radius, 90,  180),
            (x+w-radius, y+h-radius, 0,   90),
        ]:
            pygame.draw.arc(surf, color,
                            (cx-radius, cy-radius, radius*2, radius*2),
                            math.radians(sa), math.radians(ea), width)

    # ------------------------------------------------------------------
    def _status_color(self, status):
        return {"CLEAR": COL_GREEN, "NORMAL": COL_GREEN,
                "CAUTION": COL_ORANGE,
                "DANGER": COL_RED, "HIGH": COL_RED}.get(status, COL_GREY)

    # ------------------------------------------------------------------
    def draw(self, surf, speed_kmh, obstacle_dist_raw, throttle, brake, steer):
        """
        Render the HUD panel onto `surf` at the top-left corner.

        obstacle_dist_raw : float | None
            None  → no radar contact → CLEAR
            float → distance in metres
        """
        P  = self.PADDING
        PW = self.PANEL_W
        RH = self.ROW_H
        HH = self.HEADER_H

        # ── Classify obstacle ──────────────────────────────────────────
        if obstacle_dist_raw is None:
            obs_status = "CLEAR"
            obs_str    = "  --.- m"
        elif obstacle_dist_raw < self.OBSTACLE_DANGER:
            obs_status = "DANGER"
            obs_str    = f"{obstacle_dist_raw:5.1f} m"
        elif obstacle_dist_raw < self.OBSTACLE_CAUTION:
            obs_status = "CAUTION"
            obs_str    = f"{obstacle_dist_raw:5.1f} m"
        else:
            obs_status = "CLEAR"
            obs_str    = f"{obstacle_dist_raw:5.1f} m"

        # ── Classify speed ─────────────────────────────────────────────
        spd_status = "HIGH" if speed_kmh > self.SPEED_LIMIT else "NORMAL"
        spd_str    = f"{speed_kmh:5.1f} km/h"

        # ── Panel total height ─────────────────────────────────────────
        # header + 2 data rows + controls row + bottom padding
        CTRL_H   = 28
        panel_h  = HH + RH * 2 + CTRL_H + P
        px, py   = self.PANEL_X, self.PANEL_Y
        rect     = (px, py, PW, panel_h)

        # ── Background box ─────────────────────────────────────────────
        self._draw_rounded_rect(surf, COL_BG_PANEL, rect, self.CORNER_R)

        # ── Border ─────────────────────────────────────────────────────
        self._draw_rounded_rect_border(surf, COL_BORDER, rect,
                                       self.CORNER_R, self.BORDER_W)

        # ── Header bar ────────────────────────────────────────────────
        header_rect = (px, py, PW, HH)
        self._draw_rounded_rect(surf, COL_BG_HEADER, header_rect, self.CORNER_R)
        # flat bottom on header (overwrite rounded bottom corners)
        pygame.draw.rect(surf, COL_BG_HEADER,
                         (px, py + self.CORNER_R, PW, HH - self.CORNER_R))
        # header separator line
        pygame.draw.line(surf, COL_BORDER,
                         (px + P, py + HH),
                         (px + PW - P, py + HH), 1)

        htxt = self.font_header.render("═══  SAFETY ALERTS  ═══", True, COL_HEADER_TXT)
        surf.blit(htxt, (px + PW//2 - htxt.get_width()//2, py + HH//2 - htxt.get_height()//2))

        # ── Row helper ─────────────────────────────────────────────────
        def draw_row(row_idx, label, status, value):
            ry = py + HH + row_idx * RH
            # subtle alternating row tint
            if row_idx % 2 == 1:
                pygame.draw.rect(surf, (15, 20, 52),
                                 (px+2, ry, PW-4, RH))

            # separator
            pygame.draw.line(surf, (30, 40, 90),
                             (px + P, ry + RH - 1),
                             (px + PW - P, ry + RH - 1), 1)

            mid_y = ry + RH//2

            # label (left)
            lbl_surf = self.font_label.render(label, True, COL_DIM)
            surf.blit(lbl_surf, (px + P, mid_y - lbl_surf.get_height()//2))

            # status badge
            sc = self._status_color(status)
            badge_surf = self.font_value.render(status, True, sc)
            bx = px + 120
            # small coloured dot before status
            pygame.draw.circle(surf, sc, (bx - 8, mid_y), 4)
            surf.blit(badge_surf, (bx, mid_y - badge_surf.get_height()//2))

            # value (right-aligned)
            val_surf = self.font_value.render(value, True, COL_GREY)
            surf.blit(val_surf,
                      (px + PW - P - val_surf.get_width(),
                       mid_y - val_surf.get_height()//2))

        draw_row(0, "OBSTACLE", obs_status, obs_str)
        draw_row(1, "SPEED",    spd_status, spd_str)

        # ── Controls row ───────────────────────────────────────────────
        ctrl_y = py + HH + RH * 2 + 6
        ctrl_txt = (f"Throttle {throttle:.2f}   "
                    f"Brake {brake:.2f}   "
                    f"Steer {steer:+.2f}")
        cs = self.font_ctrl.render(ctrl_txt, True, COL_DIM)
        surf.blit(cs, (px + PW//2 - cs.get_width()//2, ctrl_y))


# ---------------------------------------------------------------------------
# CARLAClient
# ---------------------------------------------------------------------------

class CARLAClient:
    def __init__(self, host='localhost', port=2000):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world  = self.client.get_world()
        try:
            settings = self.world.get_settings()
            if getattr(settings, 'synchronous_mode', False):
                settings.synchronous_mode = False
                self.world.apply_settings(settings)
        except Exception:
            pass
        self.vehicle              = None
        self.sensors              = []
        self.min_obstacle_distance = None   # None = no radar contact
        self.camera_image         = None
        print("✓ CARLA Client initialized")

    def _move_spectator_behind(self, spectator):
        try:
            vt  = self.vehicle.get_transform()
            vl, vr = vt.location, vt.rotation
            yr  = math.radians(vr.yaw)
            sl  = carla.Location(
                x=vl.x - 8.0 * math.cos(yr),
                y=vl.y - 8.0 * math.sin(yr),
                z=vl.z + 4.0
            )
            spectator.set_transform(
                carla.Transform(sl, carla.Rotation(pitch=-20.0, yaw=vr.yaw))
            )
        except Exception:
            pass

    def spawn_vehicle(self):
        bp_lib      = self.world.get_blueprint_library()
        vehicle_bps = list(bp_lib.filter('vehicle.*'))
        if not vehicle_bps:
            raise RuntimeError('No vehicle blueprints found')
        m3 = [b for b in vehicle_bps if 'model3' in b.id.lower()]
        bp = m3[0] if m3 else vehicle_bps[0]

        sps = self.world.get_map().get_spawn_points()
        sp  = sps[0] if sps else carla.Transform(carla.Location(z=2))

        self.vehicle = self.world.try_spawn_actor(bp, sp) \
                    or self.world.spawn_actor(bp, sp)
        print(f"✓ Vehicle spawned: {bp.id}")
        time.sleep(0.15)
        try:
            self.vehicle.set_autopilot(False)
        except Exception:
            pass
        self._move_spectator_behind(self.world.get_spectator())
        return self.vehicle

    def attach_collision_sensor(self):
        bp = self.world.get_blueprint_library().find('sensor.other.collision')
        s  = self.world.spawn_actor(bp, carla.Transform(), attach_to=self.vehicle)
        self.sensors.append(s)
        print("✓ Collision sensor attached")
        return s

    def attach_radar(self):
        bp = self.world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', '30')
        bp.set_attribute('vertical_fov',   '30')
        s = self.world.spawn_actor(
            bp, carla.Transform(carla.Location(x=2.5)), attach_to=self.vehicle
        )
        s.listen(self._on_radar_data)
        self.sensors.append(s)
        print("✓ Radar sensor attached")
        return s

    def _on_radar_data(self, radar_data):
        detections = list(radar_data)
        self.min_obstacle_distance = (
            min(d.depth for d in detections) if detections else None
        )

    def attach_camera(self, image_width=800, image_height=600):
        """Attach RGB camera sensor — feeds live frames into self.camera_image."""
        bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(image_width))
        bp.set_attribute('image_size_y', str(image_height))
        bp.set_attribute('fov', '110')
        # Mount slightly behind and above driver eye-level for a good rear view
        s = self.world.spawn_actor(
            bp,
            carla.Transform(carla.Location(x=-5.5, z=2.5),
                            carla.Rotation(pitch=-8.0)),
            attach_to=self.vehicle
        )
        s.listen(self._on_camera_data)
        self.sensors.append(s)
        print("✓ Camera sensor attached")
        return s

    def _on_camera_data(self, image):
        """Store the latest camera frame for pygame rendering."""
        self.camera_image = image

    def apply_control(self, throttle=0.0, brake=0.0, steer=0.0):
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=float(np.clip(throttle, 0, 1)),
            brake   =float(np.clip(brake,    0, 1)),
            steer   =float(np.clip(steer,   -1, 1))
        ))

    def get_velocity(self):
        v = self.vehicle.get_velocity()
        return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

    def get_location(self):
        l = self.vehicle.get_location()
        return (l.x, l.y, l.z)

    def cleanup(self):
        if self.vehicle:
            self.vehicle.destroy()
        for s in self.sensors:
            s.destroy()
        print("✓ Cleanup complete")


# ---------------------------------------------------------------------------
# Pygame demo with 2D HUD panel
# ---------------------------------------------------------------------------

def demo_carla_pygame(total_steps=None, fps=30):
    """
    Keyboard control (W/A/S/D) with a 2D safety-alert panel
    rendered in the top-left corner of the pygame window.

    The pygame window is now full-size (800×600) and shows the
    CARLA camera feed as the background so everything looks like
    a proper driving HUD.
    """
    if not PYGAME_AVAILABLE:
        raise RuntimeError('Install pygame:  pip install pygame')

    print("\n[Demo] Starting — HUD panel in top-left of pygame window\n")

    WIN_W, WIN_H = 800, 600

    client = CARLAClient()
    hud    = HUDPanel()

    try:
        client.spawn_vehicle()
        client.attach_collision_sensor()
        client.attach_radar()
        client.attach_camera(WIN_W, WIN_H)   # attach camera for background feed

        spectator = client.world.get_spectator()

        pygame.init()
        screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption('CARLA Safety HUD')
        pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN, pygame.KEYUP])
        clock = pygame.time.Clock()

        throttle   = 0.0
        brake      = 0.0
        steer      = 0.0
        hand_brake = False
        reverse    = False

        # Background surface (filled with dark road color when no camera yet)
        bg_color = (28, 32, 28)

        step    = 0
        running = True

        while running and (total_steps is None or step < total_steps):

            # ── Events ──────────────────────────────────────────────
            for event in pygame.event.get():
                if event.type == pygame.QUIT:           # FIXED
                    running = False; break
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):  # FIXED
                        running = False; break
                    if event.key == pygame.K_r:          # FIXED
                        reverse = not reverse
                        print(f'Reverse: {reverse}')
                    if event.key == pygame.K_SPACE:      # FIXED
                        hand_brake = not hand_brake
                        print(f'Hand-brake: {hand_brake}')

            if not running:
                break

            # ── Key → controls ───────────────────────────────────────
            keys = pygame.key.get_pressed()

            throttle = min(throttle + 0.03, 1.0) if keys[pygame.K_w] \
                  else max(throttle - 0.02, 0.0)                        # FIXED
            brake    = min(brake + 0.06,    1.0) if keys[pygame.K_s] \
                  else max(brake - 0.03,    0.0)                        # FIXED
            if keys[pygame.K_a]:                                        # FIXED
                steer = max(steer - 0.06, -1.0)
            elif keys[pygame.K_d]:                                      # FIXED
                steer = min(steer + 0.06,  1.0)
            else:
                steer *= 0.9

            # ── Apply control ────────────────────────────────────────
            ctrl = carla.VehicleControl(
                throttle=throttle, brake=brake, steer=steer,
                hand_brake=hand_brake, reverse=reverse
            )
            try:
                client.vehicle.apply_control(ctrl)
            except Exception:
                pass

            # ── World tick ───────────────────────────────────────────
            try:
                client.world.wait_for_tick(timeout=1.0)
            except Exception:
                try: client.world.tick()
                except Exception: pass

            if client.vehicle is None:
                break

            # ── Spectator follow ─────────────────────────────────────
            client._move_spectator_behind(spectator)

            # ── Draw frame ───────────────────────────────────────────
            # Background: use CARLA camera image if available
            if client.camera_image is not None:
                img = client.camera_image
                # Convert CARLA raw image → pygame surface
                array = np.frombuffer(img.raw_data, dtype=np.uint8)
                array = array.reshape((img.height, img.width, 4))
                array = array[:, :, :3][:, :, ::-1]   # BGRA → RGB
                cam_surf = pygame.surfarray.make_surface(array.swapaxes(0, 1))
                screen.blit(pygame.transform.scale(cam_surf, (WIN_W, WIN_H)), (0, 0))
            else:
                screen.fill(bg_color)

            # ── 2D HUD panel (top-left) ──────────────────────────────
            try:
                speed = client.get_velocity()
            except Exception:
                speed = 0.0

            hud.draw(
                screen,
                speed_kmh        = speed,
                obstacle_dist_raw= client.min_obstacle_distance,
                throttle         = throttle,
                brake            = brake,
                steer            = steer,
            )

            # ── Key hint (bottom strip) ──────────────────────────────
            hint_font = pygame.font.SysFont("consolas", 12)
            hint = hint_font.render(
                "W=accel  S=brake  A/D=steer  R=reverse  SPACE=handbrake  Q=quit",
                True, (100, 110, 140)
            )
            screen.blit(hint, (10, WIN_H - 20))

            pygame.display.flip()
            clock.tick(fps)
            step += 1

        pygame.quit()
        print('✓ Done')

    finally:
        client.cleanup()


# ---------------------------------------------------------------------------
# Auto demo (no pygame)
# ---------------------------------------------------------------------------

def demo_carla_client(manual=False, total_steps=3000):
    print("\n" + "=" * 60)
    print("CARLA CLIENT DEMO - BASIC VEHICLE CONTROL")
    print("=" * 60 + "\n")

    client = CARLAClient()
    try:
        client.spawn_vehicle()
        client.attach_collision_sensor()
        client.attach_radar()

        spectator = client.world.get_spectator()
        throttle = brake = steer = 0.0

        for step in range(total_steps):
            if not manual:
                steer    = 0.6 * np.sin(step / 50.0)
                brake    = 0.6 if (step % 400) >= 360 else 0.0
                throttle = 0.15 if brake else 0.5

            if manual and step % 20 == 0:
                try:
                    user = input("throttle brake steer (or q): ").strip()
                except EOFError:
                    user = ''
                if user.lower() == 'q': break
                if user:
                    pts = user.split()
                    try:
                        throttle = float(pts[0]) if pts else throttle
                        brake    = float(pts[1]) if len(pts)>1 else brake
                        steer    = float(pts[2]) if len(pts)>2 else steer
                    except Exception: pass

            try: client.apply_control(throttle, brake, steer)
            except Exception: pass

            try: client.world.wait_for_tick(timeout=1.0)
            except Exception:
                try: client.world.tick()
                except Exception: pass

            if client.vehicle is None: break
            client._move_spectator_behind(spectator)

            if step % 20 == 0:
                try:
                    spd = client.get_velocity()
                    loc = client.get_location()
                    obs = client.min_obstacle_distance
                    obs_str = f"{obs:.1f} m" if obs is not None else "--"
                    print(f"  Step {step:4d} | {spd:5.1f} km/h | "
                          f"obs={obs_str} | "
                          f"({loc[0]:.1f},{loc[1]:.1f})")
                except Exception:
                    pass
    finally:
        try: client.cleanup()
        except Exception: pass
        print("\n✓ Demo complete!\n")


if __name__ == "__main__":
    demo_carla_pygame()