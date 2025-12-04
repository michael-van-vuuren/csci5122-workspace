import pygame
import random
import config
from config import *


# cow class
class Cow:
    def get_shapes(self):
        shapes = []
        cow_x = 480
        cow_y = GROUND_Y - 50
        BODY = (255, 255, 255)
        SPOT = COLOR_PLATFORM
        BLACK = (0, 0, 0)
        WHITE = (255, 255, 255)
        PINK = (255, 180, 200)
        body_rect = pygame.Rect(cow_x, cow_y, 80, 40)
        shapes.append({'type': 'ellipse', 'color': BODY, 'rect': body_rect})
        shapes.append({'type': 'circle',  'color': SPOT, 'position': (cow_x + 20, cow_y + 20), 'radius': 10})
        shapes.append({'type': 'circle',  'color': SPOT, 'position': (cow_x + 50, cow_y + 15), 'radius': 12})
        head_rect = pygame.Rect(cow_x - 30,  cow_y + 5, 35, 28)
        shapes.append({'type': 'ellipse', 'color': BODY, 'rect': head_rect})
        snout_rect = pygame.Rect(cow_x - 30, cow_y + 18, 30, 18)
        shapes.append({'type': 'ellipse', 'color': PINK, 'rect': snout_rect})
        ear_rect1 = pygame.Rect(cow_x - 22, cow_y - 3, 6, 9)
        ear_rect2 = pygame.Rect(cow_x - 8,  cow_y - 3, 6, 9)
        shapes.append({'type': 'ellipse', 'color': BLACK, 'rect': ear_rect1})
        shapes.append({'type': 'ellipse', 'color': BLACK, 'rect': ear_rect2})
        ear_rect1 = pygame.Rect(cow_x - 28, cow_y + 2, 10, 6)
        ear_rect2 = pygame.Rect(cow_x - 6,  cow_y + 2, 10, 6)
        shapes.append({'type': 'ellipse', 'color': WHITE, 'rect': ear_rect1})
        shapes.append({'type': 'ellipse', 'color': WHITE, 'rect': ear_rect2})
        eye_rect1 = pygame.Rect(cow_x - 22, cow_y + 10, 6, 10)
        eye_rect2 = pygame.Rect(cow_x - 10, cow_y + 10, 6, 10)
        shapes.append({'type': 'ellipse', 'color': BLACK, 'rect': eye_rect1})
        shapes.append({'type': 'ellipse', 'color': BLACK, 'rect': eye_rect2})
        shapes.append({'type': 'circle',  'color': WHITE, 'position': (cow_x - 20, cow_y + 13),  'radius': 1})
        shapes.append({'type': 'circle',  'color': WHITE, 'position': (cow_x - 8,  cow_y + 13),  'radius': 1})
        shapes.append({'type': 'circle',  'color': SPOT,  'position': (cow_x - 17, cow_y + 25),  'radius': 2})
        shapes.append({'type': 'circle',  'color': SPOT,  'position': (cow_x - 11,  cow_y + 25), 'radius': 2})
        for leg_x in [16, 24, 58, 66]:
            shapes.append({'type': 'rect', 'color': BODY,  'rect': (cow_x + leg_x, cow_y + 30, 6, 20)})
            shapes.append({'type': 'rect', 'color': BLACK, 'rect': (cow_x + leg_x, cow_y + 50, 6, 5)})
        shapes.append({'type': 'line', 'color': BLACK, 'start': (cow_x + 75, cow_y + 10), 'end': (cow_x + 85, cow_y + 25), 'width': 3})
        tail_rect = pygame.Rect(cow_x + 85, cow_y + 25, 8, 4)
        shapes.append({'type': 'ellipse', 'color': BLACK, 'rect': tail_rect})
        return shapes


# cloud class
class Cloud:
    def __init__(self, screen_w):
        self.screen_w = screen_w
        self.x = random.randint(0, 600)
        self.y = random.randint(10, 220)
        self.wind_offset = random.uniform(-3, 3)
        self.circles = self.generate_circles()

    def reset(self, left_side=True):
        self.x = random.randint(-200, -100) if left_side else random.randint(self.screen_w + 100, self.screen_w + 200)
        self.y = random.randint(10, 220)
        self.wind_offset = random.uniform(-3, 3)
        self.circles = self.generate_circles()

    def generate_circles(self):
        circles = []
        grid_w = random.randint(4, 5)
        grid_h = random.randint(2, 3)
        cell_w = 15
        cell_h = 8
        base_radius = random.randint(14, 20)
        pos_jitter = 4
        bottom_extra_jitter = 3
        radius_jitter = 2

        for gy in range(grid_h):
            for gx in range(grid_w):
                cx = gx * cell_w + random.randint(-pos_jitter, pos_jitter)
                cy = gy * cell_h + random.randint(-pos_jitter, pos_jitter)

                if gy == grid_h - 1:
                    spread_factor = 2.0
                    center_adjust = (grid_w - 1) * cell_w / 2
                    cx = (cx - center_adjust) * spread_factor + center_adjust
                    cy += 10 + random.randint(-bottom_extra_jitter, bottom_extra_jitter)
                    radius = int(base_radius * 1.5)
                else:
                    radius = base_radius

                radius += random.randint(-radius_jitter, radius_jitter)
                alpha = 225

                circles.append((cx, cy, radius, alpha))

        return circles

    def update(self, dt):
        self.x += (config.WIND_SPEED + self.wind_offset) * dt * 3
        # wind going right/left
        if config.WIND_SPEED >= 0:
            if self.x > self.screen_w + 150:
                self.reset(left_side=True)
        else:
            if self.x < -150:
                self.reset(left_side=False)

    def get_shapes(self):
        shapes = []
        for ox, oy, r, a in self.circles:
            color = (255, 255, 255, a)
            shapes.append({
                'type': 'circle', 
                'color': color, 
                'position': (self.x + ox, self.y + oy), 
                'radius': r, 
                'alpha': True
            })
        return shapes


# flame class
class Flame:
    def get_shapes(self, x, y, theta):
        flame_angle = theta + math.pi
        nozzle_offset = ROCKET_H / 2
        nozzle_x = x + nozzle_offset * math.cos(flame_angle)
        nozzle_y = y + nozzle_offset * math.sin(flame_angle)

        # strobing flame animation
        shapes = []
        for i in range(3):
            flame_length = random.randint(10, 40)
            flame_width = random.randint(ROCKET_W - 10, ROCKET_W)
            tip_x = nozzle_x + flame_length * math.cos(flame_angle)
            tip_y = nozzle_y + flame_length * math.sin(flame_angle)
            perp_x = (flame_width/2) * math.cos(flame_angle + math.pi/2)
            perp_y = (flame_width/2) * math.sin(flame_angle + math.pi/2)
            flame_points = [(tip_x, tip_y),
                            (nozzle_x + perp_x, nozzle_y + perp_y),
                            (nozzle_x - perp_x, nozzle_y - perp_y)]
            if i < 2:
                color = (COLOR_FLAME_EDGE[0], random.randint(COLOR_FLAME_EDGE[1], 165), COLOR_FLAME_EDGE[2])
            else:
                color = (COLOR_FLAME_CORE[0], COLOR_FLAME_CORE[1], random.randint(COLOR_FLAME_CORE[2], 150))
            shapes.append({
                'type': 'polygon', 
                'color': color, 
                'points': flame_points
            })
        return shapes


# rocket class
class Rocket:
    def __init__(self, screen_w):
        self.screen_w = screen_w
        self.flame = Flame()
        self.reset()

    def reset(self):
        self.x = 500                                    # initial x position
        self.y = 100                                    # initial y position
        self.vx = -60                                   # initial x velocity
        self.vy = 60                                    # initial y velocity
        self.theta_offset = -0.4                        # initial offset of angle
        self.theta = -math.pi/2 + self.theta_offset     # initial angle 
        self.omega = 0                                  # initial rate of change in angle over time
        self.speed = math.sqrt(self.vx**2 + self.vy**2) # total velocity magnitude
        self.is_tipping = False
        self.tipping_direction = 0
        self.game_state = 'RUNNING'
        self.thrust_active = False

        # network related
        # even though tanh is between -1 and 1, setting a higher threshold 
        # than 0 reduces the noise of the rocket's actions
        self.action_threshold = 0.5
        self.fuel_used = 0.0
        self.time_taken = 0
        self.fitness = 0.0

    # network inputs
    def get_inputs(self):
        return [
            self.x / SCREEN_W,       # -1 (x-pos)
            self.y / SCREEN_H,       # -2 (y-pos)
            self.vx / 100.0,         # -3 (x-velocity)
            self.vy / 100.0,         # -4 (y-velocity)
            self.theta,              # -5 (angle)
            self.omega,              # -6 (angular velocity)
            config.WIND_SPEED / 20.0 # -7 (wind sensor)
        ]

    def step(self, control_input, dt, mode='player'):
        if mode == 'player':
            thrust = control_input[pygame.K_UP] or control_input[pygame.K_w]
            left   = control_input[pygame.K_LEFT] or control_input[pygame.K_a]
            right  = control_input[pygame.K_RIGHT] or control_input[pygame.K_d]
        elif mode == 'nn':
            threshold = self.action_threshold
            # action: [thrust_probability, left_probability, right_probability]
            thrust = control_input[0] > threshold
            left   = control_input[1] > threshold
            right  = control_input[2] > threshold

        self.thrust_active = thrust

        # rocket on the pad
        if self.game_state == 'LANDED':
            if not self.is_tipping:
                current_deviation = self.theta - (-math.pi/2)
                if abs(current_deviation) > MAX_STABLE_ANGLE:
                    self.is_tipping = True
                    self.tipping_direction = 1 if current_deviation > 0 else -1
            if self.is_tipping:            
                self.omega += self.tipping_direction * TIPPING_ACCELERATION * dt
                self.theta += self.omega * dt                        
                if abs(self.theta - (-math.pi/2)) > math.pi / 2: 
                    self.game_state = 'EXPLODED'

        # rocket in the air         
        if self.game_state == 'RUNNING':
            ax = 0
            ay = G
            
            # effect of wind
            ax += config.WIND_SPEED / MASS

            # thrust
            if thrust:
                ax += THRUST_POWER * math.cos(self.theta) / MASS
                ay += THRUST_POWER * math.sin(self.theta) / MASS

            # angle
            alpha = 0
            if left: alpha -= TORQUE_POWER / I
            if right: alpha += TORQUE_POWER / I

            # update velocity, position, and update angle
            self.vx += ax * dt
            self.vy += ay * dt
            self.x += self.vx * dt
            self.y += self.vy * dt
            self.omega += alpha * dt
            self.theta += self.omega * dt
            self.speed = math.sqrt(self.vx**2 + self.vy**2)

            # rocket hitbox
            rocket_rect = pygame.Rect(0, 0, ROCKET_W, ROCKET_H)
            rocket_rect.center = (self.x, self.y)

            # rocket collision with ground/platform 
            if rocket_rect.colliderect(GROUND_RECT):
                self.game_state = 'EXPLODED'         
            elif rocket_rect.colliderect(PLATFORM_RECT):
                if math.sqrt(self.vx**2 + self.vy**2) > EXPLOSION_VELOCITY_LIMIT:
                    self.game_state = 'EXPLODED'
                else:
                    self.game_state = 'LANDED'
                    self.vy = self.vx = 0 
                    self.y = PLAT_Y - ROCKET_H // 2

            # rocket collision with screen walls
            if self.x < 0:             self.x = 0;             self.vx = 0
            if self.x > self.screen_w: self.x = self.screen_w; self.vx = 0
            if self.y < 0:             self.y = 0;             self.vy = 0

    def get_shapes(self):
        shapes = []
        if self.game_state in ['RUNNING', 'LANDED']:
            # flame if thrust active
            if self.thrust_active and self.game_state == 'RUNNING':
                shapes.extend(self.flame.get_shapes(self.x, self.y, self.theta))
            # rocket body
            shapes.append({
                'type': 'rocket', 
                'color': COLOR_ROCKET_BODY, 
                'size': (ROCKET_W, ROCKET_H),
                'angle': self.theta + DISPLAY_ROTATION,
                'center': (self.x, self.y)
            })

        elif self.game_state == 'EXPLODED':
            # explosion
            shapes.append({'type': 'circle', 'color': COLOR_EXPLOSION_OUTER, 'position': (int(self.x), int(self.y)), 'radius': 100})
            shapes.append({'type': 'circle', 'color': COLOR_EXPLOSION_INNER, 'position': (int(self.x), int(self.y)), 'radius': 80})
            # crash text
            crash_line = 'you crashed'
            shapes.append({
                'type': 'text',
                'content': crash_line,
                'color': COLOR_TEXT,
                'center': (300, 400),
                'shadow_offset': (2, 2)
            })
            
        return shapes


# info class
class Info:
    def __init__(self, gen_id=None, fitness=None):
        self.gen_id = gen_id
        self.fitness = fitness

    def get_shapes(self, rocket):        
        is_simulation = self.gen_id is not None and self.fitness is not None

        # top left info
        info_text = ''
        if is_simulation:
            info_text += (
                f'generation: {self.gen_id}\n'
                f'fitness: {self.fitness:.1f}\n\n'
            )
            message = 'press SPACE to skip replay'
        else:
            message = 'press SPACE to skip restart'
        stats = (
            f'speed: {rocket.speed:.1f}\n'
            f'angle: {math.degrees(rocket.theta + DISPLAY_ROTATION):.1f}\n'
            f'wind: {config.WIND_SPEED:.1f}\n\n'
        )   
        info_text += stats + message

        text_lines = info_text.split('\n')

        LINE_HEIGHT = 20
        start_x, start_y = 10, 10
        shapes = []
        for index, line in enumerate(text_lines):
            line_y = start_y + index * LINE_HEIGHT
            shapes.append({
                'type': 'text', 
                'content': line, 
                'color': COLOR_TEXT, 
                'position': (start_x, line_y),
                'shadow_offset': (1, 1)
            })
        return shapes


# background class
class Background:
    def get_shapes(self):
        return {'type': 'fill', 'color': COLOR_BACKGROUND}
    

# platform class
class Platform:
    def get_shapes(self):
        return {'type': 'rect', 'color': COLOR_PLATFORM, 'rect': PLATFORM_RECT}


# ground class
class Ground:
    def get_shapes(self):
        return {'type': 'rect', 'color': COLOR_GROUND, 'rect': GROUND_RECT}


