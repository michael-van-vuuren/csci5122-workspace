import pygame
import math
import random
import argparse
import scenery

pygame.init()
SCREEN_W, SCREEN_H = 600, 850
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

COLOR_BACKGROUND = (21, 129, 191)
COLOR_PLATFORM = (33, 33, 33)
COLOR_GROUND = (61, 182, 177)

COLOR_ROCKET_BODY = (230, 242, 255)
COLOR_FLAME_CORE = (255, 255, 100)
COLOR_FLAME_EDGE = (255, 50, 0)

COLOR_EXPLOSION_OUTER = (255, 165, 0)
COLOR_EXPLOSION_INNER = (255, 0, 0)

COLOR_TEXT = (255, 255, 255)

# rocket physics
MASS = 1.0                          # mass of the rocket (resistance to thrust)
I = 10.0                            # inertia of the rocket (resistance to torque)
G = 200                             # gravity constant
THRUST_POWER = 400                  # power of up arrow
TORQUE_POWER = 10.0                 # power of left and right arrows
EXPLOSION_VELOCITY_LIMIT = 100.0    # max speed the rocket can land on the platform with

# rocket geometry
ROCKET_W = 20
ROCKET_H = 120
CENTER_OF_MASS_OFFSET = ROCKET_H * 0.8
MAX_STABLE_ANGLE = math.atan(ROCKET_W / (2 * CENTER_OF_MASS_OFFSET)) * 0.8
TIPPING_ACCELERATION = 30.0

# platform
PLAT_W = 200
PLAT_H = 20
PLAT_X = 300 - PLAT_W // 2
PLAT_Y = 760
PLATFORM_RECT = pygame.Rect(PLAT_X, PLAT_Y, PLAT_W, PLAT_H)

# ground
GROUND_H = PLAT_H + 50
GROUND_Y = 850 - GROUND_H
GROUND_RECT = pygame.Rect(0, GROUND_Y, 600, GROUND_H)

DISPLAY_ROTATION = math.pi / 2


parser = argparse.ArgumentParser()
parser.add_argument('--wind', type=float, default=0.0, help='speed of wind')
args = parser.parse_args()

# conditions
WIND_SPEED = args.wind

clouds = [scenery.Cloud(random.randint(0, SCREEN_W), random.randint(10, 220), SCREEN_W) for _ in range(7)]

# functions for drawing
def draw_background():
    screen.fill(COLOR_BACKGROUND)
    for cloud in clouds:
        cloud.draw(screen)
    pygame.draw.rect(screen, COLOR_GROUND, GROUND_RECT)
    pygame.draw.rect(screen, COLOR_PLATFORM, PLATFORM_RECT)
    scenery.draw_cow(screen, GROUND_Y)

def draw_flame(x, y, theta):
    flame_angle = theta + math.pi
    nozzle_offset_distance = ROCKET_H / 2
    nozzle_x = x + nozzle_offset_distance * math.cos(flame_angle)
    nozzle_y = y + nozzle_offset_distance * math.sin(flame_angle)

    # strobing flame animation
    for _ in range(3):
        flame_length = random.randint(10, 40)
        flame_width = random.randint(ROCKET_W - 10, ROCKET_W)
        tip_x = nozzle_x + flame_length * math.cos(flame_angle)
        tip_y = nozzle_y + flame_length * math.sin(flame_angle)
        perp_x = (flame_width/2) * math.cos(flame_angle + math.pi/2)
        perp_y = (flame_width/2) * math.sin(flame_angle + math.pi/2)
        flame_points = [(tip_x, tip_y),
                        (nozzle_x + perp_x, nozzle_y + perp_y),
                        (nozzle_x - perp_x, nozzle_y - perp_y)]
        if _ == 0:
            color = (COLOR_FLAME_EDGE[0], random.randint(COLOR_FLAME_EDGE[1], 165), COLOR_FLAME_EDGE[2])
        else:
            color = (COLOR_FLAME_CORE[0], COLOR_FLAME_CORE[1], random.randint(COLOR_FLAME_CORE[2], 150))
        pygame.draw.polygon(screen, color, flame_points)

def draw_explosion(x, y):
    pygame.draw.circle(screen, COLOR_EXPLOSION_OUTER, (int(x), int(y)), 100)
    pygame.draw.circle(screen, COLOR_EXPLOSION_INNER, (int(x), int(y)), 80)

    # draw crash text
    font = pygame.font.Font(None, 74)
    crash_line = 'you crashed'
    crash_shadow = font.render(crash_line, True, (0, 0, 0))
    crash_rect = crash_shadow.get_rect(center=(300, 400))
    shadow_offset = (2, 2)
    screen.blit(crash_shadow, (crash_rect.x + shadow_offset[0], crash_rect.y + shadow_offset[1]))
    crash_text = font.render(crash_line, True, COLOR_TEXT)
    crash_rect = crash_text.get_rect(center=(300, 400))
    screen.blit(crash_text, crash_rect)

# rocket class
class Rocket:
    def __init__(self):
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

    def update(self, keys, dt):
        # clouds
        for cloud in clouds:
            cloud.update(dt, WIND_SPEED)
            cloud.draw(screen)

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
            ax += WIND_SPEED / MASS

            # thrust
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                ax += THRUST_POWER * math.cos(self.theta) / MASS
                ay += THRUST_POWER * math.sin(self.theta) / MASS

            # angle
            alpha = 0
            if keys[pygame.K_LEFT] or keys[pygame.K_a]: alpha -= TORQUE_POWER / I
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]: alpha += TORQUE_POWER / I

            # update velocity, position, and update angle
            self.vx += ax * dt
            self.vy += ay * dt
            self.x += self.vx * dt
            self.y += self.vy * dt
            self.omega += alpha * dt
            self.theta += self.omega * dt
            self.speed = math.sqrt(self.vx**2 + self.vy**2)

            # redraw rocket
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
            if self.x < 0:        self.x = 0;        self.vx = 0
            if self.x > SCREEN_W: self.x = SCREEN_W; self.vx = 0
            if self.y < 0:        self.y = 0;        self.vy = 0

    def draw(self, keys):
        if self.game_state in ['RUNNING', 'LANDED']:
            rocket_surface = pygame.Surface((ROCKET_W, ROCKET_H), pygame.SRCALPHA)
            pygame.draw.rect(rocket_surface, COLOR_ROCKET_BODY, (0, 0, ROCKET_W, ROCKET_H))

            # draw flame attached to bottom of rocket
            if (keys[pygame.K_UP] or keys[pygame.K_w]) and self.game_state == 'RUNNING':
                draw_flame(self.x, self.y, self.theta)

            rotated = pygame.transform.rotate(rocket_surface, -math.degrees(self.theta + DISPLAY_ROTATION))
            rect = rotated.get_rect(center=(self.x, self.y))
            screen.blit(rotated, rect)
        elif self.game_state == 'EXPLODED':
            draw_explosion(self.x, self.y)

# main
rocket = Rocket()
running = True
while running:
    dt = clock.tick(60) / 1000.0

    # 1. quit or reset
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            rocket.reset()

    # 2. game physics and control
    keys = pygame.key.get_pressed()
    rocket.update(keys, dt)

    # 3. game rendering
    draw_background()
    rocket.draw(keys)

    # top left info
    info_text = f'speed: {rocket.speed:.1f}\nangle: {math.degrees(rocket.theta + DISPLAY_ROTATION):.1f}\nwind: {WIND_SPEED:.1f}\n\npress SPACE to restart'

    text_lines = info_text.split('\n')
    LINE_HEIGHT = 20
    start_x, start_y = 10, 10
    shadow_offset = (1, 1)
    for index, line in enumerate(text_lines):
        line_y = start_y + index * LINE_HEIGHT
        shadow = font.render(line, True, (0, 0, 0))
        screen.blit(shadow, (start_x + shadow_offset[0], line_y + shadow_offset[1]))
        text_surface = font.render(line, True, COLOR_TEXT)
        screen.blit(text_surface, (start_x, line_y))
    pygame.display.flip()

pygame.quit()
