import pygame
import math
import random

pygame.init()

screen = pygame.display.set_mode((600, 800))
clock = pygame.time.Clock()

COLOR_BACKGROUND = (21, 129, 191)
COLOR_PLATFORM = (33, 33, 33)
COLOR_GROUND = (61, 182, 177)

COLOR_ROCKET_BODY = (230, 242, 255)
COLOR_FLAME_CORE = (255, 255, 100)
COLOR_FLAME_EDGE = (255, 50, 0)

COLOR_EXPLOSION_OUTER = (255, 165, 0)
COLOR_EXPLOSION_INNER = (255, 0, 0)

COLOR_TEXT = (255, 255, 255)

def reset_rocket():
    global x, y, vx, vy, theta_offset, theta, omega, game_state, is_tipping, tipping_direction
    x = 500                             # initial x position
    y = 100                             # initial y position
    vx = -40                            # initial x velocity
    vy = 20                             # initial y velocity
    theta_offset = 0.2                  # initial offset of angle
    theta = -math.pi/2 + theta_offset   # initial angle 
    omega = 0                           # initial rate of change in angle over time     
    is_tipping = False
    tipping_direction = 0
    game_state = "RUNNING" 

reset_rocket()

# rocket physics
MASS = 1.0              # mass of the rocket (resistance to thrust)
I = 10.0                # inertia of the rocket (resistance to torque)
G = 200                 # gravity constant
THRUST_POWER = 300      # power of up arrow
TORQUE_POWER = 10.0     # power of left and right arrows

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
GROUND_H = PLAT_H
GROUND_Y = 800 - GROUND_H
GROUND_RECT = pygame.Rect(0, GROUND_Y, 600, GROUND_H)

EXPLOSION_VELOCITY_LIMIT = 200.0
DISPLAY_ROTATION = math.pi / 2

# functions for drawing
def draw_background():
    screen.fill(COLOR_BACKGROUND)
    pygame.draw.rect(screen, COLOR_GROUND, GROUND_RECT)
    pygame.draw.rect(screen, COLOR_PLATFORM, PLATFORM_RECT)

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

def draw_rocket(x, y, theta, keys, game_state):
    rocket_surface = pygame.Surface((ROCKET_W, ROCKET_H), pygame.SRCALPHA)
    pygame.draw.rect(rocket_surface, COLOR_ROCKET_BODY, (0, 0, ROCKET_W, ROCKET_H))

    if keys[pygame.K_UP] and game_state == "RUNNING":
        draw_flame(x, y, theta)

    rotated = pygame.transform.rotate(rocket_surface, -math.degrees(theta + DISPLAY_ROTATION))
    rect = rotated.get_rect(center=(x, y))
    screen.blit(rotated, rect)

def draw_explosion(x, y):
    pygame.draw.circle(screen, COLOR_EXPLOSION_OUTER, (int(x), int(y)), 100)
    pygame.draw.circle(screen, COLOR_EXPLOSION_INNER, (int(x), int(y)), 80)

    # draw crash text
    font = pygame.font.Font(None, 74)
    text = font.render("CRASHED!", True, COLOR_TEXT)
    screen.blit(text, text.get_rect(center=(300,400)))
    restart_font = pygame.font.Font(None,36)
    restart_text = restart_font.render("Press SPACE to retry", True, COLOR_TEXT)
    screen.blit(restart_text, restart_text.get_rect(center=(300,450)))

# main
running = True
while running:
    dt = clock.tick(60) / 1000.0

    # 1. quit or reset
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            reset_rocket()

    # 2. game physics and control
    keys = pygame.key.get_pressed()

    # rocket on the pad
    if game_state == "LANDED":
        if not is_tipping:
            current_deviation = theta - (-math.pi/2)
            if abs(current_deviation) > MAX_STABLE_ANGLE:
                is_tipping = True
                tipping_direction = 1 if current_deviation > 0 else -1
        if is_tipping:            
            omega += tipping_direction * TIPPING_ACCELERATION * dt
            theta += omega * dt                        
            if abs(theta - (-math.pi/2)) > math.pi / 2: 
                game_state = "EXPLODED"   

    # rocket in the air         
    if game_state == "RUNNING":
        ax = 0
        ay = G

        # thrust
        if keys[pygame.K_UP]:
            ax += THRUST_POWER * math.cos(theta) / MASS
            ay += THRUST_POWER * math.sin(theta) / MASS

        # angle
        alpha = 0
        if keys[pygame.K_LEFT]:
            alpha -= TORQUE_POWER / I
        if keys[pygame.K_RIGHT]:
            alpha += TORQUE_POWER / I

        # update velocity and position
        vx += ax * dt
        vy += ay * dt
        x += vx * dt
        y += vy * dt

        # update angle
        omega += alpha * dt
        theta += omega * dt

        # rocket collision with ground/platform 
        rocket_rect = pygame.Rect(0, 0, ROCKET_W, ROCKET_H)
        rocket_rect.center = (x, y)

        if rocket_rect.colliderect(GROUND_RECT):
            game_state = "EXPLODED"         
        elif rocket_rect.colliderect(PLATFORM_RECT):
            if math.sqrt(vx**2 + vy**2) > EXPLOSION_VELOCITY_LIMIT:
                game_state = "EXPLODED"
            else:
                game_state = "LANDED"
                vy = vx = 0 
                y = PLAT_Y - ROCKET_H // 2

        # rocket collision with screen walls
        if x < 0:   x = 0;   vx = 0
        if x > 600: x = 600; vx = 0
        if y < 0:   y = 0;   vy = 0
    
    # 3. game rendering
    draw_background()
    if game_state in ["RUNNING", "LANDED"]:
        draw_rocket(x, y, theta, keys, game_state)
    elif game_state == "EXPLODED":
        draw_explosion(x, y)

    pygame.display.flip()

pygame.quit()
