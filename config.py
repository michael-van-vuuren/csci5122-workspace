import math
import pygame

# screen
SCREEN_W, SCREEN_H = 600, 850

# colors 
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

# conditions
WIND_SPEED = 0.0

# training
RANDOM_X_SPAWN = False
RANDOM_Y_SPAWN = False
NUM_EPISODES_PER_GENOME = 3

# other
DISPLAY_ROTATION = math.pi / 2
MAX_TIME = 1000
