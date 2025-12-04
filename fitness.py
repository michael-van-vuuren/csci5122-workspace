import math
from config import *


# fitness function (kinda like a reward function in reinforcement learning)
def fitness_function(rocket):
    # calculate the euclidian distance from rocket to platform
    center_x = PLAT_X + PLAT_W / 2
    dtp_x = abs(rocket.x - center_x)
    dtp_y = abs(rocket.y - PLAT_Y)
    dtp = math.sqrt(dtp_x**2 + dtp_y**2)

    # calculate upright deviation angle
    angle_deviation = abs(rocket.theta - (-math.pi/2)) 
    
    fitness = -100.0                                         # 1. running rewards               fitness
    fitness += (1000 - dtp) / 10.0                           # closer distance to platform    = increase
    fitness += (3.0 - angle_deviation) * 10                  # upright angle                  = increase

    fuel_penalty = rocket.fuel_used * 0.5                    # more fuel use                  = decrease
    fitness -= fuel_penalty

    wall_threshold = 50                                      # use screen walls for stability = decrease
    if rocket.x < wall_threshold:
        fitness -= (wall_threshold - rocket.x) * 2
    elif rocket.x > SCREEN_W - wall_threshold:
        fitness -= (rocket.x - (SCREEN_W - wall_threshold)) * 2

    if rocket.game_state == 'EXPLODED':                      # 2. end state rewards
        fitness -= 50                                        # crashed                        = decrease
        fitness += (SCREEN_W/2 - dtp_x) * 0.4                # crashed closer to center       = increase
        fitness += max(0, 200 - rocket.speed) * 0.5          # softer crash                   = increase
    elif rocket.game_state == 'LANDED':
        fitness += 200                                       # landed                         = increase
        fitness += (200 - dtp_x)                             # landed closer to center        = increase
        fitness += max(0, (20 - rocket.speed) * 3)           # softer landing                 = increase
        fitness -= angle_deviation * 150                     # less upright landing           = decrease

        time_bonus = max(0, 500 - rocket.time_taken) * 0.5
        fitness += time_bonus
    if rocket.time_taken >= MAX_TIME and rocket.game_state == 'RUNNING':       
        fitness -= 200                                       # hovering in the air            = decrease

    return fitness
