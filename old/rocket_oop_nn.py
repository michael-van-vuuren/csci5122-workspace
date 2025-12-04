import pygame
import random
import math
import neat
import neat_visualizer

pygame.init()
SCREEN_W, SCREEN_H = 600, 850
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# color
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

# conditions
WIND_SPEED = 0
MAX_TIME = 1000
RANDOM_X_SPAWN = False
RANDOM_Y_SPAWN = True
NUM_EPISODES_PER_GENOME = 3

# cow class
def draw_cow():
    cow_x = 480
    cow_y = GROUND_Y - 50
    BODY = (255, 255, 255)
    SPOT = COLOR_PLATFORM
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    PINK = (255, 180, 200)
    body_rect = pygame.Rect(cow_x, cow_y, 80, 40)
    pygame.draw.ellipse(screen, BODY, body_rect)
    pygame.draw.circle(screen, SPOT, (cow_x + 20, cow_y + 20), 10)
    pygame.draw.circle(screen, SPOT, (cow_x + 50, cow_y + 15), 12)
    head_rect = pygame.Rect(cow_x - 30, cow_y + 5, 35, 28)
    pygame.draw.ellipse(screen, BODY, head_rect)
    snout_rect = pygame.Rect(cow_x - 30, cow_y + 18, 30, 18)
    pygame.draw.ellipse(screen, PINK, snout_rect)
    ear_rect1 = pygame.Rect(cow_x - 22, cow_y - 3, 6, 9)
    ear_rect2 = pygame.Rect(cow_x - 8, cow_y - 3, 6, 9)
    pygame.draw.ellipse(screen, BLACK, ear_rect1)
    pygame.draw.ellipse(screen, BLACK, ear_rect2)
    ear_rect1 = pygame.Rect(cow_x - 28, cow_y + 2, 10, 6)
    ear_rect2 = pygame.Rect(cow_x - 6, cow_y + 2, 10, 6)
    pygame.draw.ellipse(screen, WHITE, ear_rect1)
    pygame.draw.ellipse(screen, WHITE, ear_rect2)
    eye_rect1 = pygame.Rect(cow_x - 22, cow_y + 10, 6, 10)
    eye_rect2 = pygame.Rect(cow_x - 10, cow_y + 10, 6, 10)
    pygame.draw.ellipse(screen, BLACK, eye_rect1)
    pygame.draw.ellipse(screen, BLACK, eye_rect2)
    pygame.draw.circle(screen, WHITE, (cow_x - 20, cow_y + 13), 1)
    pygame.draw.circle(screen, WHITE, (cow_x - 8,  cow_y + 13), 1)
    pygame.draw.circle(screen, SPOT, (cow_x - 17, cow_y + 25), 2)
    pygame.draw.circle(screen, SPOT, (cow_x - 11,  cow_y + 25), 2)
    for lx in [16, 24, 58, 66]:
        pygame.draw.rect(screen, BODY, (cow_x + lx, cow_y + 30, 6, 20))
        pygame.draw.rect(screen, BLACK, (cow_x + lx, cow_y + 50, 6, 5))
    pygame.draw.line(screen, BLACK, (cow_x + 75, cow_y + 10), (cow_x + 85, cow_y + 25), 3)
    tail_rect = pygame.Rect(cow_x + 85, cow_y + 25, 8, 4)
    pygame.draw.ellipse(screen, BLACK, tail_rect)

# cloud class
class Cloud:
    def __init__(self):
        self.x = random.randint(0, 600)
        self.y = random.randint(10, 220)
        self.wind_offset = random.uniform(-3, 3)
        self.circles = self.generate_circles()

    def reset(self, left_side=True):
        self.x = random.randint(-200, -100) if left_side else random.randint(SCREEN_W + 100, SCREEN_W + 200)
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

                cx = gx * cell_w
                cy = gy * cell_h

                cx += random.randint(-pos_jitter, pos_jitter)
                cy += random.randint(-pos_jitter, pos_jitter)

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
        self.x += (WIND_SPEED + self.wind_offset) * dt * 3
        # wind going right/left
        if WIND_SPEED >= 0:
            if self.x > SCREEN_W + 150:
                self.reset(left_side=True)
        else:
            if self.x < -150:
                self.reset(left_side=False)

    def draw(self):
        for ox, oy, r, a in self.circles:
            cloud_circle = pygame.Surface((r*2, r*2), pygame.SRCALPHA)
            pygame.draw.circle(cloud_circle, (255, 255, 255, a), (r, r), r)
            screen.blit(cloud_circle, (self.x + ox, self.y + oy))

clouds = [Cloud() for _ in range(7)]

# functions for drawing
def draw_background():
    screen.fill(COLOR_BACKGROUND)
    for cloud in clouds:
        cloud.draw()
    pygame.draw.rect(screen, COLOR_GROUND, GROUND_RECT)
    pygame.draw.rect(screen, COLOR_PLATFORM, PLATFORM_RECT)
    draw_cow()

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

# rocket class
class Rocket:
    def __init__(self):
        self.reset()

    def reset(self):
        if RANDOM_X_SPAWN:
            self.x = random.randint(200, SCREEN_W - 200)
        else:
            self.x = 500                                # initial x position
        if RANDOM_Y_SPAWN:
            self.y = random.randint(100, 300)
        else:
            self.y = 100                                # initial y position
        self.vx = -60                                   # initial x velocity
        self.vy = 60                                    # initial y velocity
        self.theta_offset = -0.4                        # initial offset of angle
        self.theta = -math.pi/2 + self.theta_offset     # initial angle 
        self.omega = 0                                  # initial rate of change in angle over time
        self.speed = math.sqrt(self.vx**2 + self.vy**2) # total velocity magnitude
        self.is_tipping = False
        self.tipping_direction = 0
        self.game_state = 'RUNNING' 

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
            self.x / SCREEN_W,      # -1 (x-pos)
            self.y / SCREEN_H,      # -2 (y-pos)
            self.vx / 100.0,        # -3 (x-velocity)
            self.vy / 100.0,        # -4 (y-velocity)
            self.theta,             # -5 (angle)
            self.omega,             # -6 (angular velocity)
            WIND_SPEED / 20.0       # -7 (wind sensor)
        ]

    # like the update function in the user controlled one, 
    # but now it gets an action vector made by the network
    # instead of key presses by the user
    def step(self, action, dt):
        threshold = self.action_threshold
        # action: [thrust_probability, left_probability, right_probability]
        thrust = action[0] > threshold
        left   = action[1] > threshold
        right  = action[2] > threshold

        # clouds
        for cl in clouds:
            cl.update(dt)

        # rocket on the pad
        if self.game_state == 'LANDED':
            if not self.is_tipping:
                deviation = self.theta - (-math.pi/2)
                if abs(deviation) > MAX_STABLE_ANGLE:
                    self.is_tipping = True
                    self.tipping_direction = 1 if deviation > 0 else -1
            if self.is_tipping:
                self.omega += self.tipping_direction * TIPPING_ACCELERATION * dt
                self.theta += self.omega * dt
                if abs(self.theta - (-math.pi/2)) > math.pi/2:
                    self.game_state = 'EXPLODED'
        
        # rocket in the air
        if self.game_state == 'RUNNING':
            ax = 0
            ay = G

            # effect of wind
            ax += WIND_SPEED / MASS
            
            # thrust
            if thrust:
                ax += THRUST_POWER * math.cos(self.theta) / MASS
                ay += THRUST_POWER * math.sin(self.theta) / MASS
                self.fuel_used += dt
            
            # angle
            alpha = 0
            if left: alpha -= TORQUE_POWER / I
            if right: alpha += TORQUE_POWER / I
            
            # update velocity, position, and angle
            self.vx += ax * dt
            self.vy += ay * dt
            self.x += self.vx * dt
            self.y += self.vy * dt
            self.omega += alpha * dt
            self.theta += self.omega * dt
            self.speed = math.sqrt(self.vx**2 + self.vy**2)
            
            # rocket collision with screen walls
            if self.x < 0: self.x = 0; self.vx = 0
            if self.x > SCREEN_W: self.x = SCREEN_W; self.vx = 0
            if self.y < 0: self.y = 0; self.vy = 0

            # redraw rocket
            rocket_rect = pygame.Rect(0,0,ROCKET_W,ROCKET_H)
            rocket_rect.center = (self.x, self.y)
            
            # rocket collision with ground/platform
            if rocket_rect.colliderect(GROUND_RECT):
                self.game_state = 'EXPLODED'
            elif rocket_rect.colliderect(PLATFORM_RECT):
                # check landing angle and speed
                angle_too_sharp = abs(self.theta - (-math.pi/2)) >= 0.2
                speed_too_fast = self.speed >= EXPLOSION_VELOCITY_LIMIT
                if speed_too_fast or angle_too_sharp:
                    self.game_state = 'EXPLODED'
                else:
                    self.game_state = 'LANDED'
                    self.vx = self.vy = 0
                    self.y = PLAT_Y - ROCKET_H // 2

        self.time_taken += 1

    def draw(self, action=None):
        if self.game_state in ['RUNNING', 'LANDED']:
            rocket_surface = pygame.Surface((ROCKET_W, ROCKET_H), pygame.SRCALPHA)
            pygame.draw.rect(rocket_surface, COLOR_ROCKET_BODY, (0, 0, ROCKET_W, ROCKET_H))

            # draw flame attached to bottom of rocket
            if action is not None and action[0] > self.action_threshold:
                draw_flame(self.x, self.y, self.theta)

            rotated = pygame.transform.rotate(rocket_surface,-math.degrees(self.theta + DISPLAY_ROTATION))
            rect = rotated.get_rect(center=(self.x, self.y))
            screen.blit(rotated, rect)
        elif self.game_state == 'EXPLODED':
            draw_explosion(self.x, self.y)

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

# runs a replay of a simulation on a genome
def run_simulation(genome, config, draw=False, i=None):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    rocket = Rocket()
    
    dt = 1 / 60.0
    skip_replay = False
    for _ in range(MAX_TIME):
        if draw:
            # 1. quit or skip replay
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    skip_replay = True

            if skip_replay:
                break
            draw_background()
        
        # 2. get current network inputs, do forward pass on genome network,
        # save outputs as action probabilities, then take a step using the actions 
        inputs = rocket.get_inputs()
        action = net.activate(inputs)
        rocket.step(action, dt)
        
        # 3. game rendering
        if draw:
            rocket.draw(action)

            # top left info
            info_text = f'generation: {i}\nfitness: {genome.fitness:.1f}\n\nspeed: {rocket.speed:.1f}\nangle: {math.degrees(rocket.theta + DISPLAY_ROTATION):.1f}\nwind: {WIND_SPEED:.1f}\n\npress SPACE to skip replay'

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
            clock.tick(60)

        # rocket has crashed or landed
        if rocket.game_state != 'RUNNING':
            break

    # return fitness
    return fitness_function(rocket)

# called automatically by NEAT for each generation
def eval_genomes(genomes, config):
    for id, genome in genomes:
        if RANDOM_X_SPAWN or RANDOM_Y_SPAWN:
            fitness_sum = 0.0
            for _ in range(NUM_EPISODES_PER_GENOME):
                fitness_sum += run_simulation(genome, config, draw=False)
            genome.fitness = fitness_sum / NUM_EPISODES_PER_GENOME
        else:
            genome.fitness = run_simulation(genome, config, draw=False)


def run_neat(config_file):
    config = neat.Config(
        neat_visualizer.TrackedGenome, 
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet, 
        neat.DefaultStagnation,
        config_file
    )

    # initialize genomes
    population = neat.Population(config)

    # initialize info reporter
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    visualizer = neat_visualizer.Visualizer(config)
    population.add_reporter(visualizer)

    # you can skip the first n_skip generations (N total)
    skip_until = 350
    N = 400
    
    for i in range(N):
        print(f'Generation {i}')
        
        population.run(eval_genomes, 1) 
        
        best_genome = population.best_genome
        print(f'Best fitness: {best_genome.fitness}')
        if i < skip_until:
            print('Skipping replay')
        else:
            print('Replaying best genome')
            run_simulation(best_genome, config, draw=True, i=i)   

    visualizer.filename = 'rocket_evolution.gif'
    visualizer.window.save_frames(visualizer.filename, visualizer.directory)
         
run_neat('config-feedforward-rocket.ini')
