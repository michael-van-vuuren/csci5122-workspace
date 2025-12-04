import pygame
import argparse
import math
from dataclasses import dataclass

import neat
import neat_visualizer

import game_objects

pygame.init()

SCREEN_W, SCREEN_H = 600, 850
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
clock = pygame.time.Clock()

FONT = pygame.font.SysFont(None, 24)
BIG_FONT = pygame.font.Font(None, 74)


# holds all the game objects 
@dataclass
class GameObjects:
    rocket: game_objects.Rocket
    clouds: list[game_objects.Cloud]
    cow: game_objects.Cow
    info: game_objects.Info
    background: game_objects.Background
    ground: game_objects.Ground
    platform: game_objects.Platform


# instantiates the game objects
def create_objects(gen_id=None, fitness=None):
    rocket = game_objects.Rocket(SCREEN_W)
    clouds = [game_objects.Cloud(SCREEN_W) for _ in range(7)]
    cow = game_objects.Cow()
    info = game_objects.Info(gen_id=gen_id, fitness=fitness)
    background = game_objects.Background()
    ground = game_objects.Ground()
    platform = game_objects.Platform()

    return GameObjects(
        rocket=rocket,
        clouds=clouds,
        cow=cow,
        info=info,
        background=background,
        ground=ground,
        platform=platform
    )


def draw(renderer, objects):
    shapes = []
    
    # background, ground, platform, and cow
    shapes.append(objects.background.get_shapes())
    shapes.append(objects.ground.get_shapes())
    shapes.append(objects.platform.get_shapes())
    shapes.extend(objects.cow.get_shapes())
    
    # clouds 
    for cloud in objects.clouds:
        shapes.extend(cloud.get_shapes())
    
    # rocket (with flame and crash message)
    rocket_shapes = objects.rocket.get_shapes()
    for shape in rocket_shapes:
        if shape['type'] == 'text':
             renderer.font = BIG_FONT
    shapes.extend(rocket_shapes)

    renderer.draw(shapes)

    # top left info
    renderer.font = FONT
    info_shapes = objects.info.get_shapes(objects.rocket)

    renderer.draw(info_shapes)

    pygame.display.flip()




# fitness function (kinda like a reward function in reinforcement learning)
def fitness_function(rocket):
    # calculate the euclidian distance from rocket to platform
    center_x = game_objects.PLAT_X + game_objects.PLAT_W / 2
    dtp_x = abs(rocket.x - center_x)
    dtp_y = abs(rocket.y - game_objects.PLAT_Y)
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
    if rocket.time_taken >= game_objects.MAX_TIME and rocket.game_state == 'RUNNING':       
        fitness -= 200                                       # hovering in the air            = decrease

    return fitness

# runs a replay of a simulation on a genome
def run_simulation(genome, config, draw_mode=False, gen_id=None):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    objects = create_objects(gen_id=gen_id, fitness=genome.fitness)
    rocket = objects.rocket
    clouds = objects.clouds
    
    # used to draw each update
    renderer = game_objects.Renderer(screen)

    dt = 1 / 60.0

    for _ in range(game_objects.MAX_TIME):
        if draw_mode:
            # 1. quit or skip replay
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    return fitness_function(rocket)
        
        # 2. get current network inputs, do forward pass on genome network,
        # save outputs as action probabilities, then take a step using the actions 
        inputs = rocket.get_inputs()
        action = net.activate(inputs)
        rocket.step(action, dt, mode='nn')

        # 3. game rendering
        if draw_mode:
            for cloud in clouds:
                cloud.update(dt)
            
            draw(renderer, objects)
            clock.tick(60)

        # rocket has crashed or landed
        if rocket.game_state != 'RUNNING':
            break

    # return fitness
    return fitness_function(rocket)



# called automatically by NEAT for each generation
def eval_genomes(genomes, config):
    for id, genome in genomes:
        if game_objects.RANDOM_X_SPAWN or game_objects.RANDOM_Y_SPAWN:
            fitness_sum = 0.0
            for _ in range(game_objects.NUM_EPISODES_PER_GENOME):
                fitness_sum += run_simulation(genome, config, draw_mode=False)
            genome.fitness = fitness_sum / game_objects.NUM_EPISODES_PER_GENOME
        else:
            genome.fitness = run_simulation(genome, config, draw_mode=False)


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
    skip_until = 0
    N = 10
    
    for i in range(N):
        print(f'Generation {i}')
        
        population.run(eval_genomes, 1) 
        
        best_genome = population.best_genome
        print(f'Best fitness: {best_genome.fitness}')
        if i < skip_until:
            print('Skipping replay')
        else:
            print('Replaying best genome')
            run_simulation(best_genome, config, draw_mode=True, gen_id=i)   

    visualizer.filename = 'rocket_evolution.gif'
    visualizer.window.save_frames(visualizer.filename, visualizer.directory)


def run_player():
    objects = create_objects()
    rocket = objects.rocket
    clouds = objects.clouds

    renderer = game_objects.Renderer(screen)

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        # quit or reset
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                rocket.reset()

        # game control
        keys = pygame.key.get_pressed()
        rocket.step(keys, dt, mode='player')
        for cloud in clouds:
            cloud.update(dt)

        draw(renderer, objects) 


# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wind', type=float, default=0.0, help='speed of wind')
    parser.add_argument('--train', action='store_true', help='run NEAT training')
    args = parser.parse_args()

    game_objects.WIND_SPEED = args.wind    

    if args.train:
        # nn controlled version
        run_neat('config-feedforward-rocket.ini')
    else:
        # player controlled version
        run_player()

    pygame.quit()

if __name__ == '__main__':
    main()
