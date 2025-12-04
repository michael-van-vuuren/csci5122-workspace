import argparse
import math
import pygame

import neat
import neat_visualizer

import objects
import renderer
import config
from config import *


pygame.init()
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
clock = pygame.time.Clock()

# used to draw each update
r = renderer.Renderer(screen)


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


def replay(genome, config, gen_id):
    def on_step(objects):
        r.draw(objects)
        clock.tick(60)

    def should_skip():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                return True
        return False
    
    run_simulation(
        genome,
        config,
        gen_id=gen_id,
        on_step=on_step,
        should_skip=should_skip
    )


# runs a replay of a simulation on a genome
def run_simulation(genome, config, gen_id=None, on_step=None, should_skip=None):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    objs = objects.create_objects(gen_id=gen_id, fitness=genome.fitness)
    rocket = objs.rocket
    clouds = objs.clouds

    dt = 1 / 60.0

    for _ in range(MAX_TIME):
        if should_skip and should_skip():
            break

        # game rendering
        for cloud in clouds:
            cloud.update(dt)
        
        # get current network inputs, do forward pass on genome network,
        # save outputs as action probabilities, then take a step using the actions 
        inputs = rocket.get_inputs()
        action = net.activate(inputs)
        rocket.step(action, dt, mode='nn')
        
        if on_step:
            on_step(objs)

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
                fitness_sum += run_simulation(genome, config)
            genome.fitness = fitness_sum / NUM_EPISODES_PER_GENOME
        else:
            genome.fitness = run_simulation(genome, config)


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
            replay(best_genome, config, gen_id=i)   

    visualizer.filename = 'rocket_evolution.gif'
    visualizer.window.save_frames(visualizer.filename, visualizer.directory)


def run_player():
    objs = objects.create_objects()
    rocket = objs.rocket
    clouds = objs.clouds

    running = True
    while running:
        dt = clock.tick(60) / 1000.0

        # game rendering
        for cloud in clouds:
            cloud.update(dt)

        # quit or reset
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                rocket.reset()

        # game control
        keys = pygame.key.get_pressed()
        rocket.step(keys, dt, mode='player')

        r.draw(objs) 


# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wind', type=float, default=0.0, help='speed of wind')
    parser.add_argument('--train', action='store_true', help='run NEAT training')
    args = parser.parse_args()

    config.WIND_SPEED = args.wind    

    if args.train:
        # nn controlled version
        run_neat('config-feedforward-rocket.ini')
    else:
        # player controlled version
        run_player()

    pygame.quit()

if __name__ == '__main__':
    main()
