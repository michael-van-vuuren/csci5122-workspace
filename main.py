import argparse
import pygame

import neat
import neat_visualizer

from objects import create_objects
from renderer import Renderer
from fitness import fitness_function
import config
from config import *


pygame.init()
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
clock = pygame.time.Clock()

# used to draw each update
renderer = Renderer(screen)


def replay(genome, config, gen_id):
    def on_step(objects):
        renderer.draw(objects)
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

    objects = create_objects(gen_id=gen_id, fitness=genome.fitness)
    rocket = objects.rocket
    clouds = objects.clouds

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
            on_step(objects)

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
    objects = create_objects()
    rocket = objects.rocket
    clouds = objects.clouds

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

        renderer.draw(objects) 


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
