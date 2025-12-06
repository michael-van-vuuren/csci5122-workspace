import neat
import neat_visualizer

import simulation
from simulation import Simulation
from config import *

simulation = Simulation()

# called automatically by NEAT for each generation
def eval_genomes(genomes, config):
    for id, genome in genomes:
        if RANDOM_X_SPAWN or RANDOM_Y_SPAWN:
            fitness_sum = 0.0
            for _ in range(NUM_EPISODES_PER_GENOME):
                fitness_sum += simulation.run_simulation(genome, config)
            genome.fitness = fitness_sum / NUM_EPISODES_PER_GENOME
        else:
            genome.fitness = simulation.run_simulation(genome, config)


# run the NEAT algorithm
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
    skip_until = 90
    N = 100
    
    for i in range(N):
        print(f'Generation {i}')
        
        population.run(eval_genomes, 1) 
        
        best_genome = population.best_genome
        print(f'Best fitness: {best_genome.fitness}')
        if i < skip_until:
            print('Skipping replay')
        else:
            print('Replaying best genome')
            simulation.replay(best_genome, config, gen_id=i)   

    visualizer.filename = 'rocket_evolution.gif'
    visualizer.species_filename = 'rocket_species_fitness.png'
    visualizer.save_species_graph(visualizer.species_filename)
    visualizer.window.save_frames(visualizer.filename, visualizer.directory)
