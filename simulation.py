import pygame
import neat

from fitness import fitness_function
from objects import create_objects
from renderer import Renderer
from config import *

pygame.init()
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
renderer = Renderer(screen)
clock = pygame.time.Clock()

def run_simulation(genome, config, gen_id=None, draw=False):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    objects = create_objects(gen_id=gen_id, fitness=genome.fitness)
    rocket = objects.rocket
    clouds = objects.clouds
        
    for _ in range(MAX_TIME):
        dt = clock.tick(60)/1000.0 if draw else 1/60.0

        # update clouds
        for cloud in clouds:
            cloud.update(dt)
        
        # get current network inputs, do forward pass on genome network,
        # save outputs as action probabilities, then take a step using the actions 
        inputs = rocket.get_inputs()
        action = net.activate(inputs)
        rocket.step(action, dt, mode='nn')

        if draw:
            renderer.draw(objects)
            # quit or skip
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return fitness_function(rocket)
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    return fitness_function(rocket)

        # rocket has crashed or landed
        if rocket.game_state != 'RUNNING':
            break

    return fitness_function(rocket)


def replay(genome, config, gen_id):
    run_simulation(genome, config, gen_id=gen_id, draw=True)
