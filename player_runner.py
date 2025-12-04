import pygame

from objects import create_objects
from renderer import Renderer
from config import *


# run using player controls
def run_player():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    clock = pygame.time.Clock()
    renderer = Renderer(screen)

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
