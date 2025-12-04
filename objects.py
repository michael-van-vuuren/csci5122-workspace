from dataclasses import dataclass
import classes

SCREEN_W = 600

# holds all the game objects 
@dataclass
class GameObjects:
    rocket: classes.Rocket
    clouds: list[classes.Cloud]
    cow: classes.Cow
    info: classes.Info
    background: classes.Background
    ground: classes.Ground
    platform: classes.Platform


# instantiates the game objects
def create_objects(gen_id=None, fitness=None):
    rocket = classes.Rocket(SCREEN_W)
    clouds = [classes.Cloud(SCREEN_W) for _ in range(7)]
    cow = classes.Cow()
    info = classes.Info(gen_id=gen_id, fitness=fitness)
    background = classes.Background()
    ground = classes.Ground()
    platform = classes.Platform()

    return GameObjects(
        rocket=rocket,
        clouds=clouds,
        cow=cow,
        info=info,
        background=background,
        ground=ground,
        platform=platform
    )
