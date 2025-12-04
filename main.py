import argparse

from player_runner import run_player
from neat_runner import run_neat
import config

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wind', type=float, default=0.0, help='speed of wind')
    parser.add_argument('--train', action='store_true', help='run NEAT training')
    args = parser.parse_args()

    config.WIND_SPEED = args.wind    

    if args.train:
        run_neat('config-feedforward-rocket.ini') # nn controlled version
    else:
        run_player() # player controlled version

if __name__ == '__main__':
    main()
