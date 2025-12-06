import neat
import os
import math
import numpy as np
from collections import defaultdict
import configparser
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image, ImageDraw, ImageFont
import imageio

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# fonts
FONT_PATH = '/System/Library/Fonts/Supplemental/Menlo.ttc'
FONT              = ImageFont.truetype(FONT_PATH, 32, index=0)
TINY_FONT         = ImageFont.truetype(FONT_PATH, 24, index=0)
TINIEST_FONT      = ImageFont.truetype(FONT_PATH, 20, index=0)
TINIEST_FONT_BOLD = ImageFont.truetype(FONT_PATH, 20, index=1)

# colors
COLOR_INPUT    = (166, 128, 255) # purple
COLOR_HIDDEN   = (30,  143, 255) # blue
COLOR_OUTPUT   = (255, 105, 180) # pink
COLOR_NEW      = (252, 220, 101) # yellow
COLOR_NEW_EDGE = (240, 101, 214) # purple-pink
COLOR_POS_EDGE = (100, 200, 100) # green
COLOR_NEG_EDGE = (200, 100, 100) # red
COLOR_WHITE    = (255, 255, 255)
COLOR_BLACK    = (0, 0, 0)

# activation function abbreviations
ACTIVATION_ABBREV = {
    'sigmoid': 'σ',
    'relu':    'r'
}


class Canvas:
    # creates a blank canvas and drawing object
    def __init__(self, width, height, scale=1):
        self.width = width
        self.height = height
        self.scale = scale
        self.image = Image.new('RGB', (width, height), COLOR_BLACK)
        self.draw = ImageDraw.Draw(self.image)
        self.frame_index = 0
        self.frames = []

    # drawing methods
    def fill_rect(self, x, y, w, h, color):
        self.draw.rectangle([x, y, x + w, y + h], fill=color)
    def fill_circle(self, x, y, w, h, color):
        self.draw.ellipse([x, y, x + w, y + h], outline='black', fill=color, width=5)
    def draw_line(self, x1, y1, x2, y2, color, width):
        self.draw.line([x1, y1, x2, y2], fill=color, width=width)
    def draw_text(self, x, y, text, color, font=FONT):
        self.draw.text((x, y), str(text), fill=color, font=font)
    def draw_aligned_text(self, x, y, text, color, font=FONT, align='left'):
        text_width = font.getlength(str(text))
        if align == 'center':
            x_aligned = x - (text_width / 2)
        elif align == 'right':
            x_aligned = x - text_width
        else:
            x_aligned = x
        self.draw.text((x_aligned, y), str(text), fill=color, font=font)

    # output methods
    def save_frame(self):
        self.frames.append(self.image.copy())
        self.frame_index += 1
    def save_frames(self, filename, directory, duration=300):
        if self.frames:
            # save frames to directory, save gif of combined frames to filename
            os.makedirs(directory, exist_ok=True)
            for i, frame in enumerate(self.frames):
                frame.save(os.path.join(directory, f'frame_{i:03d}.png'), 'PNG')
            imageio.mimsave(filename, self.frames, duration=duration, loop=0)


class TrackedGenome(neat.DefaultGenome):
    def __init__(self, key):
        super().__init__(key)
        self.parent1_id = None
        self.parent2_id = None
        self.species = None

    def configure_crossover(self, genome1, genome2, config):
        super().configure_crossover(genome1, genome2, config)
        self.parent1_id = genome1.key
        self.parent2_id = genome2.key
        prune_dead_nodes(self, config)

    def configure_mutate(self, genome, config):
        super().configure_mutate(genome, config)
        self.parent1_id = genome.key
        self.parent2_id = None
        prune_dead_nodes(self, config)


def prune_dead_nodes(genome, config):
    input_keys = set(getattr(config, 'input_keys', []))
    output_keys = set(getattr(config, 'output_keys', []))
    hidden_keys = set(genome.nodes.keys())
    all_node_keys = hidden_keys | input_keys | output_keys

    # map each node to incoming nodes
    incoming = {node_key: [] for node_key in all_node_keys}
    for (i, o), conn in genome.connections.items():
        if not getattr(conn, 'enabled', True):
            continue
        incoming[o].append(i)
        
    # nodes that can be reached from an output node
    reachable = set(output_keys)
    frontier = list(output_keys)
    while frontier:
        temp = []
        for node in frontier:
            for incoming_node in incoming.get(node, []):
                if incoming_node not in reachable:
                    reachable.add(incoming_node)
                    temp.append(incoming_node)
        frontier = temp

    # nodes that cannot be reached from any output node
    prune = {n for n in hidden_keys if n not in reachable}

    # nodes that cannot be reached from any input node
    for n in hidden_keys:
        if n in input_keys or n in output_keys or n in prune:
            continue
        if not any((conn.enabled and o == n) for (_, o), conn in genome.connections.items()):
            prune.add(n)

    # remove edges connected to nodes in prune
    for (i, o) in list(genome.connections.keys()):
        if i in prune or o in prune:
            del genome.connections[(i, o)]

    # remove nodes in prune
    for node in prune:
        del genome.nodes[node]


class Visualizer(neat.reporting.BaseReporter):
    # width and height of a single network (networks are visualized side by side)
    def __init__(self, config, width=900, height=900, columns=3):
        self.config = config

        self.width = width
        self.height = height
        self.total_width = width * columns + 250
        self.total_height = height
        self.window = Canvas(self.total_width, self.total_height)

        self.generation = 0
        self.previous_population = {}
        self.species_map = {}

        self.fitness_history = []

        self.directory = 'gifs'
        self.filename = 'neat_progression.gif'
        self.species_filename = 'species_fitness.png'

        # remove old frames
        os.makedirs(self.directory, exist_ok=True)
        for file in os.listdir(self.directory):
            os.remove(os.path.join(self.directory, file))

    def update_species_map(self, species):
        new_map = {}
        for id, s in species.species.items():
            for gid in s.members:
                new_map[gid] = id
        self.species_map = new_map

    def post_evaluate(self, config, population, species, best_genome):
        # record max fitness per species
        current_gen_stats = {}
        for species_id, s in species.species.items():
            member_fitnesses = [
                population[gid].fitness 
                for gid in s.members 
                if gid in population and population[gid].fitness is not None
            ]
            if member_fitnesses:
                current_gen_stats[species_id] = max(member_fitnesses)
        self.fitness_history.append(current_gen_stats)

        # update species
        self.update_species_map(species)
        for gid, genome in population.items():
            genome.species_id = self.species_map.get(gid, None)
        for gid, genome in self.previous_population.items():
            genome.species_id = self.species_map.get(gid, None)

        # get parents
        parent1 = self.previous_population.get(getattr(best_genome, 'parent1_id', None))
        parent2 = self.previous_population.get(getattr(best_genome, 'parent2_id', None))
        # force parent1 fitness to always be greater than parent2 fitness
        if parent1 is not None and parent2 is not None:
            if parent1.fitness < parent2.fitness:
                parent1, parent2 = parent2, parent1
        elif parent2 is not None:
            parent1 = parent2
            parent2 = None

        # get mutations
        mutations = {'new_nodes': set(), 'mutated_connections': set()}
        if parent1:
            child_nodes = set(best_genome.nodes.keys())
            parent_nodes = set(parent1.nodes.keys())
            child_conns = set(best_genome.connections.keys())
            parent_conns = set(parent1.connections.keys())
            mutations['new_nodes'] = child_nodes - parent_nodes
            mutations['mutated_connections'] = child_conns - parent_conns

        # visualize parent and child networks
        self.window.fill_rect(0, 0, self.total_width, self.total_height, COLOR_WHITE)
        
        NET_WIDTH = self.width
        PADDING   = 100

        CONTENT_WIDTH = (NET_WIDTH * 3) + (PADDING * 4)
        EXTRA_SPACE   = self.total_width - CONTENT_WIDTH
        LEFT_OFFSET   = EXTRA_SPACE // 2

        X_PARENT1 = LEFT_OFFSET           + PADDING
        X_PARENT2 = X_PARENT1 + NET_WIDTH + PADDING
        X_CHILD   = X_PARENT2 + NET_WIDTH + PADDING

        if parent1:
            self.draw_genome(parent1, X_PARENT1, f'Parent 1: {parent1.key}')
        if parent2 and parent2 is not parent1:
            self.draw_genome(parent2, X_PARENT2, f'Parent 2: {parent2.key}')
        self.draw_genome(best_genome, X_CHILD, f'Child: {best_genome.key}', mutations)

        self.window.draw_text(X_PARENT1 + PADDING, 10, f'Gen: {self.generation}', color=COLOR_BLACK)
        
        self.window.save_frame()
        self.previous_population = dict(population)
        self.generation += 1

    def save_species_graph(self, filename='species_fitness.png'):
        if not self.fitness_history:
            return

        plt.figure(figsize=(12, 8), dpi=300)
        
        # all unique species ids
        all_species = set()
        for gen_stats in self.fitness_history:
            all_species.update(gen_stats.keys())

        sorted_species_ids = sorted(list(all_species))    
        cmap = cm.get_cmap('Set2')
        num_species = len(sorted_species_ids)
        colors = [cmap(i / num_species) for i in range(num_species)]
        species_color_map = {
            species_id: colors[i] 
            for i, species_id in enumerate(sorted_species_ids)
        }
            
        # line for each species
        for species_id in sorted(list(all_species)):
            x = []
            y = []
            for gid, gen_stats in enumerate(self.fitness_history):
                if species_id in gen_stats:
                    x.append(gid)
                    y.append(gen_stats[species_id])
            if x:
                plt.plot(x, y, 
                label=f'Species {species_id}', 
                color=species_color_map[species_id], 
                linestyle='-',
                linewidth=2.0
            )

        plt.title('Max Fitness by Species per Generation')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def draw_genome(self, genome, x_offset, genome_id, mutations=None):
        # 1. create layers of neural network
        config = self.config
        W, H = self.width, self.height

        input_keys = set(config.genome_config.input_keys)
        output_keys = set(config.genome_config.output_keys)

        # arranges the hidden nodes in layers based on their inputs and outputs
        def get_node_layers(genome, input_keys, output_keys):
            layers = {}

            # inputs are on layer 0
            for key in input_keys:
                layers[key] = 0

            # now we do hidden nodes
            hidden_keys = set(genome.nodes.keys())
            all_node_keys = hidden_keys | input_keys | output_keys
            incoming = {k: set() for k in all_node_keys}
            outgoing = {k: set() for k in all_node_keys}

            # build graph based on edges
            for i, o in genome.connections.keys():
                if genome.connections[(i, o)].enabled:
                    if o in incoming: incoming[o].add(i)
                    if i in outgoing: outgoing[i].add(o)

            # hidden nodes are on the layer just after their furthest input node
            # (starts by adding nodes with incoming edges only from inputs)
            ordering = []
            done = set(input_keys)
            queue = [None]
            while queue:
                queue = []
                for n in hidden_keys - done:   
                    if all(node in layers for node in incoming.get(n, set())):
                        if not incoming.get(n):
                            layers[n] = 0
                        else:
                            layers[n] = max(layers[node] for node in incoming[n]) + 1
                        queue.append(n)
                        done.add(n)
                ordering.extend(queue)

            return layers

        node_layers = get_node_layers(genome, input_keys, output_keys)

        # outputs are on the last layer
        output_layer = (max(node_layers.values()) + 1) if node_layers else 1
        for n in output_keys:
            node_layers[n] = output_layer

        # invert the index
        layers_map = defaultdict(list)
        for node, depth in node_layers.items():
            layers_map[depth].append(node)

        sorted_layers = sorted(layers_map.keys())
        num_layers = len(sorted_layers)

        # 2. draw neural network
        node_xy = {}
        dx = 0
        if num_layers > 1:
            dx = (W - 100) / (num_layers - 1)

        # layer and node spacing
        for layer_index, depth in enumerate(sorted_layers):
            nodes_in_layer = layers_map[depth]
            nodes_in_layer.sort()
            num_nodes = len(nodes_in_layer)

            if num_layers > 1:
                layer_x = x_offset + 50 + (layer_index * dx)
            else:
                layer_x = x_offset + W/2

            dy = H / (num_nodes + 1)
            for i, n in enumerate(nodes_in_layer):
                y_pos = int((i + 1) * dy)
                x_pos = int(layer_x)
                node_xy[n] = (x_pos, y_pos)

        # draw edges (weights)
        for (i, o), conn in genome.connections.items():
            if conn.enabled and i in node_xy and o in node_xy:
                color = COLOR_POS_EDGE if conn.weight > 0 else COLOR_NEG_EDGE
                if mutations and (i, o) in mutations.get('mutated_connections', []):
                    color = COLOR_NEW_EDGE
                x1, y1 = node_xy[i]
                x2, y2 = node_xy[o]
                self.window.draw_line(x1, y1, x2, y2, color, width=max(1, math.ceil(abs(conn.weight))))

        # draw nodes
        NODE_RADIUS = 40
        NODE_DIAMETER = NODE_RADIUS * 2
        Y_OFFSET_ID = -10
        Y_OFFSET_BIAS = -65
        X_OFFSET_ACTIVATION = 60
        Y_OFFSET_ACTIVATION = 20

        for n, (x, y) in node_xy.items():
            # node circle
            if n in input_keys:    fill_color = COLOR_INPUT
            elif n in output_keys: fill_color = COLOR_OUTPUT
            else:                  fill_color = COLOR_HIDDEN
            if mutations and n in mutations.get('new_nodes', set()): fill_color = COLOR_NEW
            self.window.fill_circle(
                x - NODE_RADIUS, y - NODE_RADIUS, NODE_DIAMETER, NODE_DIAMETER, fill_color
            )
            # id
            self.window.draw_aligned_text(
                x, y + Y_OFFSET_ID, str(n), color=COLOR_WHITE, font=TINY_FONT, align='center'
            )
            # for the hidden and output nodes
            if n not in input_keys:
                node = genome.nodes[n]
                # bias
                bias_value = getattr(node, 'bias', config.genome_config.bias_init_mean)
                bias_text = f'bias: {bias_value:.2f}'
                self.window.draw_aligned_text(
                    x, y + Y_OFFSET_BIAS, bias_text, color=COLOR_BLACK, font=TINIEST_FONT, align='center'
                )
                # activation function
                activation = getattr(node, 'activation', config.genome_config.activation_default)
                act_text = ACTIVATION_ABBREV.get(activation, activation)
                self.window.draw_aligned_text(
                    x + X_OFFSET_ACTIVATION, y + Y_OFFSET_ACTIVATION, act_text, color=COLOR_BLACK, font=TINIEST_FONT_BOLD, align='center'
                )

        # genome info like id, fitness score, and species
        BOX_X_START = x_offset + 300
        BOX_Y_START = 40
        LINE_HEIGHT = 60
        PADDING = 10

        fitness = getattr(genome, 'fitness', 0.0)
        species = getattr(genome, 'species_id', 'N/A')

        TEXT_X = BOX_X_START + PADDING
        # fitness
        self.window.draw_text(
            TEXT_X, BOX_Y_START + PADDING, f'Fitness: {fitness:.2f}', color=COLOR_BLACK, font=FONT
        )
        # species
        self.window.draw_text(
            TEXT_X, BOX_Y_START + PADDING + LINE_HEIGHT, f'Species: {species}', color=COLOR_BLACK, font=FONT
        )
        # genome id
        self.window.draw_text(x_offset + 300, H - 50, genome_id, color=COLOR_BLACK, font=FONT)


# forward pass predictions on NEAT nn
def predict_and_score(nn, X, y, num_outputs=None):
    correct = 0
    predictions = []

    for xi, yi in zip(X, y):
        y_hat = nn.activate(xi)

        if num_outputs == 1: y_pred = 1 if y_hat[0] > 0.5 else 0    # binary        
        else: y_pred = np.argmax(y_hat)                             # multiclass

        predictions.append((xi, y_pred, yi, y_hat))
        if y_pred == yi:
            correct += 1

    return correct, predictions


def make_fitness_function(train_x, train_y, num_outputs=1, loss_type='accuracy'):
        # fitness function (kinda like a reward function in reinforcement learning)
        def fitness_function(genomes, config):
            for id, genome in genomes:
                nn = neat.nn.FeedForwardNetwork.create(genome, config)
                
                if loss_type == 'accuracy':
                    correct, _ = predict_and_score(nn, train_x, train_y, num_outputs=num_outputs)
                    fitness = correct
                elif loss_type == 'mse':
                    fitness = len(train_x)
                    for xi, yi in zip(train_x, train_y):
                        y_hat = nn.activate(xi)
                        fitness -= (y_hat[0] - yi) ** 2
                else:
                    raise ValueError(f'invalid loss_type: {loss_type}')

                # complexity regularization strength
                alpha = 0.005
                num_conns = sum(1 for conn in genome.connections.values() if conn.enabled)
                genome.fitness = fitness - (alpha * num_conns)
                
        return fitness_function


# main
def main(config_file, train_x, train_y, test_x, test_y, loss_type=None):
    if len(np.unique(train_y)) == 2: num_outputs = 1 
    else:                            num_outputs = len(np.unique(train_y))
    fitness_function = make_fitness_function(train_x, train_y, num_outputs, loss_type)

    config = neat.Config(
        TrackedGenome,
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
    visualizer = Visualizer(config)
    population.add_reporter(visualizer)

    winner = population.run(fitness_function, 100)
    print(f'\nBest genome: {winner}')

    # info of the winner and test accuracy
    winner_nn = neat.nn.FeedForwardNetwork.create(winner, config)
    correct, predictions = predict_and_score(winner_nn, test_x, test_y, num_outputs)

    print('\nPredictions on test set:')
    for xi, y_pred, yi, y_hat in predictions:
        print(f'input: {xi}, prediction probability: {y_hat[0]:.2f}, predicted: {y_pred}, actual: {yi}')
    accuracy = correct / len(test_x)
    print(f'\nTest Accuracy: {accuracy * 100:.2f}%')

    # save the gif
    visualizer.save_species_graph(visualizer.species_filename)
    visualizer.window.save_frames(visualizer.filename, visualizer.directory)


# modify the num_inputs, num_outputs, and fitness_threshold parameters in the config file
def update_config_io(X, y, fitness_threshold, config_file):
    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(config_file)

    num_inputs = X.shape[1]
    unique_classes = np.unique(y)
    if set(unique_classes) == {0, 1}: num_outputs = 1
    else:                             num_outputs = len(unique_classes)

    if 'TrackedGenome' not in config:
        raise ValueError(f'TrackedGenome section not found in config file {config_file}')
    if 'NEAT' not in config:
        raise ValueError(f'NEAT section not found in config file {config_file}')

    config['TrackedGenome']['num_inputs'] = str(num_inputs)
    config['TrackedGenome']['num_outputs'] = str(num_outputs)
    config['NEAT']['fitness_threshold'] = fitness_threshold

    with open(config_file, 'w') as f: config.write(f)
    print(f'Config updated: num_inputs={num_inputs}, num_outputs={num_outputs}')


# examples
if __name__ == '__main__':
    config_file = 'config-feedforward.ini'
    example = 1
    if example == 1:
        # xor
        X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])
        y = np.array([0, 1, 1, 0])

        fitness_threshold = '4'

        update_config_io(X, y, fitness_threshold, config_file)
        main(config_file, X, y, X, y, loss_type='mse')
    else:
        # iris
        iris = load_iris()
        X, y = iris.data, iris.target

        fitness_threshold = '118'

        update_config_io(X, y, fitness_threshold, config_file)
        train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)
        main(config_file, train_x, train_y, test_x, test_y, loss_type='accuracy')
