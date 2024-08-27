"""
Environment files constructor helper function.
Author: Nhat Nguyen (School of Computer Science - University of Adelaide)

"""

from graph import Graph
from prm import SampleFree, CollisionFree, Euclindean_dist, Near_radius
from matplotlib import pyplot as plt
from progressbar import ProgressBar
from math import floor, ceil
from copy import deepcopy
import numpy as np
import csv
import argparse
import os
import sys


def generate_rewards(G:Graph, n_rewards:int):
    # Randomly generate the rewards initial locations, equally distributed.
    rewards = SampleFree(floor(n_rewards/4),
                        G.xL + 0 * (G.xH - G.xL),
                        G.xH - 0.5 * (G.xH - G.xL),
                        G.yL + 0 * (G.yH - G.yL),
                        G.yH - 0.5 * (G.yH - G.yL),
                        G)
    rewards = np.append(rewards, SampleFree(floor(n_rewards/4),
                        G.xL + 0 * (G.xH - G.xL),
                        G.xH - 0.5 * (G.xH - G.xL),
                        G.yL + 0.5 * (G.yH - G.yL),
                        G.yH - 0 * (G.yH - G.yL),
                        G), axis=0)
    rewards = np.append(rewards, SampleFree(ceil(n_rewards/4),
                        G.xL + 0.5 * (G.xH - G.xL),
                        G.xH - 0 * (G.xH - G.xL),
                        G.yL + 0 * (G.yH - G.yL),
                        G.yH - 0.5 * (G.yH - G.yL),
                        G), axis=0)
    rewards = np.append(rewards, SampleFree(ceil(n_rewards/4),
                        G.xL + 0.5 * (G.xH - G.xL),
                        G.xH - 0 * (G.xH - G.xL),
                        G.yL + 0.5 * (G.yH - G.yL),
                        G.yH - 0 * (G.yH - G.yL),
                        G), axis=0)

    return rewards


def generate_new_graph(G:Graph, n_agents:int, n_nodes:int, n_rewards:int, path:str):
    # Initialised the agents positions.
    agents = SampleFree(floor(n_agents),
                        G.xL + 0.5 * (G.xH - G.xL),
                        G.xL + 0.5 * (G.xH - G.xL),
                        G.yL + 0.5 * (G.yH - G.yL),
                        G.yH - 0.5 * (G.yH - G.yL),
                        G)
    np.savetxt("{}/agents.csv".format(path), agents, delimiter=",")
    for i in range(n_agents):
        G.add_node(agents[i][0], agents[i][1])

    # Generate the intermediate nodes.
    nodes = np.zeros((0, 2))
    x = G.xL
    while x <= G.xH:
        nodes = np.append(nodes, np.array([[x, G.yL], [x, G.yH]]), axis=0)
        x += max_edge_length
    x = G.yL
    while x <= G.yH:
        nodes = np.append(nodes, np.array([[G.xL, x], [G.xH, x]]), axis=0)
        x += max_edge_length

    nodes = np.append(nodes, SampleFree(n_nodes-len(nodes),
                        G.xL,
                        G.xH,
                        G.yL,
                        G.yH,
                        G), axis=0)

    np.savetxt("{}/nodes.csv".format(path), nodes, delimiter=",")
    for i in range(len(nodes)):
        G.add_node(nodes[i][0], nodes[i][1])

    # Generate rewards.
    rewards = generate_rewards(G, n_rewards)
    np.savetxt("{}/rewards.csv".format(path), rewards, delimiter=",")
    for i in range(len(rewards)):
        G.add_node(rewards[i][0], rewards[i][1])

    # Generate edges between nodes.
    with ProgressBar(max_value=(len(G.nodes))) as bar:
        for v in G.nodes.items():
            U = Near_radius(G, v[1], r=max_edge_length)
            for u in U:
                if CollisionFree(G, v[1], u[1]):
                    G.add_edge(v[0], u[0], Euclindean_dist(u[1], v[1]), n_rewards)
            bar.update(v[0])
    with open("{}/edges.csv".format(path), 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for start in G.edges.keys():
            for end in G.edges[start].keys():
                writer.writerow([start, end,])

    # Check what edge covers what rewards.
    G.add_reward(rewards)
    return G, agents, rewards, nodes


def import_graph(G:Graph, dir:str):
    # Get existing agents positions.
    agents = np.genfromtxt("{}/agents.csv".format(dir), delimiter=",")
    n_agents = len(agents)
    if n_agents > 2:
        for i in range(n_agents):
            G.add_node(agents[i][0], agents[i][1])
    elif n_agents == 2:
        x = agents[0]
        y = agents[1]
        G.add_node(x, y)
        n_agents = 1
        agents = np.array([[x, y]])
    else:
        print("Format Error!")
        sys.exit()

    # Get existing nodes latlongs.
    nodes = np.genfromtxt("{}/nodes.csv".format(dir), delimiter=",")
    n_nodes = len(nodes)
    for i in range(len(nodes)):
        G.add_node(nodes[i][0], nodes[i][1])

    # Get existing rewards.
    rewards = np.genfromtxt("{}/rewards.csv".format(dir), delimiter=",")
    n_rewards = len(rewards)
    for i in range(len(rewards)):
        G.add_node(rewards[i][0], rewards[i][1])

    # Generate edges.
    with open("{}/edges.csv".format(dir), 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        for edges in reader:
            if edges:
                start = int(edges[0])
                end = int(edges[1])
                G.add_edge(start, end, Euclindean_dist(G.find_coordinates(start), G.find_coordinates(end)), n_rewards)

    return G, agents, rewards, nodes, n_agents, n_nodes, n_rewards


def modify_graph_rewards(G:Graph, anchored_latlongs, rewards, n_rewards:int, change_degree:float, mode:str, seed=12345):
    # Initialise a seed for reproductivity.
    rng =  np.random.default_rng(seed)

    # Dynamic changes mean sensors can move anywhere within the plane.
    if mode == "DC":
        n_change = int(change_degree*n_rewards)
        n_same = n_rewards - n_change
        # Some rewards remain the same.
        rewards = rng.choice(rewards, n_same, replace=False)
        # Others change randomly.
        rewards = np.append(rewards, SampleFree(n_change, G.xL, G.xH, G.yL, G.yH, G, seed), axis=0)
    # Changes with cable.
    elif mode == "LC":
        # Sensor moves in a circle with radius equal to cable length.
        delta_x = rng.uniform(low=-change_degree, high=change_degree, size=n_rewards)
        max_delta_y = np.sqrt(np.power(change_degree, 2) - np.power(delta_x, 2))
        delta_y = rng.uniform(low=-max_delta_y, high=max_delta_y, size=n_rewards)

        # Modify the latlong of the rewards.
        rewards[:, 0] = anchored_latlongs[:, 0] + delta_x
        rewards[:, 1] = anchored_latlongs[:, 1] + delta_y

        # Wrap-around
        rewards[:, 0] = np.where(rewards[:, 0] < G.xL, rewards[:, 0] + (G.xH - G.xL), rewards[:, 0])
        rewards[:, 0] = np.where(rewards[:, 0] > G.xH, rewards[:, 0] - (G.xH - G.xL), rewards[:, 0])
        rewards[:, 1] = np.where(rewards[:, 1] < G.yL, rewards[:, 1] + (G.yH - G.yL), rewards[:, 1])
        rewards[:, 1] = np.where(rewards[:, 1] > G.yH, rewards[:, 1] - (G.yH - G.yL), rewards[:, 1])

    # Update the simulated model.
    # G.reset_reward(n_rewards)
    # G.add_reward(rewards)

    return G, rewards


def remove_graph_rewards(G:Graph, acc_collected_rewards:list, rewards, n_rewards:int, fail_round:int):
    # Get collected rewards index.
    remove_idx = [i for i in range(len(acc_collected_rewards)) if acc_collected_rewards[i] == fail_round]

    # Randomly select some rewards to be removed.
    rewards[remove_idx] = np.nan

    # Update the simulated model.
    G.reset_reward(n_rewards)
    G.add_reward(rewards)

    return G, rewards


def update_graph_model(G:Graph, rewards, rewards_real, estimated_collected_rewards:list, collected_rewards:list, n_rewards:int, change_type:str):
    # Get collected rewards index.
    idx = [i for i in range(len(collected_rewards)) if collected_rewards[i]]
    # Check if reward is expected but not actually collected.
    absent_idx = [i for i in range(len(estimated_collected_rewards)) if estimated_collected_rewards[i] and not collected_rewards[i]]

    # Check flag.
    flag = False

    # Check if latlongs have changed or miscollect any rewards.
    if not np.array_equal(rewards[idx], rewards_real[idx]) or len(absent_idx) > 0:
        # Update latlongs of collected rewards.
        # rewards[idx] = deepcopy(rewards_real[idx])
        rewards = deepcopy(rewards)

        # If miscollect any rewards, set latlongs to nan.
        if change_type == "AC":
            rewards[absent_idx] = np.nan

        # Update the simulated model.
        G.reset_reward(n_rewards)
        G.add_reward(rewards)
        flag = True

    return G, rewards, flag


# Modify the number of rewards.
def modify_number_rewards(G: Graph, n_rewards:int, rewards, seed=0):
    # Delta between the desired number of rewards and the current.
    nNew = int(n_rewards) - rewards.shape[0]
    # Add more rewards.
    if nNew > 0:
        rewards = np.append(rewards, SampleFree(nNew, G.xL, G.xH, G.yL, G.yH, G, seed), axis=0)
    # Remove some rewards.
    else:
        rng = np.random.default_rng(seed)
        rewards = rng.choice(rewards, n_rewards, replace=False)

    return rewards


# Parsing the rollout files.
def parse(d):
    dictionary = dict()
    # Removes curly braces and splits the pairs into a list
    pairs = d.strip('{}').split('], ')
    for i in pairs:
        pair = i.split(': ')
        x = pair[1].strip('[]').split(', ')
        dictionary[int(pair[0])] = [float(num) for num in x[0:]]
    return dictionary


def draw_graph(G:Graph, agents, rewards):
    G.draw_edges()
    G.draw_nodes()
    G.draw_agents(agents)
    G.draw_rewards(rewards)
    G.draw_obs()
    plt.show()


if __name__ == '__main__':
    # Parsing the input options.
    parser = argparse.ArgumentParser(description="Graph constructor")
    parser.add_argument("-a", "--animation", help="Show constructed graph", action='store_true', default=False)
    parser.add_argument("-n", "--n_configs", help="No of configurations", default=1)
    parser.add_argument("-d", "--draw", help="Draw existing graph")
    args = parser.parse_args()

    # Declare system size.
    xL = -2     # min of the x-axis
    xH = 2      # max of the x-axis
    yL = -2     # min of the y-axis
    yH = 2      # max of the y-axis
    obsMask = np.array([
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ])                                  # the obstacles mask.

    # Declare graph configurations.
    n_rewards = 200                             # number of reward disks.
    n_nodes = 200                               # number of intermediate nodes.
    n_agents = 1                                # number of agents.
    reward_radius = 0.05                        # reward disk radius.
    max_edge_length = 0.8                       # max edge length between any 2 nodes.
    n_configs = int(args.n_configs)             # Number of configurations to construct.

    # Parent Directory path to save the files.
    parent_dir = "../Data/"

    if args.draw:
        dir = "../Data/{}".format(args.draw)
        G = Graph(xL, xH, yL, yH, reward_radius, obsMask)
        G, agents, rewards, _, _, _, _ = import_graph(G, dir)
        draw_graph(G, agents, rewards)

    else:
        for i in range(n_configs):
            # Create config directory.
            directory = "Config_{}".format(i)
            path = os.path.join(parent_dir, directory)
            if not os.path.isdir(path):
                os.mkdir(path)
            # Create graph.
            G = Graph(xL, xH, yL, yH, reward_radius, obsMask)
            G, agents, rewards, nodes = generate_new_graph(G, n_agents, n_nodes, n_rewards, path)

        if args.animation and n_configs == 1:
            draw_graph(G, agents, rewards)
