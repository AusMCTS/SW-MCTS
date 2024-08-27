"""
Simulation of Centralised MCTS on multi-agents task.
Author: Nhat Nguyen (School of Computer Science - University of Adelaide)

"""


import numpy as np
import argparse
import csv
import os
import time
from graph import Graph
from tree import Tree
from agent import Central_Robot
from functions import *
from graph_helper import *
from mcts import growTree
from progressbar import progressbar
from datetime import datetime
from copy import deepcopy


# Create a list of agents.
def create_robot(n_agents:int, G:Graph):
    robots = list()
    initial_actions = []

    # Get available actions at the staring positions.
    for j in range(len(G.edges_list)):
        if G.edges_list[j][0] == 0:
            initial_actions.append(j)

    for i in range(n_agents):
        robots.append(Central_Robot(np.nan, deepcopy(initial_actions), agents, i, n_agents))

    return robots


def exporting_results(directory, config, trial, rollout_path, reward_per_round, time_per_round, files_name):
    np.savetxt("{}/{}-performance.csv".format(directory, files_name), reward_per_round, delimiter=",")
    np.savetxt("{}/{}-time.csv".format(directory, files_name), time_per_round, delimiter=",")
    np.savetxt("{}/{}-planning_score.csv".format(directory, files_name), score_per_iter, delimiter=",")
    with open("{}/{}-rollout-C{}-T{}.csv".format(directory, files_name, config, trial+1), "w", newline='') as f:
        write = csv.writer(f)
        for path in rollout_path:
            write.writerow([path])


 # Joint path histories of all robots.
def f_joint(edge_history:list, n_agents:int):
    edge_history_robots = dict()
    for i in range(n_agents):
        edge_history_robots[i] = list()
    for i in range(1, len(edge_history)):
        edge_history_robots[(i - 1) % n_agents].append(edge_history[i])
    return edge_history_robots


# Terminal condition is when the travelling budget of the agent expires.
def f_terminal(edge_history:list):
    edge_history_robots = f_joint(edge_history, n_agents)
    robot_history = edge_history_robots[(len(edge_history) - 2) % n_agents]
    return True if len(robot_history) > max_actions else False


# Get reward mask of the edge between the current node to all immediate successor nodes.
def evaluate_immediate_actions(G:Graph, available_actions:list):
    pb = list()
    for i in range(len(available_actions)):
        pb.append(1 + sum(G.evaluate_edge_reward(available_actions[i])))
    return pb


if __name__ == '__main__':

    # Parsing the input options.
    parser = argparse.ArgumentParser(description="Simulate Cen-MCTS on multi-agent tasks")
    parser.add_argument("-s", "--save", help="Save performance", action='store_true', default=False)
    parser.add_argument("-c", "--changes", help="Type of environment changes", choices=['LC', 'DC'])
    parser.add_argument("-f", "--folder", help="Folder name")
    parser.add_argument("-v", "--verbose", help="Print details", action='store_true', default=False)
    parser.add_argument("-p", "--params", help="Parameter testing", nargs="+")
    args = parser.parse_args()

    # System parameters.
    xL = -2                                           # min of the x-axis.
    xH = 2                                            # max of the x-axis.
    yL = -2                                           # min of the y-axis.
    yH = 2                                            # max of the y-axis.
    obsMask = np.array([[0]])                         # the obstacles mask.

    # Model of change type.
    change_type = args.changes if args.changes != None else None

    # Fixed parameters.
    N_trial = 2                                       # number of trial experiments per configurations.
    N_configs = 10                                    # number of configurations to test.
    reward_radius = 0.05                              # reward disk radius.
    c_p = 0.4                                         # Exploration parameter                                

    # Shared variational parameters.
    max_actions = 10                                  # action budget.
    planning_time = 500                               # planning time budget for each round.
    n_agents = 10                                     # number of agents.
    n_rewards = 200
    cable_length = 0.05                               # Anchor cable length.
    change_percent = 1.0
    change_rate = 1.0 if change_type != None else 0.0
    velocity = 0.005                                  # agent's moving velocity (km/s).
    trans_rate = 5000                                 # sensor's transmission rate (bit/s).
    packet_size = 1000                                # packet size of sensor (bits).

    if args.params != None:
        param_name = args.params[0]
        param_value = args.params[1]
        if param_name == "agent":
            param_name = "Agent"
            n_agents = int(param_value)
        elif param_name == "action":
            param_name = "Action"
            max_actions = int(param_value)
        elif param_name == "planning":
            param_name = "Planning"
            planning_time = int(param_value)
        elif param_name == "reward":
            param_name = "Reward"
            n_rewards = int(param_value)
        elif param_name == "rate":
            param_name = "Planning"
            change_rate = float(param_value)
            param_value = planning_time
        elif param_name == "cable":
            param_name = "Planning"
            cable_length = float(param_value)
            param_value = planning_time
        elif param_name == "percent":
            param_name = "Planning"
            change_percent = float(param_value)
            param_value = planning_time
        elif param_name == "packet":
            packet_size = int(param_value)

    try:
        if args.save:
            if args.folder != None:
                directory = os.path.join("../Data/", args.folder)
            else:
                now = datetime.now()
                directory = os.path.join("../Data/", now.strftime("%Y-%m-%d-%H-%M"))
            if not os.path.isdir(directory):
                try:
                    os.mkdir(directory)
                except:
                    pass
            # Reward/rollout score and planning time matrix.
            if change_type == None:
                reward_per_round = np.array(np.zeros([1, N_configs*N_trial]))
                time_per_round = np.array(np.zeros([1, N_configs*N_trial]))
            else:
                reward_per_round = np.array(np.zeros([max_actions, N_configs*N_trial]))
            # File names.
            files_name = "Central"
            # Planning score.
            score_per_iter = np.array(np.zeros([planning_time, N_configs*N_trial]))

        # Initialise a seed for reproductivity.
        rng =  np.random.default_rng(12345)

        for config in range(N_configs):
            # Generate graph environment.
            G = Graph(xL, xH, yL, yH, reward_radius, obsMask)
            path = "../Data/Config_{}".format(config)
            G, agents, anchored_latlongs, nodes, _, n_nodes, _ = import_graph(G, path)
            if anchored_latlongs.shape[0] != n_rewards:
                anchored_latlongs = modify_number_rewards(G, n_rewards, anchored_latlongs, seed=config)

            # Available functions are listed of successor nodes.
            f_actions = lambda edge_history: robots[len(edge_history)-1].get_actions() if len(edge_history) <= n_agents else G.find_edge(edge_history[-n_agents])
            # UCB functions.
            f_ucb = lambda Np, Nc: 2 * c_p * np.sqrt(np.divide(np.log(Np), Nc))
            # Pick next action based on its immediate reward (greedy random).
            f_sampler = lambda available_actions: random_choice_bias(available_actions, evaluate_immediate_actions(G, available_actions))
            # Global objective function.
            f_score = lambda edge_history: sum(G.evaluate_traj_reward(f_joint(edge_history, n_agents), packet_size))/n_rewards

            if args.verbose:
                print("Config {}/{}".format(config+1, N_configs))
                print("Generating the graph environment with {} agents, {} actions, {} rewards, {} iterations".format(n_agents, max_actions, n_rewards, planning_time))
                if change_type == "LC":
                    print("Cable length: {}, change rate: {}.".format(cable_length, change_rate))
                    change_degree = cable_length
                elif change_type == "DC":
                    print("Percent change: {}, change rate: {}.".format(change_percent, change_rate))
                    change_degree = change_percent

            # Start simulation.
            for trial in range(N_trial):
                print("Trial {}/{}".format(trial+1, N_trial))
                if args.save:
                    rollout_path = list()

                # Create robots.
                robots = create_robot(n_agents, G)
                active_robots = list(range(0, n_agents))

                tree = Tree(state=[robots[0].get_state()],
                            actions_to_try=[robots[0].get_actions()],
                            score=0,
                            N=0,
                            best_rollout_score=-np.inf, 
                            best_rollout_path=[list()])

                # Reset environment model.
                rewards = deepcopy(anchored_latlongs)
                G.reset_reward(n_rewards)
                # G.add_reward(rewards)
                G.add_trans_bits(rewards, velocity, trans_rate, packet_size)
                acc_collected_rewards = np.zeros(n_rewards, dtype=int)

                if change_type == None:
                    # Planning start.
                    start = time.time()
                    # for current_iter in progressbar(range(planning_time), redirect_stdout=True):
                    for current_iter in range(planning_time):
                        # Grow the tree search.
                        rollout_score, rollout_history = growTree(tree, 0, f_score, f_actions, f_terminal, f_ucb, f_sampler)

                        if rollout_score > tree.data.at[0, 'best_rollout_score']:
                            tree.data.at[0, 'best_rollout_score'] = rollout_score
                            tree.data.at[0, 'best_rollout_path'] = f_joint(rollout_history, n_agents)

                        # Keep track of planning score.
                        if args.save:
                            score_per_iter[current_iter][N_trial*config + trial] = rollout_score
                    # Planning ends.
                    end = time.time()

                    # Evaluate the executed the joint action sequences with the real graph.
                    joint_rollout = tree.data.at[0, 'best_rollout_path']
                    acc_collected_rewards = G.evaluate_traj_reward(joint_rollout, packet_size)
                    joint_excecution_score = sum(acc_collected_rewards)/n_rewards

                    # Save values for analytics.
                    if args.save:
                        reward_per_round[0][N_trial*config + trial] = joint_excecution_score
                        time_per_round[0][N_trial*config + trial] = end - start
                        rollout_path.append(joint_rollout)
                    if args.verbose:
                        print("Score: {}".format(joint_excecution_score))

                    if args.save:
                        exporting_results(directory, config, trial, rollout_path, reward_per_round, time_per_round, files_name)
                else:
                    with open("../Data/Static/{}={}/Central-rollout-C{}-T{}.csv".format(param_name, param_value, config, trial+1), 'r') as csv_file:
                        csv_reader = list(csv.reader(csv_file))
                        central_joint_path = parse(csv_reader[-1][0])

                    collected_rewards = dict()
                    for i in range(n_agents):
                        collected_rewards[i] = np.zeros(n_rewards, dtype=int)
                    for action_order in range(max_actions):
                        joint_immediate_path = dict()
                        for i in range(n_agents):
                            joint_immediate_path[i] = central_joint_path[i][action_order:action_order+1]
                            collected_rewards[i] += G.evaluate_edge_reward(joint_immediate_path[i][0])
                            acc_collected_rewards |= (collected_rewards[i] >= packet_size)
                        reward_per_round[action_order][N_trial*config + trial] = sum(acc_collected_rewards)/n_rewards

                        if (action_order < max_actions - 1) and (rng.random() < change_rate):
                            G, rewards = modify_graph_rewards(G, anchored_latlongs, rewards, n_rewards, change_degree, change_type, seed=(100*config+10*trial+action_order))
                            G.reset_reward(n_rewards)
                            # G.add_reward(rewards)
                            G.add_trans_bits(rewards, velocity, trans_rate, packet_size)
                    np.savetxt("{}/Central-performance.csv".format(directory), reward_per_round, delimiter=",")

    except KeyboardInterrupt:
        print("Simulation discarding. Exporting results so far.")
        if args.save:
            exporting_results(directory, config, trial, rollout_path, reward_per_round, time_per_round, files_name)
