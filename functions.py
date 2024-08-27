"""
Helper functions for MCTS.
Author: Nhat Nguyen (School of Computer Science - University of Adelaide)

Reference:
Dec-MCTS: Decentralized planning for multi-robot active perception
[https://doi.org/10.1177/0278364918755924]
"""


import numpy as np
import math
from collections import deque
from tree import Tree
from copy import deepcopy


# Random choice among all available options.
def random_choice(vector:list):
    idx = np.random.randint(low=0, high=len(vector), size=1)[0]
    choice = vector[idx]
    return choice


# Random choice with probabilities.
def random_choice_bias(choices:list, pb:list):
    unnorm_pb = pb
    cuml_pb = np.cumsum(unnorm_pb)
    random_draw = cuml_pb[-1] * np.random.rand()
    idx = np.where(random_draw < cuml_pb)[0][0]
    return choices[idx]


# Backpropagation for decentralised MCTS.
def backprop_decmcts(tree:Tree, tree_idx:int, root_idx:int, current_score:float, current_path:list, gamma:float, current_epoch:int):
    # Node sequences to be backpropagated.
    backtrace = tree.get_seq_indice(tree_idx, root_idx)

    # Update new score and number of time node being selected using the dec-mcst formula.
    old_score = np.array(tree.data.score[backtrace].copy())
    last_backprop = np.array(tree.data.last_backdrop_epoch[backtrace].copy())
    current_n = np.array(tree.data.N[backtrace].copy())

    discount_factor = np.power(gamma, (current_epoch - last_backprop))
    discount_n = np.multiply(discount_factor, current_n)
    new_accumulative_score = np.multiply(discount_n, old_score) + current_score
    new_n = discount_n + 1

    tree.data.loc[backtrace, 'score'] = np.divide(new_accumulative_score, new_n)
    tree.data.loc[backtrace, 'N'] = new_n

    # Update best score, best path and current epoch.
    to_replace = list()
    for i in backtrace:
        if current_score > tree.data.loc[i, 'best_rollout_score']:
            to_replace.append(i)
            tree.data.at[i, 'best_rollout_path'] = current_path
    tree.data.loc[to_replace, 'best_rollout_score'] = current_score
    tree.data.loc[backtrace, 'last_backdrop_epoch'] = current_epoch


# Backpropagation for SW-MCTS.
def backprop_swucb(tree:Tree, tree_idx:int, root_idx:int, current_score:int, current_path:list, previous_score:deque, window_size:int, current_epoch:int):
    # Node sequences to be backpropagated.
    backtrace = tree.get_seq_indice(tree_idx, root_idx)

    # Update the sequence of actions and scores queue.
    # previous score: [[score at t, [list of nodes selected at t]], ...]
    previous_score.append((current_score, backtrace))

    for node in backtrace:
        sw_n = 0
        sw_accumulative_score = 0
        for i in range(max(0, len(previous_score) - window_size), len(previous_score)):
            if node in previous_score[i][1]:
                sw_n += 1
                sw_accumulative_score += previous_score[i][0]
        tree.data.loc[node, 'score'] = np.divide(sw_accumulative_score, sw_n)
        tree.data.loc[node, 'N'] = sw_n

    # Update best score, best path and current epoch.
    to_replace = list()
    for i in backtrace:
        if (abs(current_epoch - tree.data.loc[i, 'last_backdrop_epoch']) > window_size) or (current_score > tree.data.loc[i, 'best_rollout_score']):
            to_replace.append(i)
            tree.data.at[i, 'best_rollout_path'] = current_path
    tree.data.loc[to_replace, 'best_rollout_score'] = current_score
    tree.data.loc[backtrace, 'last_backdrop_epoch'] = current_epoch


# Update the probabilities of choosing action sequences using regret matching.
def updateRegretMatchingDistribution(num_iterations:int, robots:list, active_robots, f_payoff):
    # Set up the cumlative regret dict for each agent
    cumulative_regrets = dict()
    for rob_idx in active_robots:
        cumulative_regrets[rob_idx] = np.zeros(len(robots[rob_idx].distribution[rob_idx].prob))

    # Best joit policy and global payoff.
    best_joint_policy = dict()
    best_payoff = 0

    # Compute the strategies using regret maching
    for _ in range(num_iterations):
        action_sequences = dict()
        # Sample action sequences for every agents
        for index in active_robots:
            action_sequences[index] = random_choice_bias(robots[index].distribution[index]['path'].copy().tolist(),
                                                    robots[index].distribution[index]['prob'].copy().tolist())
        # Calculate the global payoff
        actual_payoff = f_payoff(action_sequences)
        if actual_payoff > best_payoff:
            best_payoff = actual_payoff
            best_joint_policy = deepcopy(action_sequences)

        # Update the cumulative regrets for each action sequence of each robot
        for index in active_robots:
            # Copy values for calculation
            what_if_actions = deepcopy(action_sequences)
            other_actions = robots[index].distribution[index]['path'].copy().tolist()
            # Regrets equal difference between what-if action and actual action
            for i in range(len(other_actions)):
                what_if_actions.update({index: other_actions[i]})
                cumulative_regrets[index][i] += f_payoff(what_if_actions) - actual_payoff
            # Update regret-matching strategy
            pos_cumulative_regrets = np.maximum(0, cumulative_regrets[index])
            if sum(pos_cumulative_regrets) > 0:
                robots[index].distribution[index].prob = pos_cumulative_regrets / sum(pos_cumulative_regrets)
            else:
                robots[index].distribution[index].prob = np.full(shape=len(robots[index].distribution[index].prob), fill_value=1/len(robots[index].distribution[index].prob))

    return best_joint_policy, robots


"""
Greedy search - add each action sequence that maximises the intermediate payoff M*N.
"""
def greedy_search(robots, active_robots, f_payoff):
    # Best joit policy and global payoff.
    best_joint_policy = dict()

    # Loop throughh each robot sequentially and choose the action sequence that maximises the intermediate payoff.
    action_sequences = dict()
    for index in active_robots:
        best_payoff = -1
        for i in range(len(robots[index].distribution[index].path)):
            action_sequences[index] = robots[index].distribution[index].path[i].copy()
            intermediate_payoff = f_payoff(action_sequences)
            if intermediate_payoff > best_payoff:
                best_payoff = intermediate_payoff
                best_joint_policy[index] = robots[index].distribution[index].path[i].copy()
                robots[index].distribution[index].at[i, 'prob'] = 1
        action_sequences[index] = best_joint_policy[index].copy()

    return best_joint_policy, robots


"""
Exhaustive search through the whole search space M^N.
"""
def exhaustive_search(robots, active_robots, f_payoff):
    # Get the number of agents and the number of action sequences.
    num_agents = len(active_robots)
    num_actions = 0
    for index in active_robots:
        if len(robots[index].distribution[index].prob) > num_actions:
            num_actions = len(robots[index].distribution[index].prob)
    action_sequences = dict()

    num_iterations = pow(num_actions, num_agents)

    # Best joit policy and global payoff.
    best_joint_policy = dict()
    best_payoff = 0
    j = 0

    # Compute every possible combinations.
    for i in range(num_iterations):
        action_sequences.clear()
        for index in active_robots:
            action_idx = math.floor(i / pow(num_actions, num_agents - j - 1) % num_actions)
            if action_idx < len(robots[index].distribution[index].prob):
                action_sequences[index] = robots[index].distribution[index].path[action_idx].copy()
            j = (j + 1) % num_agents

        payoff = f_payoff(action_sequences)
        # Return the best combination.
        if payoff > best_payoff:
            best_payoff = payoff
            best_joint_policy = action_sequences.copy()

    return best_joint_policy

