import argparse
import numpy as np
from pulp import *


def read_v_pi(path_policy):
    # reads the policy from planner output
    with open(path_policy,'r') as f_policy: 
        lines_policy = f_policy.readlines()
        action_map = [0,1,2,4,6]
        state_list=[]
        value_list=[]
        policy_list = []
        value_list.append(float(lines_policy[2].strip().split()[0]))
        policy_list.append(action_map[int(lines_policy[2].strip().split()[1])])
        for i in range(3,len(lines_policy),2):
            value_list.append(float(lines_policy[i].strip().split()[0]))
            policy_list.append(action_map[int(lines_policy[i].strip().split()[1])])


    return value_list, policy_list
    
def read_states(path):
    # reads the states from cricket_state_list.txt
    with open(path,'r') as f:

        lines=f.readlines()
        balls = int(str(lines[0][:2]))
        runs = int(str(lines[0][2:4]))

        player_states = []
        for i in range(len(lines)):
            player_states.append(str(lines[i].strip()))

        return player_states, balls, runs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--value-policy', type=str)
    parser.add_argument('--states', type=str)

    args = parser.parse_args()
    value_list, policy_list = read_v_pi(args.value_policy)
    # reading data


    states_list, balls, runs = read_states(args.states)

    for i in range(len(states_list)):
        print(str(states_list[i])+str(' ')+str(policy_list[i])+str(' ')+str(value_list[i]))
        # printing data in required format

