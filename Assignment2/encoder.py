import argparse
import numpy as np
from pulp import *

def read_player_stats(path):
    with open(path,'r') as f:
        player_stats = {i:0 for i in range(5)} 
        # i is the actions available to player A

        lines=f.readlines()
        num_states = len(lines)
        for i in range(5):
            player_stats[i] = [float(j) for j in lines[i+1].strip().split()[1:]]
            # removing the index of the action (the first number in that line)

    return player_stats

def get_next_player(curr_state,runs_made):
    # returns the next player in the upcoming state
    balls = int(curr_state[0:2])
    runs = int(curr_state[2:4])
    player = curr_state[-1]
    if balls%6!=1:
        return 'a'if ((runs_made%2==0 and player=='a') or (runs_made%2==1 and player=='b'))  else 'b'
    else:
        return 'b'if ((runs_made%2==0 and player=='a') or (runs_made%2==1 and player=='b'))  else 'a'

def get_next_states_a(curr_state):
    # returns the list of states for each action that player A takes
    # the probability and anything is not returned, only list of states for all environment results
    # edge cases - number of balls left is one - 
    # more runs made than number of balls at any time - game win 
    runs_outcome = [-1,0,1,2,3,4,6]
    balls = int(curr_state[0:2])
    runs = int(curr_state[2:4])
    player = curr_state[-1]
    if runs<=6 and balls>1:
        next_states = ['end']
        for i in range(1,7):
            next_player = get_next_player(curr_state,runs_outcome[i])
            if runs_outcome[i]>=runs:
                next_states.append('win')
            else:
                next_states.append(str(balls-1).zfill(2)+str(runs-runs_outcome[i]).zfill(2)+next_player)

    elif balls == 1:
        next_states = ['end']
        for i in range(1,7):
            next_player = get_next_player(curr_state,runs_outcome[i])
            # next_player = 'a'if ((runs_outcome[i]%2==0 and player=='a') or (runs_outcome[i]%2==1 and player=='b'))  else 'b'
            if runs_outcome[i]>=runs:
                next_states.append('win')
            else:
                next_states.append('end')

    else:
        next_states = ['end']
        
        for i in range(1,7):
            next_player = get_next_player(curr_state,runs_outcome[i])
            next_states.append(str(balls-1).zfill(2)+str(runs-runs_outcome[i]).zfill(2)+next_player)
    return next_states




def get_next_states_b(curr_state):
    # returns the possible next states if player b is playing
    runs_outcome = [-1,0,1]
    balls = int(curr_state[0:2])
    runs = int(curr_state[2:4])
    player = curr_state[-1]
    
    if balls>1:
        if runs>1:
            next_states = ['end']
            for i in range(1,3):
                next_player = get_next_player(curr_state,runs_outcome[i])
                next_states.append(str(balls-1).zfill(2)+str(runs-runs_outcome[i]).zfill(2)+next_player)
        if runs==1:
            next_states = ['end']
            next_player = get_next_player(curr_state,runs_outcome[1])
            next_states.append(str(balls-1).zfill(2)+str(runs).zfill(2)+next_player)
            next_states.append('win')
    
    elif balls==1:
        if runs==1:
            next_states = ['end','end','win']
        else:
            next_states = ['end','end','end']

    return next_states




def encode(states_list, player_stats,q_value):
    new_states_list = []
    transition_list = []
    new_states_list.append(states_list[0]+'a')
    for i in range(1,len(states_list)):
        new_states_list.append(str(states_list[i])+'a')  # a is playing this state
        new_states_list.append(str(states_list[i])+'b')  # b is playing this state


    for state in new_states_list:
        balls = int(state[:2])
        runs = int(state[2:4])
        player = state[-1]
        
        

        if player=='a':
            list_next_states = get_next_states_a(state)
            # these are the actions that player A can take 
            # player_actions = [0,1,2,4,6]
            # rewards are decided by the environment 
            for i in range(5): # 5 actions available to the player A
                p_list = player_stats[i]
                for j in range(7): # 7 possible next states when player A is playing
                    # appending [s1,action,s2,r,p]
                    if p_list[j]>0:
                        if list_next_states[j]=='win':
                            transition_list.append([state,i,'win',1,p_list[j]])
                        elif list_next_states[j]=='end':
                            transition_list.append([state,i,'end',0,p_list[j]])
                        else:
                            transition_list.append([state,i,list_next_states[j],0,p_list[j]])


        elif player=='b':
            list_next_states = get_next_states_b(state)
            p_list = [q_value,(1-q_value)/2,(1-q_value)/2]
            for i in range(5):
                # these are the actions that player A can take, but reward is by B itself
                # player_actions = [0,1,2,4,6]
                # rewards are decided by the environment 
                
                for j in range(3): # only 3 next states when b is playing
                    if list_next_states[j]=='win':
                        transition_list.append([state,i,'win',1,p_list[j]])
                    elif list_next_states[j]=='end':
                        transition_list.append([state,i,'end',0,p_list[j]])
                    else:
                        transition_list.append([state,i,list_next_states[j],0,p_list[j]])
                
    return transition_list
            
def read_states(path):
    # reads the states from a file
    with open(path,'r') as f:
        lines=f.readlines()
        balls = int(str(lines[0][:2]))
        runs = int(str(lines[0][2:4]))

        player_states = []
        for i in range(len(lines)):
            player_states.append(str(lines[i].strip()))
            # returning a list of states

        return player_states, balls, runs



def give_state_integer(state, max_balls, max_runs):
    # returns an encoded state to an integer format
    if state=='win':
        return 1
    elif state=='end':
        return 0
    balls = int(state[0:2])
    runs = int(state[2:4])
    player = str(state[-1])
    if balls==max_balls:
        if runs==max_runs:
            return 2
        elif player=='a':
            return (max_runs-runs)*2 + 1
        elif player=='b':
            return (max_runs-runs)*2 + 2
    else:
        if player=='a':
            return 2*max_runs*(max_balls-balls) + (max_runs-runs)*2 + 1
        elif player=='b':
            return 2*max_runs*(max_balls-balls) + (max_runs-runs)*2 + 2



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--states', type=str)
    parser.add_argument('--parameters', type=str)
    parser.add_argument('--q', type=str)
    args = parser.parse_args()

    player_stats = read_player_stats(args.parameters)
    states_list, balls, runs = read_states(args.states)
    num_states = 2*len(states_list)+1
    q_value = float(args.q)

    print('numStates '+str(num_states))
    print('numActions '+str(5))
    print('end 0 1')  # 0 is end state, 1 is win state
    transition_list = encode(states_list, player_stats,q_value)
    # we are mapping bbrr to 2 to n integers, where and the 

    for i in range(len(transition_list)):
        int_this_state = give_state_integer(transition_list[i][0], balls, runs)
        int_next_state = give_state_integer(transition_list[i][2], balls, runs)
        print('transition '+str(int_this_state)+' '+str(transition_list[i][1])+' '+str(int_next_state)+' '+str(transition_list[i][3])+' '+str(transition_list[i][4]))
        
    print('mdptype episodic')
    print('discount 1')

