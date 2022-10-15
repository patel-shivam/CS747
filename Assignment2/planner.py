import argparse
import numpy as np
from pulp import *

def read_policy(path):
    with open(path,'r') as f:
        policy_list = []
        lines=f.readlines()
        num_states = len(lines)
        for i in range(num_states):
            policy_list.append(int(lines[i].strip()))
    return policy_list

# dictionary storing the mdp
# key represents the state number
# value represents the list of lists, where each sublist is corresponding 
# to a transition - [a, s_next, t, r] 

# have also created an additional dictionary with tuple(state, action) 
# as key, and the value as a list of [r,t,s2]
def read_mdp(path):
    with open(path,'r') as f:
        dict_mdp = {}
        dict_pol_eval = {}
        lines=f.readlines()
        num_states = int(lines[0].strip().split()[1])
        num_actions = int(lines[1].strip().split()[1])
        end_states = list(map(int, lines[2].strip().split()[1:]))
        for i in range(3,len(lines)-2):
            trans = lines[i].strip().split()
            s1 = int(trans[1])
            ac = int(trans[2])
            s2 = int(trans[3])
            r = float(trans[4])
            p=float(trans[5])
            if s1 not in dict_mdp.keys():
                dict_mdp[s1] = []
            dict_mdp[s1].append([s1,ac,s2,r,p])
            if tuple([s1,ac]) not in dict_pol_eval.keys():
                dict_pol_eval[tuple([s1,ac])] = []
            dict_pol_eval[tuple([s1,ac])].append([s2,r,p])
        mdptype = lines[-2].strip().split()[1]
        discount = float(lines[-1].strip().split()[1])
        mdp_info = [num_states, num_actions, end_states, mdptype, discount]
        
        
        
        return mdp_info, dict_mdp, dict_pol_eval






def value_iteration(mdp_info, dict_mdp):
    end_states = mdp_info[2]

    gamma = mdp_info[4]
    policy = [0]*mdp_info[0]
    num_states = mdp_info[0]
    num_actions = mdp_info[1]
    v_t = np.array([0]*num_states)
    v_t1 = np.array([0]*num_states)+0.1
    for end in end_states:
        v_t1[end] = 0

    for i in range(500000):
        if np.max(np.abs(v_t1-v_t)) < 1e-12:
            break
        v_t = v_t1.copy()
        # very important to use copy here, or else same pointer to both v_t1 and v_t
        for s1 in set(range(mdp_info[0])).difference(set(end_states)):
            arr = [0]*num_actions # len = num_actions
            for ele in dict_mdp[s1]:
                arr[ele[1]]+=ele[4]*(ele[3]+gamma*v_t[ele[2]])
                # arr[action] += t(s1,action,s2)[R(s1,action,s2)+gamma*V_t(s2)]
            v_t1[s1] = max(arr)
        
    # now, the value function is calculated,
    # we need to take argmax at each state for finding optimal policy
    for s1 in set(range(mdp_info[0])).difference(set(end_states)):
        arr = [0]*mdp_info[1] # len = num_actions
        for ele in dict_mdp[s1]:
            arr[ele[1]]+=ele[4]*(ele[3]+gamma*v_t[ele[2]])
            # arr[action] += t(s1,action,s2)[R(s1,action,s2)+gamma*V_t(s2)]
        policy[s1] = arr.index(max(arr)) # argmax over actions for each state

    return v_t, policy




def linear_solver(mdp_info, dict_mdp, dict_pol_eval):
    gamma = mdp_info[4]
    end_states = mdp_info[2]
    num_states = mdp_info[0]
    prob = LpProblem('MDP-valuefn' ,LpMinimize)
    states = [(s) for s in range(num_states)]
    vars = LpVariable.dicts('state_list',(states))
    prob += (lpSum([vars[s] for s in range(num_states)]) )
    for keys in dict_pol_eval.keys():
        # ele = [s2,r,p]
        prob += lpSum([ele[2]*(ele[1] + gamma*vars[ele[0]]) for ele in dict_pol_eval[keys]]) <= vars[keys[0]]
    if end_states[0]!=-1:
        for ends in end_states :
            prob += vars[ends]==0
    prob.solve(PULP_CBC_CMD(msg=0))
    # now we have the variables and their optimum value
    x_star=[0]*mdp_info[0]
    for i in range(num_states):
        x_star[i] = vars[i].value()
    
    
    policy = [0]*mdp_info[0]
    # now we find the optimal policy
    for s1 in range(mdp_info[0]):
        if s1 in end_states:
            policy[s1]=0
            continue
        arr = [0]*mdp_info[1] # len = num_actions
        for ele in dict_mdp[s1]:
            arr[ele[1]]+=ele[4]*(ele[3]+gamma*x_star[ele[2]])
            # arr[action] += t(s1,action,s2)[R(s1,action,s2)+gamma*V_t(s2)]
        policy[s1] = arr.index(max(arr)) # argmax over actions for each state

    return x_star, policy
    




def policy_eval(policy_list, dict_pol_eval, mdp_info):
    end_states = mdp_info[2]
    gamma = mdp_info[4]
    num_states = mdp_info[0]
    # dict_pol_eval[tuple([s1,ac])].append([s2,r,p])
    # dict_mdp[s1].append([s1,ac,s2,r,p]) 
    # structure of dictionaries

    a = np.zeros((num_states, num_states))
    # holds coefficients of all v_pi(s)
    b = np.zeros(num_states) # holds the constants in => a*v_pi = b 
    for s1 in range(num_states):
        if s1 in end_states:
            b[s1]=0
            a[s1,s1]=1
            # setting zero condition for terminating states
            continue    
         
        
        for ele in dict_pol_eval[(s1,policy_list[s1])]:
            b[s1] += -ele[1]*ele[2]
            a[s1,ele[0]] += ele[2]*gamma
        a[s1,s1] += -1

    v_pi = np.linalg.solve(a,b)
    return v_pi


def print_res(v_t, policy):
    # prints the result of the functions
    for i in range(len(v_t)):
        print(str(v_t[i])+"    "+str(policy[i]))
    return 

def get_qa(dict_mdp, mdp_info, dict_pol_eval, policy_list, v_current, state_curr):
    ## returns the set of all improvable actions for each state
    num_states = mdp_info[0]
    num_actions = mdp_info[1]
    gamma = mdp_info[4]
    end_states = mdp_info[2]
    qa = np.zeros(num_actions)

    for trans in dict_mdp[state_curr]:
        qa[trans[1]] +=  trans[4]*(trans[3] + gamma*v_current[trans[2]])
    improvable_states = []
    # list of action value function for all actions for any state state_curr
    for a in range(num_actions):
        if qa[a] > v_current[state_curr] +1e-12 :
            improvable_states.append(a)

    return improvable_states



def improve_policy(policy_list, d, mdp_info):
    # improves policy by switching all improvable state actions
    num_states = mdp_info[0]
    new_policy = [0]*num_states
    end_states = mdp_info[2]
    for s1 in range(num_states):
        if s1 in end_states:
            new_policy[s1] = policy_list[s1]
        elif len(d[s1])!=0:
            new_policy[s1] = np.random.choice(np.array(d[s1]))
        else:
            new_policy[s1] = policy_list[s1]

    return new_policy



def howard_solver(mdp_info, dict_mdp, dict_pol_eval, policy_list):
    # howard solver for policy iteration
    num_states = mdp_info[0]
    end_states = mdp_info[2]
    v_current = policy_eval(policy_list, dict_pol_eval, mdp_info)
    # v_current is the present policy value function
    d = {i:0 for i in range(num_states)}
    for s1 in set(range(num_states)).difference(set(end_states)):
        d[s1] = get_qa(dict_mdp, mdp_info, dict_pol_eval, policy_list, v_current, s1)

    new_policy_list = improve_policy(policy_list, d, mdp_info)
    # policy improvement
    if new_policy_list == policy_list:
        return v_current, policy_list
    else:
        policy_list = new_policy_list
        return howard_solver(mdp_info, dict_mdp, dict_pol_eval, policy_list)
        # recursive nature




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdp', type=str)
    parser.add_argument('--algorithm', type=str, default="vi")
    parser.add_argument('--policy', default="not_given")

    args = parser.parse_args()
    mdp_info, dict_mdp, dict_pol_eval = read_mdp(args.mdp)
    # reading the data
    if args.policy!='not_given':
            # policy_evaluation
            policy_list = read_policy(args.policy)
            mdp_info, dict_mdp, dict_pol_eval = read_mdp(args.mdp)
            v_t= policy_eval(policy_list, dict_pol_eval, mdp_info)
            print_res(v_t,policy_list)
        
    else:
        # optimal policy calculation 
        if args.algorithm =='vi':
            mdp_info, dict_mdp, dict_pol_eval = read_mdp(args.mdp)
            v_t, policy = value_iteration(mdp_info, dict_mdp)
            print_res(v_t,policy)
        elif args.algorithm =='lp':
            mdp_info, dict_mdp, dict_pol_eval = read_mdp(args.mdp)
            v_t, policy = linear_solver(mdp_info, dict_mdp, dict_pol_eval)
            print_res(v_t, policy)
        elif args.algorithm == 'hpi':
            mdp_info, dict_mdp, dict_pol_eval = read_mdp(args.mdp)
            init_policy = [0]*mdp_info[0]
            init_value_function =  [0]*mdp_info[0]
            v_t, policy = howard_solver(mdp_info, dict_mdp, dict_pol_eval, init_policy)
            print_res(v_t, policy)
            