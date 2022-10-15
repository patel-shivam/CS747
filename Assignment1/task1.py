"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
np.random.seed()
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value


# START EDITING HERE

# END EDITING HERE


class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.ucb_un = np.zeros(num_arms)  # keeps value of the second term in the ucb
        self.ucb_pa = np.zeros(num_arms)  # keeps the first term, empirical estimate pa
        self.ucb_total = np.zeros(num_arms)  # the sum of two terms, the upper confidence bound
        self.counts = np.ones(num_arms)   # the number of times the arm has been smaples, kept 1 initially to avoid divide by zero 
        self.total_samples = num_arms
       # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        return np.argmax(self.ucb_total)
        # return 0
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.total_samples += 1
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        self.ucb_pa[arm_index] = self.ucb_pa[arm_index]*(n-1)/n + reward/(n)
        self.ucb_un = np.sqrt((2*math.log(self.total_samples))/(self.counts))
        self.ucb_total = self.ucb_un + self.ucb_pa
        
        # END EDITING HERE

# start editing here
def kl(p,q):
    if p>0+1e-4 and p<1-1e-4:
        k = p*math.log(p/(q)) + (1-p)*math.log((1-p)/(1-q)) 
    elif p<1e-4:
        k = (1)*math.log((1)/(1-q))
    elif p>1-1e-4:
        k = 1*math.log(1/(q))
    return k
        
# def binsearch(p,qlow,qhigh,tk):
#     for i in range(10):
#         q_mid = (qlow+qhigh)/2
#         if (kl(p,qlow)-tk)*(kl(p,q_mid)-tk)<0:
#             qhigh = q_mid
#         else:
#             qlow=q_mid
#     return q_mid
# end editing here


class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        # self.klucb_un = np.zeros(num_arms)  # keeps value of the second term in the ucb
        self.klucb_pa = np.zeros(num_arms)  # keeps the first term, empirical estimate pa
        self.klucb_q = np.ones(num_arms)/1000
        self.kl_counts = np.ones(num_arms)    # the number of times the arm has been smaples, kept 1 initially to avoid divide by zero 
        self.time = 3
        self.num = num_arms
        self.c = 3
        # self.eps = 1e-4
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        for j in range(self.num):
            low = self.klucb_pa[j]
            high = 1
            ua = self.kl_counts[j]
            mid = (high+low)/2
            t = self.time
            p = self.klucb_pa[j]
            for i in range(10):
                mid = (high+low)/2
                if kl(p,mid) - (math.log(t) + self.c*math.log(math.log(t)))/ua <=0:
                    low = mid
                else:
                    high = mid

            self.klucb_q[j] = mid
        self.time += 1
        ix = np.argmax(self.klucb_q)
        self.kl_counts[ix] += 1
        return ix
        


        # return np.argmax(self.klucb_q)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        
        self.kl_counts[arm_index] += 1
        n = self.kl_counts[arm_index]

        self.klucb_pa[arm_index] = self.klucb_pa[arm_index]*(n-1)/n + reward/n                
        return 
        # for j in range(self.num):
        #     tk = (math.log(n) + self.c*math.log(math.log(n)))/self.kl_counts[j]
        #     self.klucb_q[j] = binsearch(self.klucb_pa[j], self.klucb_pa[j],1,self.kl_counts[j],self.time, self.c,tk)

        
        # END EDITING HERE


class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.a = np.zeros(num_arms)
        self.b = np.zeros(num_arms)

        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        return np.argmax(np.random.beta(self.a+1, self.b+1))
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        if reward==1:
            self.a[arm_index] +=1
        else:
            self.b[arm_index] +=1

        # END EDITING HERE
