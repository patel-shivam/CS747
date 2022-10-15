"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the AlgorithmManyArms class. Here are the method details:
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
"""

import numpy as np
import math
# START EDITING HERE
# class AlgorithmManyArms: Thompson Sampling (failing)
#     def __init__(self, num_arms, horizon):
#         self.num_arms = num_arms
#         self.a = np.zeros(num_arms)
#         self.b = np.zeros(num_arms)
#         self.horizon = horizon
#         self.counts = np.zeros(num_arms)
#         self.num_explore = math.floor(1*math.sqrt(self.horizon))
#         self.chosen_samples = np.random.choice(self.num_arms, size=(self.num_explore), replace=False)
#         # Horizon is same as number of arms
    
#     def give_pull(self):
#         # START EDITING HERE
#         if self.t < 1*self.num_explore:
#            maxm_ix = self.chosen_samples[self.t % self.num_explore]
#         else:
#            samples_arr = np.random.beta(self.a+1, self.b+1)
#            maxm_ix = np.where(samples_arr == samples_arr.max())[0][0]
#         return maxm_ix
#         # END EDITING HERE
    
#     def get_reward(self, arm_index, reward):
#         # START EDITING HERE
#         if reward==1:
#             self.a[arm_index] += 1 
#         else:
#             self.b[arm_index] += 1
        
#         return 

#         # if n==0:
#         #     self.pa[arm_index] = 0.5
#         # else:
#         #     self.pa[arm_index] = self.counts[arm_index]*(n-1)/n + reward/n
#         # self.counts[arm_index] += 1
#         # return 
#         raise NotImplementedError
#         # END EDITING HERE

# END EDITING HERE



# np.random.seed(109)
class AlgorithmManyArms:
    def __init__(self, num_arms, horizon):
        # START EDITING HERE
        # np.random.seed(1)
        self.num_arms = num_arms
        self.pa = np.zeros(num_arms)
        self.t = 0
        self.horizon = horizon
        self.counts = np.zeros(num_arms)
        self.num_explore = math.floor(1*math.sqrt(self.horizon))
        self.eps = 1/self.num_explore
        # I experimented with how many arms to randomly sample initially (effectively blocking out all other arms from the algorithm)
        self.chosen_samples = np.random.choice(self.num_arms, size=(self.num_explore), replace=False)
        # returns an array of indices
        # Horizon is same as number of arms
        # END EDITING HERE

    def give_pull(self):  # epsilon greedy 3 
        # START EDITING HERE
        # I tried thompson sampling as well on the num_explore chosen arms, but due to effectively low number of times each arm is samples,
        # success and failure is low, meaning that the variance is high. This led to "bad" arms being sampled frequently
        if self.t < 1*self.num_explore: # exploring stage
            # here the coefficient of self.num_Explore determines how many times each of the num_explore chosen arms is to be sampled
            maxm_ix = self.chosen_samples[self.t % self.num_explore]
            # the index returned is the tantamount to the round robin arm pulling method (hence modulo operator)
        else: # exploiting stage
            if np.random.random() < self.eps:
                maxm_ix = self.chosen_samples[self.t % self.num_explore]
            else:
                maxm_ix = np.argmax(self.pa)
        self.t += 1
        return maxm_ix

        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        self.pa[arm_index] = self.pa[arm_index]*(n-1)/n + reward/n        
        return
        # END EDITING HERE