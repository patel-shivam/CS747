"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

You need to complete the following methods:
    - give_pull(self): This method is called when the algorithm needs to
        select the arms to pull for the next round. The method should return
        two arrays: the first array should contain the indices of the arms
        that need to be pulled, and the second array should contain how many
        times each arm needs to be pulled. For example, if the method returns
        ([0, 1], [2, 3]), then the first arm should be pulled 2 times, and the
        second arm should be pulled 3 times. Note that the sum of values in
        the second array should be equal to the batch size of the bandit.
    
    - get_reward(self, arm_rewards): This method is called just after the
        give_pull method. The method should update the algorithm's internal
        state based on the rewards that were received. arm_rewards is a dictionary
        from arm_indices to a list of rewards received. For example, if the
        give_pull method returned ([0, 1], [2, 3]), then arm_rewards will be
        {0: [r1, r2], 1: [r3, r4, r5]}. (r1 to r5 are each either 0 or 1.)
"""

import numpy as np

# START EDITING HERE
# You can use this space to define any helper functions that you need.
# END EDITING HERE

class AlgorithmBatched:
    def __init__(self, num_arms, horizon, batch_size):
        # np.random.seed(1)
        self.num_arms = num_arms
        self.horizon = horizon
        self.batch_size = batch_size
        self.a = np.zeros(num_arms)
        self.b = np.zeros(num_arms)

        assert self.horizon % self.batch_size == 0, "Horizon must be a multiple of batch size"
        # START EDITING HERE
        # Add any other variables you need here
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        sampled_matrix = np.random.beta(self.a+1, self.b+1, (self.batch_size, self.num_arms))
        # each column of the above represents the num_arms and each row represents the ith pull of the batch_size
        # we are performing thompson sampling each arm for batch_size number of times,
        # the maximum value of the thompson sample for each pull in btach_size is taken
        # THOMPSON SUBSAMPLING
        arms_to_sample = []
        number_of_samples = []
        # creating the output arrays containing the arm index and respective number of times to pull that arm
        for i in range(self.batch_size):
            k = np.argmax(sampled_matrix[i,:])
            if k not in arms_to_sample:
                arms_to_sample.append(k)
                number_of_samples.append(0)
            number_of_samples[arms_to_sample.index(k)] += 1
        #print(arms_to_sample,number_of_samples)
        return arms_to_sample, number_of_samples

        # END EDITING HERE
    
    def get_reward(self, arm_rewards):
        # START EDITING HERE
        # calculating the number of times reward is obtained, to update a and b 
        for arm in arm_rewards:
            l = len(arm_rewards[arm])
            ones = sum(arm_rewards[arm])
            zeros = l - ones
            self.a[arm] += ones
            self.b[arm] += zeros
        return 
        # END EDITING HERE