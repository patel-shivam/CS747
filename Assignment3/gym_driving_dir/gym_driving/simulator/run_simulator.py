from importlib.resources import path
from gym_driving.assets.car import *
from gym_driving.envs.environment import *
from gym_driving.envs.driving_env import *
from gym_driving.assets.terrain import *

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import time
import pygame, sys
from pygame.locals import *
import random
import math
import argparse

# Do NOT change these values
TIMESTEPS = 1000
FPS = 30
NUM_EPISODES = 10

class Task1():

    def __init__(self):
        """
        Can modify to include variables as required
        """

        super().__init__()
        self.init_region = None
        self.exited_corner_region_once = False

    def set_init_region(self, state):
        
        if state[0]>300 and state[1]<-150:
            # x>300 and y<-150
            # return 'A'
            self.init_region = 'A'
        elif state[0]>300 and state[1]>150:
            # x>300 and y>150
            # return 'B'
            self.init_region = 'B'
        else:
            self.init_region = 'C'
        
        return

    def get_current_region(self, state):
        
        if state[0]>300 and state[1]<-150:
            return 'A'
        elif state[0]>300 and state[1]>150:
            return 'B'
        else:
            return 'C'

    def next_action(self, state):
        """
        Input: The current state
        Output: Action to be taken
        TO BE FILLED
        """

        x=state[0]
        y=state[1]
        vel=state[2]
        angle=state[3]
        curr_region = self.get_current_region(state)
        x_target = 350
        y_target = 0
     
        
        if angle >=180 :
            angle = angle-360   # converting angle to [-180, 179]

        # to check if car has ever exited 'edge-regions'
        # to avoid continous steering at inner boundary
        if self.init_region == 'A' or self.init_region == 'B' :
            if self.init_region != curr_region:
                self.exited_corner_region_once = True
        

        if self.init_region == 'A' and curr_region == 'A' and self.exited_corner_region_once == False:
            # theta_slope = 90, as we want to go straight up to x axis, and then let the normal algorithm chose what to do
            theta_slope=90


        elif self.init_region == 'B' and curr_region == 'B' and self.exited_corner_region_once == False:
            # theta_slope = -90, as we want to go straight down to x axis, and then let the normal algorithm chose what to do
            theta_slope=-90
            

        else:
            theta_slope = np.degrees(np.arctan((y_target-y)/(x_target-x + 1e-12)))
            # calculating the target angle

        # control algorithm, either steer or full acc
        if abs(angle-theta_slope)>5:
            if angle-theta_slope>0:
                action_steer = 0 
                action_acc = 2
            else :
                action_steer = 2 
                action_acc = 2
        else:
            action_steer = 1
            action_acc = 4

        return np.array([action_steer, action_acc])



    def controller_task1(self, config_filepath=None, render_mode=False):
        """
        This is the main controller function. You can modify it as required except for the parts specifically not to be modified.
        Additionally, you can define helper functions within the class if needed for your logic.
        """
    
        ######### Do NOT modify these lines ##########
        pygame.init()
        fpsClock = pygame.time.Clock()

        if config_filepath is None:
            config_filepath = '../configs/config.json'

        simulator = DrivingEnv('T1', render_mode=render_mode, config_filepath=config_filepath)

        time.sleep(3)
        ##############################################

        # e is the number of the current episode, running it for 10 episodes
        for e in range(NUM_EPISODES):
        
            ######### Do NOT modify these lines ##########
            
            # To keep track of the number of timesteps per epoch
            cur_time = 0

            # To reset the simulator at the beginning of each episode
            state = simulator._reset()
            
            # Variable representing if you have reached the road
            road_status = False
            ##############################################



            # sets the initial state
            self.set_init_region(state)
            if self.init_region == 'A' or self.init_region == 'B':
                self.exited_corner_region_once = False
            else:
                self.exited_corner_region_once = True 

            # if it was never in the corner region, then it is equivalent to exiting it 
            # The following code is a basic example of the usage of the simulator
            for t in range(TIMESTEPS):
        
                # Checks for quit
                if render_mode:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                            sys.exit()

                action = self.next_action(state)
                state, reward, terminate, reached_road, info_dict = simulator._step(action)
                fpsClock.tick(FPS)

                cur_time += 1

                if terminate:
                    road_status = reached_road
                    break

            # Writing the output at each episode to STDOUT
            print(str(road_status) + ' ' + str(cur_time))

class Task2():

    def __init__(self):
        """
        Can modify to include variables as required
        """

        super().__init__()
        self.pit_centers = []

    def is_above_pit(self, state):
        x=state[0]
        y=state[1]
        vel=state[2]
        angle=state[3]
        c_list = self.pit_centers
        flag = False
        for i in range(4):
            x_c = c_list[i][0]
            y_c = c_list[i][1]
            if y_c>0 and y>y_c: # case for the upper two mudpits, for y > 0
                if x < x_c + 80 and x > x_c - 80:
                    flag = True
            elif y_c<0 and y<y_c: # case for the upper two mudpits, for y > 0
                if x < x_c + 80 and x > x_c - 80:
                    flag = True
        return flag

    def get_quadrant(self, state):
        x=state[0]
        y=state[1]
        vel=state[2]
        angle=state[3]

        if x>0 :
            if y>0 :
                return '1'
            else: 
                return '4'
        else:
            if y>0:
                return '2'
            else:
                return '3'

    def next_action(self, state):
        """
        Input: The current state
        Output: Action to be taken
        TO BE FILLED

        You can modify the function to take in extra arguments and return extra quantities apart from the ones specified if required
        """

        x=state[0]
        y=state[1]
        vel=state[2]
        angle=state[3]
        if angle >=180 :
            angle = angle-360   # converting angle to [-180, 179]

        # Replace with your implementation to determine actions to be taken
        
        c_list = self.pit_centers

        # go towards the y axis from above the pit
        if self.is_above_pit(state)==True: 
            if self.get_quadrant(state)=='1':
                theta_slope = 175
            elif self.get_quadrant(state)=='2':
                theta_slope = 0
            elif self.get_quadrant(state)=='3':
                theta_slope = 0
            elif self.get_quadrant(state)=='4':
                theta_slope = 175

        else: # go towards the x axis or the target (350,0)

            if y>15:
                theta_slope = -90
            elif y<-15:
                theta_slope = 90
            else:
                x_target = 350
                y_target = 0
                theta_slope = np.degrees(np.arctan((y_target-y)/(x_target-x + 1e-12)))

        # if it was never in the corner region, then it is equivalent to exiting it 
        # The following code is a basic example of the usage of the simulator
        if abs(angle-theta_slope)>2:
            if angle-theta_slope>0:
                action_steer = 0 
                action_acc = 2
            else :
                action_steer = 2 
                action_acc = 2
        else:
            action_steer = 1
            action_acc = 4

        action = np.array([action_steer, action_acc])  

        return action

    def controller_task2(self, config_filepath=None, render_mode=False):
        """
        This is the main controller function. You can modify it as required except for the parts specifically not to be modified.
        Additionally, you can define helper functions within the class if needed for your logic.
        """
        
        ################ Do NOT modify these lines ################
        pygame.init()
        fpsClock = pygame.time.Clock()

        if config_filepath is None:
            config_filepath = '../configs/config.json'

        time.sleep(3)
        ###########################################################

        # e is the number of the current episode, running it for 10 episodes
        for e in range(NUM_EPISODES):

            ################ Setting up the environment, do NOT modify these lines ################
            # To randomly initialize centers of the traps within a determined range
            ran_cen_1x = random.randint(120, 230)
            ran_cen_1y = random.randint(120, 230)
            ran_cen_1 = [ran_cen_1x, ran_cen_1y]

            ran_cen_2x = random.randint(120, 230)
            ran_cen_2y = random.randint(-230, -120)
            ran_cen_2 = [ran_cen_2x, ran_cen_2y]

            ran_cen_3x = random.randint(-230, -120)
            ran_cen_3y = random.randint(120, 230)
            ran_cen_3 = [ran_cen_3x, ran_cen_3y]

            ran_cen_4x = random.randint(-230, -120)
            ran_cen_4y = random.randint(-230, -120)
            ran_cen_4 = [ran_cen_4x, ran_cen_4y]

            ran_cen_list = [ran_cen_1, ran_cen_2, ran_cen_3, ran_cen_4]            
            eligible_list = []

            # To randomly initialize the car within a determined range
            for x in range(-300, 300):
                for y in range(-300, 300):

                    if x >= (ran_cen_1x - 110) and x <= (ran_cen_1x + 110) and y >= (ran_cen_1y - 110) and y <= (ran_cen_1y + 110):
                        continue

                    if x >= (ran_cen_2x - 110) and x <= (ran_cen_2x + 110) and y >= (ran_cen_2y - 110) and y <= (ran_cen_2y + 110):
                        continue

                    if x >= (ran_cen_3x - 110) and x <= (ran_cen_3x + 110) and y >= (ran_cen_3y - 110) and y <= (ran_cen_3y + 110):
                        continue

                    if x >= (ran_cen_4x - 110) and x <= (ran_cen_4x + 110) and y >= (ran_cen_4y - 110) and y <= (ran_cen_4y + 110):
                        continue

                    eligible_list.append((x,y))

            simulator = DrivingEnv('T2', eligible_list, render_mode=render_mode, config_filepath=config_filepath, ran_cen_list=ran_cen_list)
        
            # To keep track of the number of timesteps per episode
            cur_time = 0

            # To reset the simulator at the beginning of each episode
            state = simulator._reset(eligible_list=eligible_list)
            ###########################################################

            # The following code is a basic example of the usage of the simulator
            road_status = False

            self.pit_centers = ran_cen_list





            for t in range(TIMESTEPS):
        
                # Checks for quit
                if render_mode:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                            sys.exit()

                action = self.next_action(state)
                state, reward, terminate, reached_road, info_dict = simulator._step(action)
                fpsClock.tick(FPS)

                cur_time += 1

                if terminate:
                    road_status = reached_road
                    break

            print(str(road_status) + ' ' + str(cur_time))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config filepath", default=None)
    parser.add_argument("-t", "--task", help="task number", choices=['T1', 'T2'])
    parser.add_argument("-r", "--random_seed", help="random seed", type=int, default=0)
    parser.add_argument("-m", "--render_mode", action='store_true')
    parser.add_argument("-f", "--frames_per_sec", help="fps", type=int, default=30) # Keep this as the default while running your simulation to visualize results
    args = parser.parse_args()

    config_filepath = args.config
    task = args.task
    random_seed = args.random_seed
    render_mode = args.render_mode
    fps = args.frames_per_sec

    FPS = fps

    random.seed(random_seed)
    np.random.seed(random_seed)

    if task == 'T1':
        
        agent = Task1()
        agent.controller_task1(config_filepath=config_filepath, render_mode=render_mode)

    else:

        agent = Task2()
        agent.controller_task2(config_filepath=config_filepath, render_mode=render_mode)
