import os

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import pybullet as p

from os.path import exists, basename, abspath, dirname
from time import time
# from copy import copy
# from json import dump
# from tqdm import tqdm
import numpy as np
import pybullet_data
import pybullet

import os,inspect
import time
import gym
from Env_a_k import Env_a_k
from p_utils import (
    draw_sphere_body,
    remove_markers,
    remove_obstacles
)

def simulate():

    env = Env_a_k(renders=True, isDiscrete=False,showBallMarkers = False,maxSteps = 500)
    print("===================================")
    env.reset()
 ################################################################################################################
    for j in range(20000):
        
        time.sleep(0.05)
        act = env.action_space.sample()
        obs, reward, done, episode_info = env.step(act)  
        if done:
            print(episode_info)
            time.sleep(3)
            env.reset()
################################################################################################################
    # for i in range (100):
    #     obs = env.reset()




if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    simulate()
