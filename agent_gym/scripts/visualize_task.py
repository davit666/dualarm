import os

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import pybullet as p

from os.path import exists, basename, abspath, dirname
# from copy import copy
# from json import dump
# from tqdm import tqdm
import numpy as np
import pybullet_data
import pybullet as p

import os, inspect
import time

from gym_envs.Env_tasks import Mode, Env_tasks
from config import load_config, load_task_config
from stable_baselines3 import PPO

if __name__ == "__main__":

    # Load config
    train_config, env_config = load_config()
    task_config = load_task_config()

    env_name = task_config['env_name']
    alg_name = task_config['alg_name']

    num_cpu = task_config['num_cpu']
    reward_type = task_config['task_allocator_reward_type']
    obs_type = task_config['task_allocator_obs_type']
    print(reward_type, obs_type)
    custom_network = task_config['custom_network']
    dict_obs = task_config['dict_obs']
    load_model = task_config['load_model']
    task_allocator_load_model_path = task_config['task_allocator_load_model_path']

    env = Env_tasks(env_config, renders=True, useInverseKinematics=True, freezeAction=False,
                    showBallMarkers=False, maxSteps=task_config['maxSteps'],
                    action_type=task_config['action_type'],
                    task_allocator_reward_type=task_config['task_allocator_reward_type'],
                    task_allocator_obs_type=task_config['task_allocator_obs_type'],
                    motion_planner_reward_type=task_config['motion_planner_reward_type'],
                    motion_planner_obs_type=task_config['motion_planner_obs_type'],
                    fragment_length=task_config['fragment_length'], parts_num=task_config['parts_num'])

    if task_config['selected_motion_planner_policy']:
        motion_planner_load_model_path = task_config['motion_planner_load_model_path']
        motion_planner_policy = PPO.load(motion_planner_load_model_path)
    else:
        motion_planner_policy = None
    env.assign_policy(policy=motion_planner_policy)

    if alg_name == "PPO":
        task_allocator_policy = PPO.load(task_allocator_load_model_path)
    else:
        task_allocator_policy = SAC.load(task_allocator_load_model_path)



    for i in range(100):
        count = 0
        obs = env.reset()
        done = False
        while not done:
            count += 1
            print(env.state_info['masks_obs'])
            dist_cost = env.state_info['dist_cost']
            # act = task_allocator(obs=dist_cost)
            # act = env.action_space.sample()
            act, _states = task_allocator_policy.predict(obs, deterministic=True)

            obs, r, done, info = env.step(act)
            print('allocator action:\t', act)
            print('rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr:\t',r)
            # time.sleep(1)


        print("total_step used:\t", count * task_config['fragment_length'] , "\t!!!!!!!!!!")
        time.sleep(3)




