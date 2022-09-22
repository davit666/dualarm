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
# from gym_envs.Env_gr import Env_gr
# from gym_envs.Env_rearrangement import Env_rearrangement
from gym_envs.Env_tasks import Mode, Env_tasks
from config import load_config, load_task_config
from stable_baselines3 import PPO

if __name__ == "__main__":

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
    motion_planner_load_model_path = task_config['motion_planner_load_model_path']

    env = Env_tasks(env_config, renders=True, useInverseKinematics=True, freezeAction=False,
                    showBallMarkers=False, maxSteps=task_config['maxSteps'],
                    task_allocator_action_type=task_config['task_allocator_action_type'],
                    task_allocator_reward_type=task_config['task_allocator_reward_type'],
                    task_allocator_obs_type=task_config['task_allocator_obs_type'],
                    motion_planner_reward_type=task_config['motion_planner_reward_type'],
                    motion_planner_obs_type=task_config['motion_planner_obs_type'],
                    motion_planner_action_type = task_config['motion_planner_action_type'],
                    fragment_length=task_config['fragment_length'], parts_num=task_config['parts_num'])

    policy = PPO.load(motion_planner_load_model_path)
    env.assign_policy(policy=policy)
    task_allocator = env.random_allocator
    task_allocator = env.distance_based_allocator
    # time.sleep(5)
    # for i in range(100):
    #     dist_cost = env.reset()
    #     act = task_allocator(obs=dist_cost)
    #     env.allocate_task(act)
    #     time.sleep(1)
    #     done = env._termination()
    #     okay = True
    #     while not done:
    #         okay = env.execute_planner(customize_fragment_length=30)
    #         done = env._termination()
    #         # time.sleep(5)
    #         print(okay)
    #
    #         if env.need_task_allocation:
    #             dist_cost = env.get_states()
    #             # if not okay:
    #             #     act = env.random_allocator(obs=dist_cost)
    #             # else:
    #             #     act = task_allocator(obs=dist_cost)
    #             act = task_allocator(obs=dist_cost)
    #             env.allocate_task(act)
    #         # time .sleep(0.5)
    #     time.sleep(1)

    # for i in range(100):
    #     time.sleep(0.5)
    #     print(env.action_space.sample())
    # time.sleep(10)




    # for i in range(100):
    #     count = 0
    #     obs = env.reset()
    #     for r in env.robots:
    #         print(r.goal)
    #     done = False
    #     while not done:
    #         count += 1
    #         print(count * 20)
    #         print(env.state_info['masks_obs'])
    #         # time.sleep(10)
    #         dist_cost = env.state_info['dist_cost']
    #         act = task_allocator(obs=dist_cost)
    #         # act = env.action_space.sample()
    #
    #         dist_cost, r, done, info = env.step(act)
    #         # time.sleep(1)
    #         print('allocator action:\t', act)
    #         print('rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr:\t', r)
    #         # time.sleep(1)
    #
    #     print("total_step used:\t", count * 20, "\t!!!!!!!!!!")
    #     time.sleep(1)

    #
    t0 = time.time()
    success = 0
    fail = 0
    task_done =[]
    step_count_success = []
    step_count_fail = []
    loop = 10
    for i in range(loop):
        print("loop:\t", i)
        count = 0
        obs = env.reset()
        done = False
        while not done:
            count += 1

            dist_cost = env.state_info['dist_cost']
            act = task_allocator(obs=dist_cost)
            dist_cost, r, done, info = env.step(act)
        if env.success:
            success += 1
            step_count_success.append(count * task_config['fragment_length'])
        if env.fail:
            fail += 1
            step_count_fail.append(count * task_config['fragment_length'])
        task_done.append(sum(env.state_info['parts_success']))

        print("total_step used:\t", count * task_config['fragment_length'] , "\t!!!!!!!!!!")
        print("task done:\t", sum(env.state_info['parts_success']))
        print("time used:\t", time.time() - t0)
    print("success_times:\t", success)
    if success > 0:
        print("ave step:\t", sum(step_count_success)/success)
    print("fail_times:\t", fail)
    if fail > 10:
        print("ave step:\t", sum(step_count_fail)/fail)
    print("ave task done:\t", sum(task_done)/ loop)
    print("time used:\t", time.time() - t0)
