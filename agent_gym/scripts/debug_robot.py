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
import pybullet as p

import os, inspect
import time
from gym_envs.Env_gr import Env_gr
from gym_envs.Env_asynchronous_gr import Env_asynchronous_gr
from gym_envs.Env_gr_dict import Env_gr_dict
# from gym_envs.Env_rearrangement import Env_rearrangement
from gym_envs.Env_tasks import Mode, Env_tasks
from config import load_config
from stable_baselines3 import PPO
import pybullet as p


def create_cube_object(position, orientation=[0, 0, 0, 1], halfExtents=[0.1, 0.1, 0.1], color=[0, 1, 0, 1]):
    vs_id = p.createVisualShape(p.GEOM_BOX, halfExtents=halfExtents, rgbaColor=color)
    coll_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=halfExtents)
    # object_id = p.createMultiBody(basePosition=position, baseOrientation=orientation, baseCollisionShapeIndex=coll_id,
    #                               baseVisualShapeIndex=vs_id)
    object_id = p.createMultiBody(1,coll_id,vs_id)
    p.resetBasePositionAndOrientation(object_id, position, orientation)
    return object_id




if __name__ == "__main__":

    train_config, env_config = load_config()
    # env_config['move_with_obj'] = False
    load_model_path = train_config['load_model_path']
    load_model = train_config['load_model']
    num_cpu = train_config['num_cpu']
    reward_type = train_config['reward_type']
    obs_type = train_config['obs_type']
    action_type = train_config['action_type']

    # env = Env_asynchronous_gr(env_config, renders=True, reward_type=reward_type, obs_type=obs_type)
    env = Env_gr(env_config, renders=True, reward_type=reward_type, obs_type=obs_type, action_type=action_type)

    obs = env.reset()
    # cube = create_cube_object([0,0,1])
    # constrain_cid = p.createConstraint(env.robots[0].robotUid, env.robots[0].robotEndEffectorIndex, cube,
    #                                         -1,
    #                                         p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], [0, 0, 0])
    time.sleep(1)

    for i in range(1000000):

        # p.stepSimulation()
        # done =False
        # time.sleep(1/240.)
        # state = p.getBasePositionAndOrientation(cube)
        # pos, orn = list(state[0]), list(state[1])
        # print(pos,orn)




        action = env.go_straight_planning()
        # print("action:\t", action)
        obs, reward, done, info = env.step(action)


        # print("@@@@@@@@@@")
        # print(env.obs_info["obj_bb_obs_1"],'\n', env.obs_info["obj_bb_obs_2"])
        # print("###############")
        # print(env.obs_info["obj_bounding_box_1"],'\t', env.obs_info["obj_bounding_box_2"])
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print(env.obs_info["obj_center_pose_1"] - env.obs_info["obj_grasp_pose_1"])
        # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        # print(env.obs_info["achieved_goal_1"] - env.obs_info["obj_grasp_pose_1"])
        # print(00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000)


        # # print("observation:\t", obs)
        # # print("reward:\t", reward)
        # # print("state info:")
        # # print(env.obs_info)
        time.sleep(0.1)


        if done:
            print("info:\t", info)
            time.sleep(10)
            env.reset()
