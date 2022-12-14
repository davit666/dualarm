import os
import torch as th

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import time


# from stable_baselines3.common.policies import MlpPolicy

from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
import torch
from torchinfo import summary


from train_utils import CallbackEpisodeMetrics, linear_schedule
from model_utils_task import CustomActorCriticPolicy

from gym_envs.pseudo_Env_tasks import Env_tasks
from config_task import load_config

from prediction_model import Prediction_Model, get_features, NeuralNetwork
from custom_subproc_vec_env import CustomSubprocVecEnv

from model_utils_task import CustomActorCriticPolicy


def make_env(task_config, renders=False):
    def _init():
        task_allocator_reward_type = task_config["task_allocator_reward_type"]
        task_allocator_obs_type = task_config["task_allocator_obs_type"]
        task_allocator_action_type = task_config["task_allocator_action_type"]

        env = Env_tasks(
            task_config,
            renders=renders,
            task_allocator_reward_type=task_allocator_reward_type,
            task_allocator_obs_type=task_allocator_obs_type,
            task_allocator_action_type=task_allocator_action_type,
        )
        return env

    return _init


if __name__ == "__main__":
    task_config = load_config()

    env_name = task_config["env_name"]
    alg_name = task_config["alg_name"]

    reward_type = task_config["task_allocator_reward_type"]
    obs_type = task_config["task_allocator_obs_type"]
    action_type = task_config["task_allocator_action_type"]
    print(reward_type, obs_type, action_type)

    num_cpu = task_config["num_cpu"]

    load_model = task_config["load_model"]
    load_model_path = task_config["load_model_path"]

    ############ load prediction model

    cost_type = task_config["cost_type"]
    obs_type = task_config["obs_type"]

    cost_model_path = task_config["cost_model_path"]
    mask_model_path = task_config["mask_model_path"]
    use_prediction_model = task_config["use_prediction_model"]
    #### load prediction model
    prediction_model = Prediction_Model(obs_type=obs_type, cost_type=cost_type, cost_model_path=cost_model_path, mask_model_path=mask_model_path)
    print("prediction model loaded, input_type:{}, output_type:{}".format(obs_type, cost_type))

    ############ create custom subproc env and mount prediction model
    num_cpu = 1

    env = CustomSubprocVecEnv([make_env(task_config) for i in range(num_cpu)])
    env.load_prediction_model(prediction_model, input_type=obs_type, output_type=cost_type, use_prediction_model=use_prediction_model)

    ########### load trained policy
    policy = PPO.load(load_model_path)

    obs = env.reset()
    z_o = []
    z_a = []
    z_r = []
    z_d = []
    z_i = []
    for i in range(300):
        acts = policy.predict(obs, deterministic=True)[0]
        print("predicted act:")
        print(acts)
        obs, rews, dones, infos = env.step(acts)
        z_o.append(obs)
        z_a.append(acts)
        z_r.append(rews)
        z_d.append(dones)
        z_i.append(infos)
        # print(i,acts)
        if dones[0]:
            break
