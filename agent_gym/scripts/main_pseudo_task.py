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


if __name__ == '__main__':
    task_config = load_config()

    env_name = task_config['env_name']
    alg_name = task_config['alg_name']

    reward_type = task_config['task_allocator_reward_type']
    obs_type = task_config['task_allocator_obs_type']
    action_type = task_config['task_allocator_action_type']
    print(reward_type, obs_type,action_type)

    num_cpu = task_config['num_cpu']

    load_model = task_config['load_model']
    load_model_path = task_config['load_model_path']

    ############ load prediction model

    cost_type = task_config['cost_type']
    obs_type = task_config['obs_type']

    cost_model_path = task_config['cost_model_path']
    mask_model_path = task_config['mask_model_path']

    use_prediction_model = task_config['use_prediction_model']
    #### load prediction model
    prediction_model = Prediction_Model(obs_type=obs_type, cost_type=cost_type, cost_model_path=cost_model_path,
                                        mask_model_path=mask_model_path)
    print("prediction model loaded, input_type:{}, output_type:{}".format(obs_type,cost_type))


    ############ create custom subproc env and mount prediction model

    env = CustomSubprocVecEnv(
        [make_env(task_config) for i in range(num_cpu)])
    env.load_prediction_model(prediction_model, input_type=obs_type, output_type=cost_type,use_prediction_model=use_prediction_model)

    ########### define learning rate scheduler
    if task_config['use_lr_scheduler']:
        ## use linear scheduler
        lr_scheduler = linear_schedule(task_config['learning_rate'])
    else:
        lr_scheduler = task_config['learning_rate']



    if alg_name == "PPO":
        model = PPO(CustomActorCriticPolicy, env, learning_rate=lr_scheduler, verbose=0,
                    tensorboard_log=task_config['log_path'], n_steps=task_config['n_steps'],
                    batch_size=task_config['batch_size'], n_epochs=task_config['n_epochs'])
    else:
        pass

    # check model sturcture
    summary(model.policy)

    # Define model name and callbacks
    model_date = task_config['model_date']
    model_name = task_config['model_name']
    custom_network_type = task_config['custom_network_type']
    model_type = model_date + "/{}/{}/{}_{}_{}_{}_{}".format(model_name, custom_network_type,alg_name,
                                                       reward_type, obs_type,action_type,
                                                       time.strftime("%Y-%m-%d-%H-%M-%S"))
    model_save_path = task_config['save_path']
    model_save_freq = task_config['model_save_freq']
    callback_episode_metrics = CallbackEpisodeMetrics(model_save_path + model_type,
                                                      "model", model_save_freq=model_save_freq,
                                                      num_cpu=num_cpu)

    # Train the agent

    model.learn(total_timesteps=task_config['total_timesteps'], tb_log_name=model_type,
                callback=callback_episode_metrics)

    # Save the agent
    del model  # delete trained model to demonstrate loading




























