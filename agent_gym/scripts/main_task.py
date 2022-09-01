import os
import torch as th

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import time
from gym_envs.Env_gr import Env_gr
from gym_envs.Env_gr_dict import Env_gr_dict
from gym_envs.Env_tasks import Mode, Env_tasks
from config import load_config

# from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from torchinfo import summary

from config import load_config, load_task_config

from train_utils import CallbackEpisodeMetrics, linear_schedule
from model_utils_task import CustomActorCriticPolicy


def make_env(env_config, task_config, policy=None, renders=False):
    def _init():
        if task_config['dict_obs']:
            pass
        else:
            env = Env_tasks(env_config, renders=renders, useInverseKinematics=True, freezeAction=False,
                            showBallMarkers=False, maxSteps=task_config['maxSteps'],
                            action_type=task_config['action_type'],
                            task_allocator_reward_type=task_config['task_allocator_reward_type'],
                            task_allocator_obs_type=task_config['task_allocator_obs_type'],
                            motion_planner_reward_type=task_config['motion_planner_reward_type'],
                            motion_planner_obs_type=task_config['motion_planner_obs_type'],
                            fragment_length=task_config['fragment_length'], parts_num=task_config['parts_num'])
        if policy is not None:
            # if task_config['selected_motion_planner_policy']:
            #     motion_planner_load_model_path = task_config['motion_planner_load_model_path']
            #     policy = PPO.load(motion_planner_load_model_path)
            env.assign_policy(policy=policy)
        return env

    return _init


if __name__ == '__main__':
    # import torch
    # torch.cuda.set_per_process_memory_fraction(0.5, 0)
    # torch.cuda.empty_cache()
    # total_memory = torch.cuda.get_device_properties(0).total_memory

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
    #### load motion planner
    if task_config['selected_motion_planner_policy']:
        motion_planner_load_model_path = task_config['motion_planner_load_model_path']
        policy = PPO.load(motion_planner_load_model_path)
    else:
        policy = None

    #### choose lr scheduler
    if task_config['use_lr_scheduler']:
        ## use linear scheduler
        lr_scheduler = linear_schedule(task_config['learning_rate'])
    else:
        lr_scheduler = task_config['learning_rate']

    #### choose if to use dictionary type observation
    if dict_obs:
        pass
        # policy_name = 'MultiInputPolicy'
        # policy_kwargs = dict(features_extractor_class=CustomCombinedExtractor, activation_fn=th.nn.ReLU,
        #                      net_arch=[dict(pi=[512], vf=[512])])
    else:
        policy_kwargs = dict(activation_fn=th.nn.ReLU,
                             net_arch=[dict(pi=[512, 512], vf=[512, 512])])

    #### load env
    env = SubprocVecEnv(
        [make_env(env_config, task_config, policy=policy) for i in range(num_cpu)])
    #### choose algorithm
    if alg_name == "PPO":
        model = PPO(CustomActorCriticPolicy, env, learning_rate=lr_scheduler, verbose=0,
                    tensorboard_log=task_config['log_path'], n_steps=task_config['n_steps'],
                    batch_size=task_config['batch_size'], n_epochs=task_config['n_epochs'])
    elif alg_name == "SAC":
        model = SAC(CustomActorCriticPolicy, env, learning_rate=lr_scheduler, verbose=0,
                    tensorboard_log=task_config['log_path'], batch_size=train_config['batch_size'],
                    buffer_size=task_config['buffer_size'])
    else:
        pass

    # check model sturcture
    summary(model.policy)

    # Define model name and callbacks
    model_date = task_config['model_date']
    model_name = task_config['model_name']
    model_type = model_date + "/{}/{}_{}_{}_{}".format(model_name, alg_name,
                                                       reward_type, obs_type,
                                                       time.strftime("%Y-%m-%d-%H-%M-%S"))
    model_save_path = task_config['save_path']
    model_save_freq = task_config['model_save_freq']
    callback_episode_metrics = CallbackEpisodeMetrics(model_save_path + model_type,
                                                      model_name, model_save_freq=model_save_freq,
                                                      num_cpu=num_cpu)

    # Train the agent

    model.learn(total_timesteps=task_config['total_timesteps'], tb_log_name=model_type,
                callback=callback_episode_metrics)

    # Save the agent
    del model  # delete trained model to demonstrate loading

    ###################################################################################################
