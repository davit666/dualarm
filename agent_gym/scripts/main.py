import os
import torch as th

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import time
from gym_envs.Env_gr import Env_gr
from gym_envs.Env_gr_dict import Env_gr_dict
# from gym_envs.Env_rearrangement import Env_rearrangement
# from gym_envs.Env_asynchronous_gr import Env_asynchronous_gr


# from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from torchinfo import summary

from config import load_config

from train_utils import CallbackEpisodeMetrics, linear_schedule
from model_utils import CustomActorCriticPolicy, CustomCombinedExtractor, CustomLinkExtractor


def make_env(env_config, env_name, renders=False, reward_type=None, obs_type=None, action_type=None, dict_obs=False):
    def _init():
        assert reward_type is not None and obs_type is not None
        print(reward_type, obs_type)
        if env_name == "goal_reaching":
            if dict_obs:
                env = Env_gr_dict(env_config, renders=renders, reward_type=reward_type, obs_type=obs_type,
                                  action_type=action_type)
            else:
                env = Env_gr(env_config, renders=renders, reward_type=reward_type, obs_type=obs_type,
                             action_type=action_type)
        # elif env_name == "rearrangement":
        #     env = Env_rearrangement(env_config, renders=renders, reward_type=reward_type, obs_type=obs_type,
        #                             action_type=action_type)
        else:
            return None
        return env

    return _init


if __name__ == '__main__':

    # Load config
    train_config, env_config = load_config()

    #############################################
    # Create environment
    env_name = train_config['env_name']
    load_model = train_config['load_model']
    load_model_path = train_config['load_model_path']
    num_cpu = train_config['num_cpu']
    reward_type = train_config['reward_type']
    obs_type = train_config['obs_type']
    action_type = train_config['action_type']
    print(reward_type, obs_type)

    custom_network = train_config['custom_network']
    dict_obs = train_config['dict_obs']
    extractor_name = CustomLinkExtractor #CustomCombinedExtractor
    assert not (custom_network and dict_obs)
    if dict_obs:
        policy_name = 'MultiInputPolicy'
        policy_kwargs = dict(features_extractor_class=extractor_name, activation_fn=th.nn.ReLU,
                             net_arch=[dict(pi=[64], vf=[512])])
        # net_arch=[dict(pi=[512, 512], vf=[512, 512])])
    else:
        policy_name = 'MlpPolicy'
        policy_kwargs = dict(activation_fn=th.nn.ReLU,
                             net_arch=[dict(pi=[64, 64], vf=[512, 512])])

    if train_config['use_lr_scheduler']:
        ## use linear scheduler
        lr_scheduler = linear_schedule(train_config['learning_rate'])
    else:
        lr_scheduler = train_config['learning_rate']

    # Instantiate the agent
    if load_model:
        env = SubprocVecEnv(
            [make_env(env_config, env_name, renders=False, reward_type=reward_type, obs_type=obs_type,
                      action_type=action_type,
                      dict_obs=dict_obs) for i in
             range(num_cpu)])
        model = PPO.load(load_model_path)
        model.learning_rate = lr_scheduler
        model.tensorboard_log = train_config['log_path']
        model.set_env(env)
    else:
        if train_config['alg_name'] == "PPO":
            env = SubprocVecEnv(
                [make_env(env_config, env_name, renders=False, reward_type=reward_type, obs_type=obs_type,
                          action_type=action_type,
                          dict_obs=dict_obs) for i in
                 range(num_cpu)])
            if custom_network:
                # model = PPO('MlpPolicy', env, learning_rate=lr_scheduler, verbose=0,
                #             tensorboard_log=train_config['log_path'], n_steps=train_config['n_steps'],
                #             batch_size=train_config['batch_size'], n_epochs=train_config['n_epochs'], policy_kwargs=policy_kwargs)
                model = PPO(CustomActorCriticPolicy, env, learning_rate=lr_scheduler, verbose=0,
                            tensorboard_log=train_config['log_path'], n_steps=train_config['n_steps'],
                            batch_size=train_config['batch_size'], n_epochs=train_config['n_epochs'])
            else:
                model = PPO(policy_name, env, learning_rate=lr_scheduler, verbose=0,
                            tensorboard_log=train_config['log_path'], n_steps=train_config['n_steps'],
                            batch_size=train_config['batch_size'], n_epochs=train_config['n_epochs'],
                            policy_kwargs=policy_kwargs)

        elif train_config['alg_name'] == "A2C":
            env = SubprocVecEnv(
                [make_env(env_config, env_name, renders=False, reward_type=reward_type, obs_type=obs_type,
                          action_type=action_type,
                          dict_obs=dict_obs) for i in
                 range(num_cpu)])
            model = A2C('MlpPolicy', env, learning_rate=lr_scheduler, verbose=0,
                        tensorboard_log=train_config['log_path'])

        elif train_config['alg_name'] == "SAC":
            env = SubprocVecEnv(
                [make_env(env_config, env_name, renders=False, reward_type=reward_type, obs_type=obs_type,
                          action_type=action_type,
                          dict_obs=dict_obs) for i in
                 range(num_cpu)])
            model = SAC('MlpPolicy', env, learning_rate=lr_scheduler, verbose=0,
                        tensorboard_log=train_config['log_path'],
                        batch_size=train_config['batch_size'])

    # Define model name and callbacks
    model_date = train_config['model_date']
    model_name = train_config['model_name']
    model_type = model_date + "/{}_{}_{}_{}_{}".format(train_config['alg_name'], train_config['reward_type'],
                                                       train_config['obs_type'], train_config['action_type'],
                                                       time.strftime("%Y-%m-%d-%H-%M-%S"))
    callback_episode_metrics = CallbackEpisodeMetrics(train_config['save_path'] + model_type,
                                                      train_config['model_name'],
                                                      model_save_freq=train_config['model_save_freq'], num_cpu=num_cpu)

    # Train the agent
    summary(model.policy)
    model.learn(total_timesteps=train_config['total_timesteps'], tb_log_name=model_type,
                callback=callback_episode_metrics)

    # Save the agent
    del model  # delete trained model to demonstrate loading

    ###################################################################################################
    # Load the trained agent
    model = PPO.load(train_config['save_path'] + model_type + '/' + train_config['model_name'] + '_at_the_end')

    # Enjoy trained agent
    env = Env_gr(env_config, renders=True, showBallMarkers=True, reward_type=reward_type, obs_type=obs_type)
    obs = env.reset()
    for i in range(20000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        # env.render()
        if done:
            time.sleep(1)
            env.reset()

    env.close()
