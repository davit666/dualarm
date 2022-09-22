import os

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import pybullet as p
import time
from gym_envs.Env_gr import Env_gr
# from gym_envs.Env_asynchronous_gr import Env_asynchronous_gr


# from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO,A2C
import gym

from train_utils import CallbackEpisodeMetrics
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from model_utils import CustomActorCriticPolicy
from config import load_config


def make_env(env_config,renders=False,reward_type = None, obs_type = None):
    def _init():
        assert reward_type is not None and obs_type is not None
        print(reward_type,obs_type)
        env = Env_gr(env_config,renders=renders, reward_type = reward_type, obs_type = obs_type)
        return env
    return _init


if __name__ == '__main__':
    
    # Load config
    train_config,env_config = load_config()



###################################################################################################
    #Load the trained agent
    load_model_path = train_config['load_model_path']
    load_model = train_config['load_model']
    num_cpu = train_config['num_cpu']
    reward_type = train_config['reward_type']
    obs_type = train_config['obs_type']
    action_type = train_config['action_type']
    # model = PPO(CustomActorCriticPolicy, env, learning_rate=lr_scheduler, verbose=0,
    #             tensorboard_log=train_config['log_path'], n_steps=train_config['n_steps'],
    #             batch_size=train_config['batch_size'], n_epochs=train_config['n_epochs'])

    # policy = PPO.load(load_model_path).policy ### custom
    model = PPO.load(load_model_path)
    render = True
    from torchinfo import summary
    # summary(actor)
    # Enjoy trained agent
    env_name = train_config['env_name']
    if env_name == "goal_reaching":
        env = Env_gr(env_config,renders=render, evaluate = False, showBallMarkers = False, reward_type = reward_type, obs_type = obs_type,action_type=action_type,maxSteps=500,use_plot=False)
    # elif env_name == "asynchronous_goal_reaching":
    #     env = Env_asynchronous_gr(env_config, renders=render, reward_type=reward_type, obs_type=obs_type,action_type=action_type)
    else:
        env = None

    # time.sleep(10)
    obs = env.reset()
    # time.sleep(80)
    # print(range(p.getNumBodies()))

    # time.sleep(5)
    #
    for i in range(20000):
        action, _states = model.predict(obs, deterministic=True)

        # policy = model.policy
        # observation, vectorized_env = policy.obs_to_tensor(obs)
        # features = policy.extract_features(observation)
        # latent_pi = policy.mlp_extractor.forward(features)[0]
        # distribution = policy._get_action_dist_from_latent(latent_pi)
        # action2 = distribution.get_actions(deterministic=True).cpu().detach().numpy()[0]


        obs, rewards, done, info = env.step(action,scale_action=True)
        # print(rewards)
        # time.sleep(0.1)
        # env.render()

        if done:
            time.sleep(1)
            env.reset()
    #
    # for i in range(20000):
    #     # action, _states = policy.predict(obs)
    #     observation, vectorized_env = policy.obs_to_tensor(obs)
    #     features = policy.extract_features(observation)
    #     latent_pi = policy.mlp_extractor.forward(features)[0]
    #     distribution = policy._get_action_dist_from_latent(latent_pi)
    #     action = distribution.get_actions(deterministic=True).cpu().detach().numpy()[0]
    #     # print(action)
    #
    #
    #
    #     obs, rewards, done, info = env.step(action)
    #     # print(rewards)
    #     # time.sleep(0.1)
    #     # env.render()
    #     if done:
    #         time.sleep(1)
    #         env.reset()

    env.close()


