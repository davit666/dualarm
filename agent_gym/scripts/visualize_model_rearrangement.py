import os

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import pybullet as p
import time
from gym_envs.Env_rearrangement import Env_rearrangement

# from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO,A2C
import gym

from train_utils import CallbackEpisodeMetrics
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from config import load_config
def make_env(env_config,renders=False,reward_type = None, obs_type = None):
    def _init():
        assert reward_type is not None and obs_type is not None
        print(reward_type,obs_type)
        env = Env_rearrangement(env_config,renders=renders, reward_type = reward_type, obs_type = obs_type)
        return env
    return _init


if __name__ == '__main__':
    
    # Load config
    train_config,env_config = load_config()



###################################################################################################
    #Load the trained agent
    model = PPO.load('test_models/rearrangement/1203')
    # model = A2C.load('test_models/1018/A2C1018_2')


    reward_type = train_config['reward_type']
    obs_type = train_config['obs_type']
    # Enjoy trained agent
    env = env = Env_rearrangement(env_config,renders=True, showBallMarkers = False, reward_type = reward_type, obs_type = obs_type,maxSteps=1000)
    # time.sleep(25)
    obs = env.reset()
    # time.sleep(10)
    
    # print(range(p.getNumBodies()))

    # time.sleep(30)
    for i in range(20000):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        # print(rewards)
        # time.sleep(0.1)
        # env.render()
        if done:

            for j in range(10):
                p.stepSimulation()
            time.sleep(1)

            env.reset()
    # for i in range(20000):
    #     action, _states = model.predict(obs)
    #     obs, rewards, done, info = env.step(action)
    #     print(rewards)
    #     # time.sleep(0.1)
    #     # env.render()
    #     for robot in env.robots:
    #         if robot.is_success is True:
    #             time.sleep(1)
    #             env.reset_robot(robot)
    #     if done:
    #         time.sleep(1)
    #         env.reset()           


    env.close()