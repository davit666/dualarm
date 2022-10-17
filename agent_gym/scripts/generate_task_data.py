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
from stable_baselines3 import PPO, A2C
import gym

from train_utils import CallbackEpisodeMetrics
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from model_utils import CustomActorCriticPolicy
from config import load_config
import numpy as np
from tqdm import tqdm
from train_utils import create_folder


def make_env(env_config, env_name, renders=False, reward_type=None, obs_type=None, action_type=None, dict_obs=False):
    def _init():
        assert reward_type is not None and obs_type is not None and action_type is not None
        print(reward_type, obs_type, action_type)
        if env_name == "goal_reaching":
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

    ###################################################################################################
    # Load the trained agent
    load_model_path = train_config['load_model_path']
    load_model = train_config['load_model']

    reward_type = train_config['reward_type']
    obs_type = train_config['obs_type']
    action_type = train_config['action_type']

    num_cpu = 24  # train_config['num_cpu']
    render = False

    task_date = "1010"
    task_name = "3M_data_24cpu"
    model_name = load_model_path.split("/")[-1]
    start_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    save_path = "../../generated_datas/task_datas/{}/{}/{}/{}/".format(task_date, task_name, model_name, start_time)
    create_folder(save_path)

    policy = PPO.load(load_model_path)

    env_name = train_config['env_name']
    if env_name == "goal_reaching":
        # env = Env_gr(env_config,renders=render, evaluate = False, showBallMarkers = False, reward_type = reward_type, obs_type = obs_type,action_type=action_type,maxSteps=500,use_plot=False)
        env = SubprocVecEnv(
            [make_env(env_config, env_name, renders=render, reward_type=reward_type, obs_type=obs_type,
                      action_type=action_type) for i in range(num_cpu)])
    else:
        env = None

    task_num = 10000
    loop_num = 300
    total_num = task_num * loop_num
    data_size = 84
    time0 = time.time()

    pbar_update_freq = 100
    with tqdm(total=total_num, desc="generating data:  ",) as pbar:
        for l in range(loop_num):

            task_count = 0
            succ_count = 0
            obs = env.reset()
            task_data = np.zeros((task_num, data_size))
            while task_count < task_num:
                action, _states = policy.predict(obs, deterministic=True)
                obs, rewards, dones, infos = env.step(action)

                for index, done in enumerate(dones):
                    if done:
                        if index >= len(infos):
                            continue
                        elif infos[index]['prediction_model_state_data'] is None:
                            continue
                        elif len(infos[index]['prediction_model_state_data']) != data_size:
                            continue

                        task_data[task_count, :] = infos[index]['prediction_model_state_data'][:]
                        task_count += 1
                        succ_count += infos[index]['is_success/all']

                        # print(task_count,index,succ_count)
                        if (task_count + 1) % pbar_update_freq == 0:
                            # print(task_count + 1)
                            pbar.update(pbar_update_freq)
                        if task_count >= task_num:
                            break

            # print(task_data)
            curr_time = time.strftime("%Y-%m-%d-%H-%M-%S")
            np.save(save_path + curr_time, task_data)
            print("\ntask data in loop: {} saved,\t{}\t data in total generated".format((l + 1) ,
                                                                                      (l + 1) * task_num))
            print("saved path:  ",save_path)
            print("time used:\t", time.time() - time0)
            print(" ")

    env.close()
