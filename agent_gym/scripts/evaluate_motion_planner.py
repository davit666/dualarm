import os

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import pybullet as p
import numpy as np
import time
import csv
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


# def make_env(env_config,renders=False,reward_type = None, obs_type = None):
#     def _init():
#         assert reward_type is not None and obs_type is not None
#         print(reward_type,obs_type)
#         env = Env_gr(env_config,renders=renders, reward_type = reward_type, obs_type = obs_type)
#         return env
#     return _init


def evaluate_obj(env, loop=1000, obj_shape_type_list=['random'], save_name='test', save_path=''):
    type_num = len(obj_shape_type_list)

    success_num = np.array([0] * type_num)
    fail_num = np.array([0] * type_num)
    step_success = np.array([0] * type_num)
    robot1_success = np.array([0] * type_num)
    robot2_success = np.array([0] * type_num)
    robot1_motion_eff_success = np.array([0] * type_num)
    robot2_motion_eff_success = np.array([0] * type_num)

    time0 = time.time()

    for k, type in enumerate(obj_shape_type_list):
        env.obj_shape_type = type
        success_count = 0
        for l in range(loop):
            count_while = 0
            obs = env.reset()
            done = False
            while not done:
                count_while += 1
                action, _states = model.predict(obs, deterministic=True)
                obs, rewards, done, info = env.step(action)

            robot1_success[k] += env.robots[0].is_success
            robot2_success[k] += env.robots[1].is_success
            if all(robot.is_success for robot in env.robots):
                success_count += 1
                success_num[k] += 1
                step_success[k] += info['num_steps/num_steps_in_a_episode_when_success']
                robot1_motion_eff_success[k] += info['motion_efficiency/robot_1']
                robot2_motion_eff_success[k] += info['motion_efficiency/robot_2']
            if any(robot.is_failed for robot in env.robots):
                fail_num[k] += 1

            if l % 100 == 0:
                print(save_name, type, l + 1, '\tsuccess rate:\t', success_count / (l + 1),
                      '\ttime used:\t', time.time() - time0)

        step_success[k] /= success_count
        robot1_motion_eff_success[k] /= success_count
        robot2_motion_eff_success[k] /= success_count
    title = ['type', 'loop num', 'success num', 'fail num', 'success rate', 'step when succ', 'robot 1 succ',
             'robot 2 succ',
             'robot 1 me', 'robot 2 me', ]
    with open(save_path + '_obj_eva.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(title)
        for k, type in enumerate(obj_shape_type_list):
            data = [type, loop, success_num[k], fail_num[k], success_num[k] / loop, step_success[k], robot1_success[k],
                    robot2_success[k], robot1_motion_eff_success[k], robot2_motion_eff_success[k]]
            writer.writerow(data)

    return 0


if __name__ == '__main__':

    # Load config
    train_config, env_config = load_config()

    ###################################################################################################
    # Load the trained agent
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
    render = False
    from torchinfo import summary

    # summary(actor)
    # Enjoy trained agent
    env_name = train_config['env_name']
    if env_name == "goal_reaching":
        env = Env_gr(env_config, renders=render, evaluate=False, showBallMarkers=False, reward_type=reward_type,
                     obs_type=obs_type, action_type=action_type, maxSteps=500, use_plot=False)
    elif env_name == "asynchronous_goal_reaching":
        env = Env_asynchronous_gr(env_config, renders=render, reward_type=reward_type, obs_type=obs_type,
                                  action_type=action_type)
    else:
        env = None

    save_name = load_model_path.split('/')[-1]
    type_list = ['random', 'bar', 'disk', 'ball', 'small']
    evaluate_obj(env, loop=10000, obj_shape_type_list=type_list, save_name=save_name, save_path=load_model_path)

    # loop = 100
    # success_num = 0
    # hit_num = 0
    # step_success = []
    # robot1_motion_eff_success = []
    # robot2_motion_eff_success = []
    # robot1_task_num = []
    # robot1_task_ratio = []
    # robot2_task_num = []
    # robot2_task_ratio = []
    #
    # suss_robot1_task_num = []
    # suss_robot1_task_ratio = []
    # suss_robot2_task_num = []
    # suss_robot2_task_ratio = []
    #
    # fail_robot1_task_num = []
    # fail_robot1_task_ratio = []
    # fail_robot2_task_num = []
    # fail_robot2_task_ratio = []
    #
    # time0 = time.time()
    # for k in range(loop):
    #     count_while = 0
    #     obs = env.reset()
    #     done = False
    #     while not done:
    #         count_while += 1
    #         action, _states = model.predict(obs, deterministic=True)
    #         obs, rewards, done, info = env.step(action)
    #
    #
    #     #### save data
    #     robot1_task_num.append(info['reach_num/robot_1'])
    #     robot1_task_ratio.append(info['reach_rate/robot_1'])
    #     robot2_task_num.append(info['reach_num/robot_2'])
    #     robot2_task_ratio.append(info['reach_rate/robot_2'])
    #     if all(robot.is_success for robot in env.robots):
    #         success_num += 1
    #         step_success.append(info['num_steps/num_steps_in_a_episode_when_success'])
    #         robot1_motion_eff_success.append(info['motion_efficiency/robot_1'])
    #         robot2_motion_eff_success.append(info['motion_efficiency/robot_2'])
    #
    #         suss_robot1_task_num.append(info['reach_num/robot_1'])
    #         suss_robot1_task_ratio.append(info['reach_rate/robot_1'])
    #         suss_robot2_task_num.append(info['reach_num/robot_2'])
    #         suss_robot2_task_ratio.append(info['reach_rate/robot_2'])
    #
    #
    #         # first = env.who_reaced_first
    #         # if first ==0:
    #         #     robot1_first += 1
    #         # elif first ==1:
    #         #     robot2_first += 1
    #     else:
    #         fail_robot1_task_num.append(info['reach_num/robot_1'])
    #         fail_robot1_task_ratio.append(info['reach_rate/robot_1'])
    #         fail_robot2_task_num.append(info['reach_num/robot_2'])
    #         fail_robot2_task_ratio.append(info['reach_rate/robot_2'])
    #         if info['is_failed/all']:
    #             hit_num += 1
    #     print("success : {} / {}     time used: ".format(success_num,k + 1), time.time() - time0)
    #     # print(info['motion_efficiency/robot_1'],info['motion_efficiency/robot_2'])
    # print("loop_num: {}".format(loop))
    # print("success_num: {}       success_rate: {}".format(success_num, success_num / loop))
    # print("fail_num: {}       fail_rate: {}".format(hit_num, hit_num / loop))
    # print("ave_step_used_when_success: {}".format(sum(step_success) / success_num))
    # print("robot1 motion_efficiency: {}".format(sum(robot1_motion_eff_success) / len(robot1_motion_eff_success)))
    # print("robot2 motion_efficiency: {}".format(sum(robot2_motion_eff_success) / len(robot2_motion_eff_success)))
    # print("robot1 reach count: {} , robot1 reach rate: {}".format(sum(robot1_task_num)/loop,sum(robot1_task_ratio)/loop))
    # print("robot2 reach count: {} , robot1 reach rate: {}".format(sum(robot2_task_num)/loop,sum(robot2_task_ratio)/loop))
    #
    # print("when suss :robot1 reach count: {} , robot1 reach rate: {}".format(sum(suss_robot1_task_num) / success_num,
    #                                                               sum(suss_robot1_task_ratio) / success_num))
    # print("when suss :robot2 reach count: {} , robot1 reach rate: {}".format(sum(suss_robot2_task_num) / success_num,
    #                                                               sum(suss_robot2_task_ratio) / success_num))
    #
    # print("when fail :robot1 reach count: {} , robot1 reach rate: {}".format(sum(fail_robot1_task_num) / (loop -success_num),
    #                                                                          sum(fail_robot1_task_ratio) / (loop -success_num)))
    # print("when fail :robot2 reach count: {} , robot1 reach rate: {}".format(sum(fail_robot2_task_num) / (loop -success_num),
    #                                                                          sum(fail_robot2_task_ratio) / (loop -success_num)))
    #
    # print("evaluation time used : {}  /  {}".format(time.time() - time0, loop))
