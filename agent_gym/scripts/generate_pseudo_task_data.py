import os

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)


import time
from gym_envs.pseudo_Env_tasks import Env_tasks

from prediction_model import Prediction_Model, get_features, NeuralNetwork
from custom_subproc_vec_env import CustomSubprocVecEnv
import gym


from config_task import load_config
import numpy as np
from tqdm import tqdm
from train_utils import create_folder


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
    print(reward_type, obs_type, action_type)

    num_cpu = task_config['num_cpu']
    parts_num = task_config["part_num"]

    load_model = task_config['load_model']
    load_model_path = task_config['load_model_path']

    #### load prediction model
    cost_type = task_config['cost_type']
    obs_type = task_config['obs_type']
    cost_model_path = task_config['cost_model_path']
    mask_model_path = task_config['mask_model_path']
    use_prediction_model = False
    prediction_model = Prediction_Model(obs_type=obs_type, cost_type=cost_type, cost_model_path=cost_model_path,
                                        mask_model_path=mask_model_path) if use_prediction_model else None
    ############ create custom subproc env and mount prediction model
    env = CustomSubprocVecEnv(
        [make_env(task_config) for i in range(num_cpu)])
    env.load_prediction_model(prediction_model, input_type=obs_type, output_type=cost_type,
                              use_prediction_model=use_prediction_model)

    ########## create folder to save data

    task_date = time.strftime("%m%d")
    task_name = "{}_part/mix".format(parts_num)
    start_time = time.strftime("%Y-%m-%d-%H-%M-%S")
    save_path = "../../generated_datas/pseudo_task_datas/{}/{}/{}/".format(task_date, task_name, start_time)
    create_folder(save_path)

    ########## start generation
    task_num = 10000
    loop_num = 10
    total_num = task_num * loop_num
    data_size = (4 * 4) + (4 * 2 * parts_num)
    time0 = time.time()

    pbar_update_freq = 10
    with tqdm(total=total_num, desc="generating data:  ", ) as pbar:
        for l in range(loop_num):
            task_count = 0
            task_data = np.zeros((task_num, data_size))

            while task_count < task_num:

                obs = env.reset()
                data = env.get_current_task_data()
                # print(data)
                shape0 = data.shape[0]
                for k in range(shape0):
                    task_data[task_count,:] = data[k,:]
                    task_count += 1

                    if (task_count + 1) % pbar_update_freq == 0:
                        pbar.update(pbar_update_freq)
                    if task_count >= task_num:
                        break

            curr_time = time.strftime("%Y-%m-%d-%H-%M-%S")
            np.save(save_path + curr_time, task_data)
            print("\ntask data in loop: {} saved,\t{}\t data in total generated".format((l + 1),
                                                                                        (l + 1) * task_num))
            print("saved path:  ", save_path, curr_time)
            print("time used:\t", time.time() - time0)
            print(" ")

    env.close()

