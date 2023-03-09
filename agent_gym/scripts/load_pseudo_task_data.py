import os
import numpy as np
from train_utils import create_folder


# import time
# from gym_envs.pseudo_Env_tasks import Env_tasks

# from prediction_model import Prediction_Model, get_features, NeuralNetwork
# from custom_subproc_vec_env import CustomSubprocVecEnv
# from config_task import load_config



def load_pseudo_task_datas(task_data_path):
    # load task datas
    task_data_file_names = os.listdir(task_data_path)
    task_data_file_num = len(task_data_file_names)

    task_datas = []
    task_data_num = 0
    for name in task_data_file_names:
        task_data = np.load(task_data_path + name)
        task_datas.append(task_data)
        task_data_num += task_data.shape[0]

    task_datas = np.concatenate(task_datas, axis=0)
    return task_datas

def extract_pseudo_task_data(data):
    extracted_data = {}
    robots_data = []
    parts_data = []
    data_shape = data.shape[0]
    p_num = data_shape // (4 * 2) - 2
    idx = 0
    for i in range(2):
        r_data = {}
        r_data['init'] = data[idx:idx + 4]
        r_data['goal'] = data[idx + 4:idx + 8]
        robots_data.append(r_data)
        idx += 8

    for j in range(p_num):
        p_data = {}
        p_data['init'] = data[idx:idx + 4]
        p_data['goal'] = data[idx + 4:idx + 8]
        parts_data.append(p_data)
        idx += 8

    extracted_data['robots'] = robots_data
    extracted_data['parts'] = parts_data
    return extracted_data



# if __name__ == '__main__':

#     task_config = load_config()

#     env_name = task_config['env_name']
#     alg_name = task_config['alg_name']

#     task_allocator_reward_type = task_config["task_allocator_reward_type"]
#     task_allocator_obs_type = task_config["task_allocator_obs_type"]
#     task_allocator_action_type = task_config["task_allocator_action_type"]

#     num_cpu = task_config['num_cpu']
#     parts_num = task_config["part_num"]

#     load_model = task_config['load_model']
#     load_model_path = task_config['load_model_path']

#     ####  load data path
#     task_data_path = "../../generated_datas/pseudo_task_datas/0227/6_part/mix/2023-02-27-11-50-12/"

#     task_datas = load_pseudo_task_datas(task_data_path)
#     # print(task_datas)

#     data = task_datas[0,:]
#     print(data)

#     extracted_task_data = extract_pseudo_task_data(data)
#     for k in extracted_task_data:
#         print(k)
#         for element in extracted_task_data[k]:
#             print(element)

#     env = Env_tasks(
#         task_config,
#         renders=False,
#         task_allocator_reward_type=task_allocator_reward_type,
#         task_allocator_obs_type=task_allocator_obs_type,
#         task_allocator_action_type=task_allocator_action_type,
#     )

#     obs = env.reset(load_task_data = data)

#     info = env.state_info




