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
from load_pseudo_task_data import load_pseudo_task_datas, extract_pseudo_task_data


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
    part_num = task_config["part_num"]

    load_model = True#task_config["load_model"]
    load_model_path = task_config["load_model_path"]
    loaded_task_datas = None

    ############ load prediction model
    cost_type = task_config["cost_type"]
    obs_type = task_config["obs_type"]

    cost_model_path = task_config["cost_model_path"]
    mask_model_path = task_config["mask_model_path"]
    use_prediction_model = task_config["use_prediction_model"]
    #### load prediction model
    prediction_model = (
        Prediction_Model(obs_type=obs_type, cost_type=cost_type, cost_model_path=cost_model_path,
                         mask_model_path=mask_model_path)
        if use_prediction_model
        else None
    )
    print("prediction model loaded, input_type:{}, output_type:{}".format(obs_type, cost_type))

    ############ create custom subproc env and mount prediction model
    num_cpu = 1

    env = CustomSubprocVecEnv([make_env(task_config) for i in range(num_cpu)])
    env.load_prediction_model(prediction_model, input_type=obs_type, output_type=cost_type,
                              use_prediction_model=use_prediction_model, predict_content=task_config['predict_content'])

    # if not use_prediction_model:
    #     del prediction_model

    ########### load trained policy
    if load_model:
        policy = PPO.load(load_model_path)
        # policy.set_env(env)
        policy.policy.observation_space = env.observation_space
        policy.policy.action_space = env.action_space
        policy.policy.mlp_extractor.mask_dim = task_config["part_num"] + 1
    ################## evaluate many episodes and get average
    from gym_envs.baselines_pseudo_env_tasks import baselines_offline_heuristics, baseline_offline_brute_force, \
        baseline_online_MCTS, calcul_cost

    use_baseline = "min_cost_sample"  # "min_cost_sample"
    baseline_function = baseline_online_MCTS

    #######
    if part_num == 6:
        task_data_path = "../../../../generated_datas/pseudo_task_datas/0227/6_part/mix/2023-02-27-11-50-12/"
    elif part_num == 8:
        task_data_path = "../../../../generated_datas/pseudo_task_datas/0227/8_part/mix/2023-02-27-12-14-36/"
    elif part_num == 10:
        task_data_path = "../../../../generated_datas/pseudo_task_datas/0227/10_part/mix/2023-02-27-12-20-08/"
    elif part_num == 12:
        task_data_path = "../../../../generated_datas/pseudo_task_datas/0227/12_part/mix/2023-02-27-12-21-12/"
    elif part_num == 20:
        task_data_path = "../../../../generated_datas/pseudo_task_datas/0227/20_part/mix/2023-02-27-12-23-03/"
    else:
        task_data_path = "../../../../generated_datas/pseudo_task_datas/0227/6_part/mix/2023-02-27-11-50-12/"

    loaded_task_datas = load_pseudo_task_datas(task_data_path)
    #######
    loop = 10000
    succ_num = 0
    early_finish_num = 0
    step_early_finish = []
    step_succ = []
    acc_cost = []
    ave_cost = []
    global_acc_cost = []
    global_ave_cost = []
    acc_r = []
    ave_r = []
    task_num = []
    early_finish_task_num = []
    robot1_task_num = []
    robot2_task_num = []
    wrong_allocation_rate = []
    time0 = time.time()
    for l in range(loop):
        if loaded_task_datas is not None:
            start_idx = l * num_cpu
            end_idx = (l + 1) * num_cpu
            task_data = loaded_task_datas[start_idx:end_idx, :]

            obs = env.reset_with_task_data(load_task_data=task_data)
        else:
            obs = env.reset()
        count_while = 0
        done = False
        while not done:
            count_while += 1
            if load_model:
                # print(1)
                action = policy.predict(obs, deterministic=True)[0]
            elif use_baseline == "random_sample":
                action = env.sample_action()
            elif use_baseline == "min_cost_sample":
                action = env.sample_action()
            elif use_baseline == "baselines":
                d = env.get_data_for_offline_planning()[0]
                c, m, p = d["c"], d["m"], d["p"]
                costx, best_order, t = baseline_function(c, m, p)
                # print(best_order)
                if len(best_order) >= 2:
                    action = best_order[0] * (part_num + 1) + best_order[1]
                else:
                    action = (part_num + 1) ** 2 - 1
            obss, rewards, dones, infos = env.step([action])
            obs = obss
            reward = rewards[0]
            done = dones[0]
            info = infos[0]

        acc_r.append(info["3_reward/accumulated_reward"])
        ave_r.append(info["3_reward/average_reward"])
        task_num.append(info["4_succ_task_num/task_num"])
        robot1_task_num.append(info["5_robot_info/robot1_task"])
        robot2_task_num.append(info["5_robot_info/robot2_task"])
        wrong_allocation_rate.append(info["5_robot_info/robot1_wrong_allocation"] / info["1_num_steps"])
        global_acc_cost.append(info["2_cost/accumulated_cost"])
        global_ave_cost.append(info["2_cost/average_cost"])
        s = info["6_global_success"]
        if s:
            succ_num += 1
            step_succ.append(info["1_num_steps"])
            acc_cost.append(info["2_cost/accumulated_cost"])
            ave_cost.append(info["2_cost/average_cost"])
        elif info["1_num_steps"] < 50:
            early_finish_num += 1
            step_early_finish.append(info["1_num_steps"])
            early_finish_task_num.append(info["4_succ_task_num/task_num"])

        if l % (loop // 100) == 0:
            print(l // (loop // 100), "\t", time.time() - time0)
            print("suss:\t", succ_num / max(l, 1))
            print("cost:\t", sum(acc_cost) / max(succ_num, 1))

    eva_data = {}

    if use_baseline is None:
        name = load_model_path
        eva_data["policy_name"] = name
    else:
        eva_data["policy_name"] = use_baseline
    eva_data["loop_num"] = loop
    eva_data["succ_rate"] = succ_num / loop
    # eva_data["early_finish_rate"] = early_finish_num / loop
    eva_data["ave_succ_step"] = sum(step_succ) / succ_num
    # eva_data["ave_early_finish_step"] = sum(step_early_finish) / early_finish_num if early_finish_num > 0 else 0
    eva_data["ave_succ_acc_cost"] = sum(acc_cost) / succ_num
    eva_data["ave_succ_ave_cost"] = sum(ave_cost) / succ_num
    eva_data["ave_global_acc_cost"] = sum(global_acc_cost) / loop
    eva_data["ave_global_ave_cost"] = sum(global_ave_cost) / loop
    eva_data["ave_acc_reward"] = sum(acc_r) / loop
    eva_data["ave_ave_reward"] = sum(ave_r) / loop
    eva_data["ave_task_done"] = sum(task_num) / loop
    # eva_data["ave_early_finish_task_done"] = sum(early_finish_task_num) / early_finish_num if early_finish_num > 0 else 0
    eva_data["ave_r1_task_done"] = sum(robot1_task_num) / loop
    eva_data["ave_r2_task_done"] = sum(robot2_task_num) / loop
    eva_data["ave_wrong_allocation_rate"] = sum(wrong_allocation_rate) / loop

    for k, v in eva_data.items():
        print(k, ":\t", v)
    print("time used:\t", time.time() - time0)

    # ################## evaluate single episode
    # obs = env.reset()
    # z_o = []
    # z_a = []
    # z_r = []
    # z_d = []
    # z_i = []
    # for i in range(300):
    #     acts = policy.predict(obs, deterministic=True)[0]
    #     print("predicted act:")
    #     print(acts)
    #     obs, rews, dones, infos = env.step(acts)
    #     z_o.append(obs)
    #     z_a.append(acts)
    #     z_r.append(rews)
    #     z_d.append(dones)
    #     z_i.append(infos)
    #     # print(i,acts)
    #     if dones[0]:
    #         break
