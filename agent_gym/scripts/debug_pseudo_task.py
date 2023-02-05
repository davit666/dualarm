import os

currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import pybullet as p

from os.path import exists, basename, abspath, dirname
from time import time

# from copy import copy
# from json import dump
# from tqdm import tqdm
import numpy as np
import pybullet_data
import pybullet as p

import os, inspect
import time
from gym_envs.pseudo_Env_tasks import Env_tasks

from config_task import load_config
from stable_baselines3 import PPO
import pybullet as p

import torch
from torch import nn

from prediction_model import Prediction_Model, get_features, NeuralNetwork
from custom_subproc_vec_env import CustomSubprocVecEnv

from model_utils_task import (
    CustomFeatureExtractor,
    CustomNetwork_FlattenNodes,
    CustomNetwork_FlattenNodesAndEdges,
    make_custom_proba_distribution,
)


def make_env(task_config, renders=True):
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
    ## set gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    task_config = load_config()

    # task_config["task_allocator_action_type"] = "Discrete"

    task_allocator_reward_type = task_config["task_allocator_reward_type"]
    task_allocator_obs_type = task_config["task_allocator_obs_type"]
    task_allocator_action_type = task_config["task_allocator_action_type"]

    use_prediction_model = task_config["use_prediction_model"]

    # # #### load env
    num_cpu = 1

    env = CustomSubprocVecEnv([make_env(task_config, renders=False) for i in range(num_cpu)])
    #### define input output type to use
    cost_type = "coord_steps"  # "coord_steps"
    obs_type = "norm_ee_only"
    #### set loading path of prediction model
    cost_model_path = "../../generated_datas/good_models/cost/1203/1010_task_datas_100_epochs/norm_ee_only_succ_only_predict_step/512-512-512_ce_adam_rl1e-3_batch_512/2022-12-02-13-13-04/model_saved/2022-12-02-14-04-40.pth"
    mask_model_path = "../../generated_datas/good_models/mask/1203/1024_with_failure_task_datas_100_epochs/norm_ee_only_predict_succ/256-256-256_ce_adam_rl1e-3_batch_512/2022-12-02-11-47-26/model_saved/2022-12-02-12-29-48.pth"

    print("!!!!!!!!!!!!!!!!!!!!!!!!!1")
    #### load prediction model
    prediction_model = (
        Prediction_Model(obs_type=obs_type, cost_type=cost_type, cost_model_path=cost_model_path, mask_model_path=mask_model_path)
        if use_prediction_model
        else None
    )
    # input_features, cost_features, mask_features = prediction_model.get_input_and_output()

    # input_shape = len(input_features)
    # cost_shape = len(cost_features)
    # mask_shape = len(mask_features)

    ######################## test a full episode with prediction
    use_prediction_model = task_config["use_prediction_model"]
    env.load_prediction_model(prediction_model, input_type=obs_type, output_type=cost_type, use_prediction_model=use_prediction_model)

    obs = env.reset()
    z_o = []
    z_a = []
    z_r = []
    z_d = []
    z_i = []
    for i in range(100):
        acts = env.sample_action()
        print(acts)
        obs, rews, dones, infos = env.step(acts)

        z_o.append(obs)
        z_a.append(acts)
        z_r.append(rews)
        z_d.append(dones)
        z_i.append(infos)
        # print(i,acts)
        if dones[0]:
            break

    ################ test feature extractor
    # from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor

    # # feature extractor
    # obs_space = env.observation_space
    # feature_extractor = CustomFeatureExtractor(obs_space)
    # print("feature extractor:\n", feature_extractor.extractors)

    # # custom net flatten nodes
    # from model_utils_task import (
    #     CustomNetwork_FlattenNodesAndEdges,
    #     CustomNetwork_SelfAttention,
    #     CustomNetwork_SelfCrossAttention,
    #     CustomNetwork_SelfAttentionWithTaskEdge,
    #     CustomNetwork_SelfCrossAttentionWithEdge,
    # )

    # node_feature_dim = feature_extractor._features_dim
    # mask_dim = feature_extractor.mask_dim

    # # distribution
    # action_space = env.action_space
    # action_dist, discrete_action_space = make_custom_proba_distribution(action_space)

    # # network
    # # net_fn = CustomNetwork_FlattenNodes(node_feature_dim, mask_dim, discrete_action_space=discrete_action_space)
    # # net_fn = CustomNetwork_FlattenNodesAndEdges(node_feature_dim, mask_dim, discrete_action_space=discrete_action_space)
    # # net_fn = CustomNetwork_SelfAttention(node_feature_dim, mask_dim, discrete_action_space=discrete_action_space)
    # # net_fn = CustomNetwork_SelfCrossAttention(node_feature_dim, mask_dim, discrete_action_space=discrete_action_space)
    # # net_fn = CustomNetwork_SelfAttentionWithTaskEdge(node_feature_dim, mask_dim, discrete_action_space=discrete_action_space)
    # net_fn = CustomNetwork_SelfCrossAttentionWithEdge(node_feature_dim, mask_dim, discrete_action_space=discrete_action_space)
    # print("flatten network:\n", net_fn)

    # obs = env.reset()
    # actions = env.sample_action()
    # obs, rews, dones, infos = env.step(actions)
    # observation = {}
    # for key, _ in obs.items():
    #     # print(obs[key])
    #     obs_ = np.array(obs[key])
    #     obs_ = obs_.reshape((-1,) + obs_space[key].shape)
    #     observation[key] = torch.as_tensor(obs_).to("cpu")

    # # observation = obs_as_tensor(observation,'cpu')

    # encoded_obs = feature_extractor.forward(observation)

    # for k, v in encoded_obs.items():
    #     print(k)
    #     print(v.detach().numpy().shape)

    # pi_logits, vf_logits = net_fn.forward(encoded_obs)

    # results = {}
    # for key, _ in encoded_obs.items():
    #     results[key] = encoded_obs[key].cpu().detach().numpy()

    # distribution = action_dist.proba_distribution(action_logits=pi_logits)
    # actions = distribution.get_actions(deterministic=True)

    # actions = actions.cpu().detach().numpy()
    # pi_logits, vf_logits = pi_logits.cpu().detach().numpy(), vf_logits.cpu().detach().numpy()

    # obs2, rews, dones, infos = env.step(actions)

    # ################# test action sample

    # env0 = Env_tasks(
    #     task_config,
    #     renders=False,
    #     task_allocator_reward_type=task_allocator_reward_type,
    #     task_allocator_obs_type=task_allocator_obs_type,
    #     task_allocator_action_type=task_allocator_action_type,
    # )

    # act1 = env0.action_space.sample()
    # act2 = env.sample_action()

    # ################

    #################################################################################### test custom subproc env
    # train_config, env_config = load_config()
    # num_cpu = 1
    #
    # env = CustomSubprocVecEnv(
    #     [make_env(env_config) for i in range(num_cpu)])
    #
    # env.load_prediction_model(prediction_model, input_type=obs_type, output_type=cost_type)
    #
    # obs = env.reset()
    # acts = env.sample_action()
    # print("now in phase 1")
    # # print("acts",acts)
    # print("##########")
    # obs2, rews, dones, infos = env.step(acts)

    ##################################################################################### test prediction model

    # train_config, env_config = load_config()
    #
    # env = Env_tasks(env_config)
    #
    # obs = env.reset()
    #
    # state_info = obs

    # prediction_inputs = state_info["prediction_inputs"]
    # prediction_inputs_shape = prediction_inputs.shape
    #
    # assert prediction_inputs_shape[-1] == input_shape
    #
    # X = torch.tensor(prediction_inputs).to(device)
    # pred_cost = prediction_model.predict_cost(X).cpu().detach().numpy()
    # pred_mask = prediction_model.predict_mask(X).cpu().detach().numpy()
    #
    # pred_cost = pred_cost.reshape((2, env.parts_num + 1, env.parts_num + 1))
    # pred_mask = pred_mask.reshape((2, env.parts_num + 1, env.parts_num + 1))
    #
    # pred_cost1 = np.sum(pred_cost, axis=0)
    # pred_mask1 = np.multiply(pred_mask[0, :, :], pred_mask[1, :, :]).reshape((env.parts_num + 1, env.parts_num + 1))
    #
    # unpred_mask = state_info["coop_edge_mask"]

    # for k in range(50):
    #
    #     env.prediction_updated = True
    #
    #     action = sample_action(env)
    #
    #     o, r, d, info = env.step(action)
    #     if d:
    #         break
    #
    # state_info = env.state_info

    # for k in state_info.keys():
    #     print(k)
    #     print(state_info[k])
    #     print("\n")

    ######################### test plot
    # task_config["use_prediction_model"] = False
    # from gym_envs.baselines_pseudo_env_tasks import baselines_offline_heuristics, baseline_offline_brute_force, baseline_online_MCTS, calcul_cost

    # env = Env_tasks(
    #     task_config,
    #     renders=False,
    #     task_allocator_reward_type=task_allocator_reward_type,
    #     task_allocator_obs_type=task_allocator_obs_type,
    #     task_allocator_action_type=task_allocator_action_type,
    # )

    # prediction_model = (
    #     Prediction_Model(obs_type=obs_type, cost_type=cost_type, cost_model_path=cost_model_path, mask_model_path=mask_model_path)
    #     if task_config["use_prediction_model"]
    #     else None
    # )

    # time0 = time.time()
    # parts_num = task_config["part_num"]

    #### test prediction
    # obs = env.reset()
    # action = env.sample_action()
    # obs, r, done, info = env.step(action)
    # if task_config["use_prediction_model"]:
    #     f, m0, p = env.get_data_for_offline_planning()
    #     c, m = prediction_model.predict_data_for_offline_planning(f, m0)
    # else:
    #     c, m, p = env.get_data_for_offline_planning()

    ####
    # obs = env.reset()
    # c, m, p = env.get_data_for_offline_planning()
    # baseline_function = baseline_online_MCTS
    # # print("offline heuristic")
    # # cost_GT, best_order_GT, t_GT = baseline_offline_brute_force(c, m, p)
    # cost, best_order, t = baseline_function(c, m, p)

    # cost2 = calcul_cost(c, m, p, best_order, show=True)

    # while len(best_order) >= 2:
    #     a = None
    #     for p in best_order:
    #         if a is None:
    #             a = int(p)
    #         else:
    #             print("a:\t", int(a), int(p))
    #             action = a * (parts_num + 1) + int(p)
    #             action2 = env.sample_action()
    #             print("a2:\t", action2[0] // (parts_num + 1), action2[0] % (parts_num + 1))
    #             o, r, done, info = env.step(action)
    #             a = None
    #             c, m, p = env.get_data_for_offline_planning()
    #             costx, best_order, t = baseline_function(c, m, p)
    #             break
    # # a = None
    # # for p in best_order:
    # #     if a is None:
    # #         a = int(p)
    # #     else:
    # #         print("a:\t", int(a), int(p))
    # #         action = a * (parts_num + 1) + int(p)
    # #         action2 = env.sample_action()
    # #         print("a2:\t", action2[0] // (parts_num + 1), action2[0] % (parts_num + 1))
    # #         o, r, done, info = env.step(action)
    # #         a = None

    # action = (parts_num + 1) ** 2 - 1
    # o, r, done, info = env.step(action)

    # planner_cost = info["2_cost/accumulated_cost"]
    # print("search cost:\t", cost)
    # print("recalcul cost:\t", cost2)
    # print("execution cost:\t", planner_cost)
    # # print("GT cost:\t", cost_GT)
    ####
    # obs = env.reset()
    # c, m, p = env.get_data_for_offline_planning()
    # cost, best_order, t = baseline_online_MCTS(c, m, p, step=8)
    # a = None
    # for p in best_order:
    #     if a is None:
    #         a = int(p)
    #     else:

    #         action = a * (parts_num + 1) + int(p)
    #         o, r, done, info = env.step(action)
    #         a = None
    # action = (parts_num + 1) ** 2 - 1
    # o, r, done, info = env.step(action)

    # planner_cost = info["2_cost/accumulated_cost"]
    # print("recalcul cost:\t", planner_cost)
    #####################################
    # loop = 20000
    # cost_heuristics = [0]
    # cost_MCTS_6 = [0]
    # cost_MCTS_2 = [0]
    # cost_brute_force = [0]
    # cost_min_cost = [0]
    # cost_min_cost_recal = [0]

    # suss_heuristics = [0]
    # suss_MCTS_6 = [0]
    # suss_MCTS_2 = [0]
    # suss_brute_force = [0]
    # suss_min_cost = [0]
    # suss_min_cost_recal = [0]

    # num_heuristics = [0]
    # num_MCTS_6 = [0]
    # num_MCTS_2 = [0]
    # num_brute_force = [0]
    # num_min_cost = [0]
    # num_min_cost_recal = [0]

    # time_heuristics = [0]
    # time_MCTS_6 = [0]
    # time_MCTS_2 = [0]
    # time_brute_force = [0]
    # time_min_cost = [0]
    # time_collect_cost_and_mask = [0]

    # for l in range(loop):
    #     time00 = time.time()
    #     obs = env.reset()
    #     if not task_config["use_prediction_model"]:
    #         c, m, p = env.get_data_for_offline_planning()
    #     else:
    #         f, m0, p = env.get_data_for_offline_planning()
    #         c, m = prediction_model.predict_data_for_offline_planning(f, m0)
    #     time_collect_cost_and_mask.append(time.time() - time00)
    #     # c_s2n = c["n2n"][-1, :-1, -1, :-1]
    #     # c = c_s2n + c["n"]
    #     # obs_c = obs["coop_edge_cost"][:-1, :-1]
    #     # check = obs_c - c
    #     print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    #     print("loop:\t", l + 1)
    #     print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")

    #     # print("\n##############\n")
    #     # print("offline heuristic")
    #     # cost, best_order, t = baselines_offline_heuristics(c, m, p)

    #     # time_heuristics.append(t)
    #     # num_heuristics.append(len(best_order))

    #     # cost0 = calcul_cost(c, m, p, best_order)
    #     # print("recal:\t", cost0)

    #     # if cost0 < 100:
    #     #     suss_heuristics.append(1)
    #     #     cost_heuristics.append(cost)
    #     # else:
    #     #     suss_heuristics.append(0)

    #     # print("\n##############\n")
    #     # print("offline brute force")
    #     # cost, best_order, t = baseline_offline_brute_force(c, m, p)
    #     # time_brute_force.append(t)
    #     # num_brute_force.append(len(best_order))

    #     # cost0 = calcul_cost(c, m, p, best_order)
    #     # print("recal:\t", cost0)

    #     # if cost0 < 100:
    #     #     suss_brute_force.append(1)
    #     #     cost_brute_force.append(cost)
    #     # else:
    #     #     suss_brute_force.append(0)

    #     print("\n##############\n")
    #     print("online MCTS")
    #     cost, best_order, t = baseline_online_MCTS(c, m, p)
    #     time_MCTS_6.append(t)
    #     num_MCTS_6.append(len(best_order))

    #     cost0 = calcul_cost(c, m, p, best_order)
    #     print("recal:\t", cost0)

    #     if cost0 < 100:
    #         suss_MCTS_6.append(1)
    #         cost_MCTS_6.append(cost)
    #     else:
    #         suss_MCTS_6.append(0)

    #     print("\n##############\n")
    #     print("online MCTS_currstep")
    #     cost, best_order, t = baseline_online_MCTS(c, m, p, step=2)
    #     time_MCTS_2.append(t)
    #     num_MCTS_2.append(len(best_order))

    #     cost0 = calcul_cost(c, m, p, best_order)
    #     print("recal:\t", cost0)

    #     if cost0 < 100:
    #         suss_MCTS_2.append(1)
    #         cost_MCTS_2.append(cost)
    #     else:
    #         suss_MCTS_2.append(0)

    #     print("\n##############\n")
    #     print("online planner")
    #     done = False
    #     a = []
    #     time_online = time.time()
    #     while not done:
    #         # print("before", obs["coop_edge_mask"])
    #         if task_config["use_prediction_model"]:
    #             obs = prediction_model.predict_data_for_online_planning(obs)
    #             env.update_prediction([obs["coop_edge_cost"], obs["coop_edge_mask"]])
    #         # print("after", obs["coop_edge_mask"])
    #         action = env.sample_action()
    #         print(action // (parts_num + 1), action % (parts_num + 1))
    #         obs, r, done, info = env.step(action)
    #         if not done:
    #             action = env.extract_allocator_action(action)
    #             a.append(action[0])
    #             a.append(action[1])
    #     online_planner_cost = info["2_cost/accumulated_cost"]

    #     print("cost:\t", online_planner_cost)
    #     print("time used:\t", time.time() - time_online)
    #     print("\n action order:")
    #     print(a)
    #     cost0 = calcul_cost(c, m, p, a)
    #     print("re-calculate cost:\t", cost0)
    #     time_min_cost.append(time.time() - time_online)
    #     num_min_cost.append(len(a))

    #     if cost0 < 100:
    #         suss_min_cost.append(1)
    #         cost_min_cost.append(online_planner_cost)
    #         suss_min_cost_recal.append(1)
    #         cost_min_cost_recal.append(online_planner_cost)

    #     else:
    #         suss_min_cost.append(0)
    #         suss_min_cost_recal.append(0)

    #     print("\n##############\n")
    #     print("loop end")
    #     print("loop total time:\t", time.time() - time00)

    # print("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    # print("evaluation end, time used:\t", time.time() - time0)
    # print("loop num:\t", loop)

    # data_c = {}
    # data_n = {}
    # data_s = {}
    # data_t = {}
    # data_c["heuristics"] = cost_heuristics
    # data_c["bruteforce"] = cost_brute_force
    # data_c["MCTS_6"] = cost_MCTS_6
    # data_c["MCTS_2"] = cost_MCTS_2
    # data_c["min_cost"] = cost_min_cost
    # data_c["min_cost_recal"] = cost_min_cost_recal

    # data_s["heuristics"] = suss_heuristics
    # data_s["bruteforce"] = suss_brute_force
    # data_s["MCTS_6"] = suss_MCTS_6
    # data_s["MCTS_2"] = suss_MCTS_2
    # data_s["min_cost"] = suss_min_cost
    # data_s["min_cost_recal"] = suss_min_cost_recal

    # data_n["heuristics"] = num_heuristics
    # data_n["bruteforce"] = num_brute_force
    # data_n["MCTS_6"] = num_MCTS_6
    # data_n["MCTS_2"] = num_MCTS_2
    # data_n["min_cost"] = num_min_cost

    # data_t["heuristics"] = time_heuristics
    # data_t["bruteforce"] = time_brute_force
    # data_t["MCTS_6"] = time_MCTS_6
    # data_t["MCTS_2"] = time_MCTS_2
    # data_t["min_cost"] = time_min_cost
    # data_t["collect_cost_and_mask"] = time_collect_cost_and_mask

    # data = {}
    # data["part_num"] = task_config["part_num"]
    # print("part_num:\t", data["part_num"])
    # for key in data_s:
    #     data[key + "_suss_rate"] = sum(data_s[key]) / loop

    # for key in data_n:
    #     data[key + "_num_parts"] = sum(data_n[key]) / loop

    # for key in data_c:
    #     data[key + "_succ_cost"] = (sum(data_c[key]) / sum(data_s[key])) if sum(data_s[key]) > 0 else 0

    # for key in data_t:
    #     data[key + "_search_time"] = sum(data_t[key]) / loop

    # for key in data:
    #     print(key, ":\t", data[key])
