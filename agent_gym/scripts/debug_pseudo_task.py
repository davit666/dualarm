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
    ## set gpu
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    #### define input output type to use
    cost_type = "coord_steps"  # "coord_steps"
    obs_type = "norm_ee_only"
    #### set loading path of prediction model
    cost_model_path = "../../generated_datas/good_models/cost/1203/1010_task_datas_100_epochs/norm_ee_only_succ_only_predict_step/512-512-512_ce_adam_rl1e-3_batch_512/2022-12-02-13-13-04/model_saved/2022-12-02-14-04-40.pth"
    mask_model_path = "../../generated_datas/good_models/mask/1203/1024_with_failure_task_datas_100_epochs/norm_ee_only_predict_succ/256-256-256_ce_adam_rl1e-3_batch_512/2022-12-02-11-47-26/model_saved/2022-12-02-12-29-48.pth"

    print("!!!!!!!!!!!!!!!!!!!!!!!!!1")
    #### load prediction model
    prediction_model = Prediction_Model(obs_type=obs_type, cost_type=cost_type, cost_model_path=cost_model_path, mask_model_path=mask_model_path)
    # input_features, cost_features, mask_features = prediction_model.get_input_and_output()
    #
    # input_shape = len(input_features)
    # cost_shape = len(cost_features)
    # mask_shape = len(mask_features)

    ######################## test a full episode with prediction
    task_config = load_config()

    task_config["task_allocator_action_type"] = "Discrete"

    task_allocator_reward_type = task_config["task_allocator_reward_type"]
    task_allocator_obs_type = task_config["task_allocator_obs_type"]
    task_allocator_action_type = task_config["task_allocator_action_type"]

    num_cpu = 1
    use_prediction_model = task_config["use_prediction_model"]

    env = CustomSubprocVecEnv([make_env(task_config, renders=False) for i in range(num_cpu)])
    env.load_prediction_model(prediction_model, input_type=obs_type, output_type=cost_type, use_prediction_model=use_prediction_model)

    obs = env.reset()
    # z_o = []
    # z_a = []
    # z_r = []
    # z_d = []
    # z_i = []
    # for i in range(100):
    #     acts = env.sample_action()
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

    ################ test feature extractor
    from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor

    # feature extractor
    obs_space = env.observation_space
    feature_extractor = CustomFeatureExtractor(obs_space)
    print("feature extractor:\n", feature_extractor.extractors)

    # custom net flatten nodes
    from model_utils_task import (
        CustomNetwork_FlattenNodesAndEdges,
        CustomNetwork_SelfAttention,
        CustomNetwork_SelfCrossAttention,
        CustomNetwork_SelfAttentionWithTaskEdge,
        CustomNetwork_SelfCrossAttentionWithEdge,
    )

    node_feature_dim = feature_extractor._features_dim
    mask_dim = feature_extractor.mask_dim

    # distribution
    action_space = env.action_space
    action_dist, discrete_action_space = make_custom_proba_distribution(action_space)

    # network
    # net_fn = CustomNetwork_FlattenNodes(node_feature_dim, mask_dim, discrete_action_space=discrete_action_space)
    # net_fn = CustomNetwork_FlattenNodesAndEdges(node_feature_dim, mask_dim, discrete_action_space=discrete_action_space)
    # net_fn = CustomNetwork_SelfAttention(node_feature_dim, mask_dim, discrete_action_space=discrete_action_space)
    # net_fn = CustomNetwork_SelfCrossAttention(node_feature_dim, mask_dim, discrete_action_space=discrete_action_space)
    # net_fn = CustomNetwork_SelfAttentionWithTaskEdge(node_feature_dim, mask_dim, discrete_action_space=discrete_action_space)
    net_fn = CustomNetwork_SelfCrossAttentionWithEdge(node_feature_dim, mask_dim, discrete_action_space=discrete_action_space)
    print("flatten network:\n", net_fn)

    obs = env.reset()
    actions = env.sample_action()
    obs, rews, dones, infos = env.step(actions)
    observation = {}
    for key, _ in obs.items():
        # print(obs[key])
        obs_ = np.array(obs[key])
        obs_ = obs_.reshape((-1,) + obs_space[key].shape)
        observation[key] = torch.as_tensor(obs_).to("cpu")

    # observation = obs_as_tensor(observation,'cpu')

    encoded_obs = feature_extractor.forward(observation)

    for k, v in encoded_obs.items():
        print(k)
        print(v.detach().numpy().shape)

    pi_logits, vf_logits = net_fn.forward(encoded_obs)

    results = {}
    for key, _ in encoded_obs.items():
        results[key] = encoded_obs[key].cpu().detach().numpy()

    distribution = action_dist.proba_distribution(action_logits=pi_logits)
    actions = distribution.get_actions(deterministic=True)

    actions = actions.cpu().detach().numpy()
    pi_logits, vf_logits = pi_logits.cpu().detach().numpy(), vf_logits.cpu().detach().numpy()

    obs2, rews, dones, infos = env.step(actions)

    ################# test action sample

    env0 = Env_tasks(
        task_config,
        renders=False,
        task_allocator_reward_type=task_allocator_reward_type,
        task_allocator_obs_type=task_allocator_obs_type,
        task_allocator_action_type=task_allocator_action_type,
    )

    act1 = env0.action_space.sample()
    act2 = env.sample_action()

    ################

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
