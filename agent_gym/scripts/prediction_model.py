import os
import sys
import time

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, w):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_shape, w),
            nn.ReLU(),
            nn.Linear(w, w),
            nn.ReLU(),
            nn.Linear(w, output_shape),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


def get_features():
    robot_features = [
        "js1",
        "js2",
        "js3",
        "js4",
        "js5",
        "js6",
        "ee_x",
        "ee_y",
        "ee_z",
        "ee_rz",
        "goal_x",
        "goal_y",
        "goal_z",
        "goal_rz",
        "dist_xyz",
        "dist_rpy",
        "is_picking",
        "norm_js1",
        "norm_js2",
        "norm_js3",
        "norm_js4",
        "norm_js5",
        "norm_js6",
        "norm_ee_x",
        "norm_ee_y",
        "norm_ee_z",
        "norm_ee_rz",
        "norm_goal_x",
        "norm_goal_y",
        "norm_goal_z",
        "norm_goal_rz",
        "norm_dist_xyz",
        "norm_dist_rpy",
    ]
    robots = ["r1", "r2"]
    task_features = []
    for r in robots:
        for rf in robot_features:
            task_features.append(r + "_" + rf)

    eva_feature = ["suss", "fail", "traj_length", "steps", "norm_traj_length", "norm_steps"]
    policy_features = []
    for k in ["r1_", "r2", "coord"]:
        for ef in eva_feature:
            policy_features.append(k + "_" + ef)

    # print("task_features:\t", len(task_features), task_features)
    # print("policy features:\t", len(policy_features), policy_features)
    return task_features, policy_features


class Prediction_Model():
    def __init__(self, obs_type=None, cost_type=None, cost_model_path=None, mask_model_path=None):

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")

        self.cost_types = ["coord_traj_length", "coord_steps", "coord_norm_traj_length", "coord_norm_steps"]
        self.obs_types = ["common", "ee_only", "norm_common", "norm_ee_only"]
        self.cost_prediction_model = None
        self.mask_prediction_model = None

        assert obs_type in self.obs_types
        assert cost_type in self.cost_types

        self.obs_type = obs_type
        self.cost_type = cost_type
        self.cost_model_path = cost_model_path
        self.mask_model_path = mask_model_path

        self.define_input_and_output()
        self.load_cost_model()
        self.load_mask_model()
        print("prediction model loaded, obs_type:\t{}\tcost_type:{}".format(self.obs_type, self.cost_type))
        pass

    def define_input_and_output(self):
        task_features, policy_features = get_features()
        if self.obs_type == "common":
            task_features = task_features[:]
        elif self.obs_type == "ee_only":
            task_features = task_features[6:17] + task_features[23:33] + task_features[39:50] + task_features[56:]
        elif self.obs_type == "norm_common":
            task_features = task_features[17:33] + task_features[50:]
        elif self.obs_type == "norm_ee_only":
            task_features = task_features[23:33] + task_features[56:]

        print("input features are:\t", task_features)
        print("cost feature is:\t", self.cost_type)
        self.input_features = task_features
        self.cost_features = [self.cost_type]
        self.mask_features = ["coord_suss"]

    def get_input_and_output(self):
        return self.input_features.copy(), self.cost_features.copy(), self.mask_features.copy()

    def load_cost_model(self):
        self.cost_prediction_model = torch.load(self.cost_model_path).to(self.device)
        print("cost prediction model loaded, path:\t", self.cost_model_path)

    def load_mask_model(self):
        self.mask_prediction_model = torch.load(self.mask_model_path).to(self.device)
        print("mask prediction model loaded, path:\t", self.mask_model_path)

    def model_eval(self):
        self.cost_prediction_model.eval()
        self.mask_prediction_model.eval()

    def predict_cost(self, x):
        # x = x.to(self.device)
        pred_cost = self.cost_prediction_model.forward(x)  # .item()
        return torch.reshape(pred_cost, pred_cost.shape[:-1])

    def predict_mask(self, x):
        dim = len(x.shape)
        # x = x.to(self.device)
        pred_mask = self.mask_prediction_model.forward(x).argmax(dim=dim - 1)
        return pred_mask

    def predict_correct(self, x):
        # x = x.to(self.device)
        pred_mask = self.mask_prediction_model.forward(x).argmax(1)
        mask_shape = list(pred_mask.shape)[0]
        # print(mask_shape,"!!!!!!!!!!!!!!")
        pred_mask = torch.reshape(pred_mask, (mask_shape, 1))
        pred_mask = (pred_mask == 0).type(torch.float)  # .item()
        return pred_mask

    def predict_data_for_online_planning(self, obs):
        prediction_inputs = obs["prediction_inputs"]
        unpred_cost = obs["coop_edge_cost"]
        unpred_mask = obs["coop_edge_mask"]

        prediction_inputs_shape = prediction_inputs.shape

        X = torch.tensor(prediction_inputs).to(self.device)
        pred_cost = self.predict_cost(X).cpu().detach()  # .numpy()
        pred_mask = self.predict_mask(X).cpu().detach()  # .numpy()
        # print(pred_cost.shape, pred_mask.shape)
        pred_mask[:, -1] = 1

        pred_cost = torch.sum(pred_cost, -2).reshape(unpred_cost.shape).numpy().astype(np.float32)
        pred_mask = torch.multiply(pred_mask[0, :], pred_mask[1, :]).reshape(unpred_mask.shape).numpy().astype(
            np.float32)

        obs["coop_edge_mask"] = np.multiply(pred_mask, unpred_mask)
        # print("before",obs["coop_edge_mask"])
        mask_terminate = obs["coop_edge_mask"][:-1, :-1].sum(axis=-2).sum(axis=-1)
        # print("mask_termination",mask_terminate)
        mask_terminate = mask_terminate < 1
        # print("mask_termination", mask_terminate)

        obs["coop_edge_mask"][-1, -1] = mask_terminate
        # print("after",obs["coop_edge_mask"])
        obs["coop_edge_cost"] = np.multiply(pred_cost, obs["coop_edge_mask"]) + np.multiply(unpred_cost,
                                                                                            1 - obs["coop_edge_mask"])
        obs["coop_edge_cost"] = obs["coop_edge_cost"] / 1000

        return obs
    def predict_data_for_offline_planning(self, feature, mask):
        feature_n = feature['n']
        feature_n2n = feature['n2n']
        mask_n = mask['n']
        mask_n2n = mask['n2n']
        feature_dim = feature['dim']
        max_cost = feature['max_cost']
        part_num = feature['part_num']

        # print(feature_n.shape, feature_n2n.shape,mask_n.shape, mask_n2n.shape)

        f_n = torch.tensor(feature_n).reshape((part_num) * (part_num), feature_dim).to(self.device)
        f_n2n = torch.tensor(feature_n2n).reshape((part_num + 1) * (part_num + 1), (part_num + 1) * (part_num + 1),
                                                  feature_dim).to(self.device)

        # m_n = torch.tensor(mask_n).reshape((part_num) * (part_num), 1).to(self.device)
        # m_n2n = torch.tensor(mask_n2n).reshape((part_num + 1) * (part_num + 1), (part_num + 1) * (part_num + 1),
        #                                        1).to(self.device)

        # print(f_n.shape, f_n2n.shape)

        pred_cost_n = self.predict_cost(f_n)
        pred_mask_n = self.predict_mask(f_n)

        pred_cost_n2n = []
        pred_mask_n2n = []
        for k in range(f_n2n.shape[0]):
            pred_cost_n2n.append(self.predict_cost(f_n2n[k]))
            pred_mask_n2n.append(self.predict_mask(f_n2n[k]))

        pred_cost_n2n = torch.stack(pred_cost_n2n, dim=0)
        pred_mask_n2n = torch.stack(pred_mask_n2n, dim=0)

        # print("predicted:\t",pred_cost_n.shape, pred_cost_n2n.shape, pred_mask_n.shape, pred_mask_n2n.shape)

        pred_cost_n = pred_cost_n.cpu().detach().numpy().reshape(mask_n.shape)
        pred_mask_n = pred_mask_n.cpu().detach().numpy().reshape(mask_n.shape)
        pred_cost_n2n = pred_cost_n2n.cpu().detach().numpy().reshape(mask_n2n.shape)
        pred_mask_n2n = pred_mask_n2n.cpu().detach().numpy().reshape(mask_n2n.shape)

        # print("shaped:\t", pred_cost_n.shape, pred_cost_n2n.shape, pred_mask_n.shape, pred_mask_n2n.shape)
        pred_mask_n = np.multiply(pred_mask_n,mask_n)
        pred_cost_n = np.multiply(pred_cost_n,pred_mask_n) / max_cost + (1 - pred_mask_n)

        pred_mask_n2n = np.multiply(pred_mask_n2n, mask_n2n)
        pred_cost_n2n = np.multiply(pred_cost_n2n, pred_mask_n2n) / max_cost + (1 - pred_mask_n2n)

        # print("customed:\t", pred_cost_n.shape, pred_cost_n2n.shape, pred_mask_n.shape, pred_mask_n2n.shape)



        c = {}
        m ={}
        c['n'] = pred_cost_n
        c['n2n'] = pred_cost_n2n
        m['n'] = pred_mask_n
        m['n2n'] = pred_mask_n2n

        return c, m