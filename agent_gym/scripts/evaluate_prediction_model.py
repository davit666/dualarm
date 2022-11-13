import os
import sys
import time

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.utils.data as Data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from prediction_model import Prediction_Model, get_features, NeuralNetwork

from train_utils import create_folder


# Define dataset
class MyDataset(Dataset):  #
    def __init__(self, df, input_features, cost_features, mask_features):
        self.x_data = df[input_features].values
        self.cost_data = df[cost_features].values
        self.mask_data = df[mask_features].values

        self.length = len(self.cost_data)

    def __getitem__(self, index):
        return self.x_data[index], self.cost_data[index], self.mask_data[index]

    def __len__(self):
        return self.length


def test(device, dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.model_eval()
    cost_loss = 0
    # mask_loss = 0
    mask_accuracy = 0
    mask_precision = 0
    mask_recall = 0

    precision_num = 0
    recall_num = 0
    with torch.no_grad():
        for X0, c0, m0 in dataloader:
            X, c, m = X0.to(device), c0.to(device), m0.to(device)
            pred_cost = model.predict_cost(X)
            pred_mask = model.predict_mask(X)
            cost_loss += nn.MSELoss()(pred_cost, c).item()

            correct = (pred_mask == m).type(torch.float)
            true_positive = torch.multiply(correct, m)  # (correct == m).type(torch.float)

            mask_accuracy += correct.sum().item()
            mask_precision += true_positive.sum().item()
            mask_recall += true_positive.sum().item()

            precision_num += (m == 1).type(torch.float).sum().item()
            recall_num += (correct == 1).type(torch.float).sum().item()

            # print(pred_mask.shape, m.shape, correct.shape)
            # print(correct.sum().item())
            # print(true_positive.sum().item())
            # print(pred_mask.sum().item())
            # print(m.sum().item())
            # time.sleep(0.4)

    cost_loss /= num_batches
    mask_accuracy /= size
    mask_precision /= precision_num
    mask_recall /= recall_num
    print(f"Avg cost loss: {cost_loss:>8f} \n")
    print(f"Mask Accuracy: {(100 * mask_accuracy):>0.1f} \n")
    print(f"Mask Precision: {mask_precision:>8f} \n")
    print(f"Mask Recall: {mask_recall:>8f} \n")
    return cost_loss, mask_accuracy, mask_precision, mask_recall


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    task_features, policy_features = get_features()
    features = task_features + policy_features
    cost_types = ["coord_traj_length", "coord_steps", "coord_norm_traj_length", "coord_norm_steps"]
    obs_types = ["common", "ee_only", "norm_common", "norm_ee_only"]
    #### define input output type to use
    cost_type = "coord_steps"
    obs_type = "ee_only"

    #### define task data to use

    task_data_path = "../../generated_datas/task_datas/1010/3M_data_24cpu/policy_0.1_gap/2022-10-10-23-30-56/"
    # task_data_path = "../../generated_datas/task_datas/1024/with_failure/3M_data_24cpu/policy_0.1_gap/2022-10-24-18-44-01/"

    #### define prediction model to use

    cost_model_path = "../../generated_datas/good_models/cost/1109/succ_data_only_ee_only_predict_steps/2022-11-07-22-22-13.zip"
    mask_model_path = "../../generated_datas/good_models/mask/1107/ee_only_predict_suss/2022-11-07-01-51-09.zip"

    model = torch.load(cost_model_path).to(device)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!1")
    #### load prediction model
    prediction_model = Prediction_Model(obs_type=obs_type, cost_type=cost_type, cost_model_path=cost_model_path, mask_model_path=mask_model_path)
    input_features, cost_features, mask_features = prediction_model.get_input_and_output()

    input_shape = len(input_features)
    cost_shape = len(cost_features)
    mask_shape = len(mask_features)
    #### load data
    batch_size = 512
    train_ratio = 0.7
    validation_ratio = 0.1
    test_ratio = 0.2

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

    # create data table
    task_datas = pd.DataFrame(task_datas, columns=features)
    task_datas = task_datas.astype(np.float32)

    ############################################################## preprocess data
    task_datas = task_datas[task_datas["coord_suss"] == 1][task_datas["coord_steps"] >= 10]
    ##############################################################################
    task_data_num = task_datas.shape[0]
    print("task num:\t", task_data_num)
    # print(task_datas)

    #### create train, evaluate, test data set

    train_data_num = int(task_data_num * train_ratio)
    validation_data_num = int(task_data_num * validation_ratio)
    test_data_num = int(task_data_num * test_ratio)
    print("train data num:\t", train_data_num, "\tvalidation data num:\t", validation_data_num, "\ttest data num:\t", test_data_num)

    train_data_set = MyDataset(task_datas[:train_data_num], input_features, cost_features, mask_features)
    validation_data_set = MyDataset(task_datas[train_data_num : train_data_num + validation_data_num], input_features, cost_features, mask_features)
    test_data_set = MyDataset(task_datas[train_data_num + validation_data_num :], input_features, cost_features, mask_features)

    print(train_data_set.length, validation_data_set.length, test_data_set.length)

    #### create dataloader

    train_dataloader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_data_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data_set, batch_size=batch_size, shuffle=True)

    for X, c, m in test_dataloader:
        print("Shape of X :\t", X.shape)
        print("Shape of cost :\t", c.shape)
        print("Shape of mask :\t", m.shape)
        break

    #### test
    time0 = time.time()
    cost_losses = []
    mask_accuracies = []
    mask_precisions = []
    mask_recalls = []

    train_cl, train_ma, train_mp, train_mr = test(device, train_dataloader, prediction_model)

    print("For training data:")
    print("cost loss is :\t", train_cl)
    print("mask accuracy is :\t", train_ma)
    print("mask precision is :\t", train_mp)
    print("msk recall is :\t", train_mr)

    test_cl, test_ma, test_mp, test_mr = test(device, test_dataloader, prediction_model)

    print("For testing data:")
    print("cost loss is :\t", test_cl)
    print("mask accuracy is :\t", test_ma)
    print("mask precision is :\t", test_mp)
    print("msk recall is :\t", test_mr)
