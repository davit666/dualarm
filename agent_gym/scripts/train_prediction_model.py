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


from train_utils import create_folder

# Define dataset
class MyDataset(Dataset):  #
    def __init__(self, df, input_features, output_features):

        self.x_data = df[input_features].values
        self.y_data = df[output_features].values

        self.length = len(self.y_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, w):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_shape, w),
            nn.ReLU(),
            # nn.Linear(w, w),
            # nn.ReLU(),
            # nn.Linear(w, w),
            # nn.ReLU(),
            nn.Linear(w, w),
            nn.ReLU(),
            nn.Linear(w, output_shape),
            # nn.Sigmoid(),
            # nn.Tanh(),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits


# Train
def train(dataloader, model, loss_fn, optimizer, get_correct=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # print(pred.size(), y.size(), pred.argmax(1).size())
        # print((pred.argmax(1) == y).type(torch.float))
        # print((pred.argmax(1) == y).type(torch.float).sum())
        if get_correct:
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= num_batches
    correct /= size
    return train_loss, correct


# Test
def test(dataloader, model, loss_fn, get_correct=False):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

            if get_correct:
                correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}% ")
    print(f"Avg loss: {test_loss:>8f} \n")
    if get_correct:
        print(f"Accuracy: {(100*correct):>0.1f} \n")
    return test_loss, correct


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

    print("task_features:\t", len(task_features), task_features)
    print("policy features:\t", len(policy_features), policy_features)
    return task_features, policy_features


if __name__ == "__main__":

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # define save name and basic status
    save_date = time.strftime("%m%d")
    prediction_model_path = "../../generated_datas/prediction_models/"
    save_name = "1024_with_failure_task_datas_100_epochs/ee_only_predict_succ"
    save_name = "1109_w_failure_task_datas_100_epochs/true_succ_only_ee_only_predict_steps"

    batch_size = 512
    epochs = 100

    # define parameters
    task_data_path = "../../generated_datas/task_datas/1010/3M_data_24cpu/policy_0.1_gap/2022-10-10-23-30-56/"
    task_data_path = "../../generated_datas/task_datas/1024/with_failure/3M_data_24cpu/policy_0.1_gap/2022-10-24-18-44-01/"
    task_data_path = "../../generated_datas/task_datas/1109/no_failure/3M_data_24cpu/policy_0.1_gap/2022-11-09-22-39-41/"
    task_data_path = "../../generated_datas/task_datas/1109/with_failure/3M_data_24cpu/policy_0.1_gap/2022-11-09-22-39-13/"

    train_ratio = 0.8
    validation_ratio = 0.1
    test_ratio = 0.1

    task_features, policy_features = get_features()
    features = task_features + policy_features

    input_features = task_features[6:17] + task_features[23:33] + task_features[39:50] + task_features[56:]
    # input_features = task_features[23:33] + task_features[56:]

    output_features = policy_features[-6:-4]
    # output_features = policy_features[-3:-2]  # [-6:-4]  # [-1:]
    output_features = policy_features[-3:-2]

    input_shape = len(input_features)
    output_shape = len(output_features)

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

    # print(task_datas)

    # create data table
    task_datas = pd.DataFrame(task_datas, columns=features)
    task_datas = task_datas.astype(np.float32)

    task_datas = task_datas[task_datas["coord_suss"] == 1][task_datas["coord_steps"] >= 10]

    # suss_t = task_datas[task_datas["coord_suss"] == 1][task_datas["coord_steps"] >= 10]
    # suss_f = task_datas[task_datas["coord_fail"] == 1]
    # suss_t = suss_t[: suss_f.shape[0]]

    # task_datas = pd.concat([suss_t, suss_f], axis=0)
    # task_datas = task_datas.sample(frac=1.0).reset_index(drop=True)

    task_data_num = task_datas.shape[0]

    print("task num:\t", task_data_num)
    # print(task_datas)

    # create train, evaluate, test data set

    train_data_num = int(task_data_num * train_ratio)
    validation_data_num = int(task_data_num * validation_ratio)
    test_data_num = int(task_data_num * test_ratio)
    print("train data num:\t", train_data_num, "\tvalidation data num:\t", validation_data_num, "\ttest data num:\t", test_data_num)

    train_data_set = MyDataset(task_datas[:train_data_num], input_features, output_features)
    validation_data_set = MyDataset(task_datas[train_data_num : train_data_num + validation_data_num], input_features, output_features)
    test_data_set = MyDataset(task_datas[train_data_num + validation_data_num :], input_features, output_features)

    print(train_data_set.length, validation_data_set.length, test_data_set.length)

    # create dataloader

    train_dataloader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_data_set, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data_set, batch_size=batch_size, shuffle=True)

    for X, y in test_dataloader:
        print("Shape of X :\t", X.shape)
        print("Shape of Y :\t", y.shape)
        break

    ##############################################################################################################
    ###############################################################################################################
    # define network
    w = 64
    model_name = "{}-{}_ce_adam_rl1e-3_batch_512".format(
        w,
        w,
    )
    save_path = prediction_model_path + save_date + "/" + save_name + "/" + model_name + "/" + time.strftime("%Y-%m-%d-%H-%M-%S") + "/"
    create_folder(save_path + "model_saved/")
    get_acc = True
    get_acc = False
    # create model

    model = NeuralNetwork(input_shape, output_shape, w).to(device)
    print(model)
    loss_fn = nn.MSELoss()
    # loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # train
    time0 = time.time()
    x_epochs = list(range(epochs + 1))[1:]
    y_train_losses = []
    y_validation_losses = []
    y_train_correct = []
    y_validation_correct = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        tr_loss, tr_acc = train(train_dataloader, model, loss_fn, optimizer, get_correct=get_acc)
        vl_loss, vl_acc = test(validation_dataloader, model, loss_fn, get_correct=get_acc)
        y_train_losses.append(tr_loss)
        y_validation_losses.append(vl_loss)
        y_train_correct.append(tr_acc)
        y_validation_correct.append(vl_acc)

        torch.save(model, save_path + "model_saved/" + time.strftime("%Y-%m-%d-%H-%M-%S") + ".zip")
        print("time used:\t", time.time() - time0)

    print("Done!")

    y_train_losses = np.array(y_train_losses)
    y_validation_losses = np.array(y_validation_losses)
    losses_data = np.stack([y_train_losses, y_validation_losses])
    np.save(save_path + "losses_data.npy", losses_data)
    correct_data = np.stack([y_train_correct, y_validation_correct])
    np.save(save_path + "correct_data.npy", correct_data)

    fig = plt.figure(figsize=(10, 6), dpi=800)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title(save_name + "---" + model_name)
    plt.plot(x_epochs, y_train_losses, "b-", x_epochs, y_validation_losses, "r*")
    plt.savefig(save_path + "train_loss.png")
    fig.show()

    time.sleep(2)
    del model, fig

    fig = plt.figure(figsize=(10, 6), dpi=800)
    plt.xlabel("epochs")
    plt.ylabel("correct")
    plt.title(save_name + "---" + model_name)
    plt.plot(x_epochs, y_train_correct, "b-", x_epochs, y_validation_correct, "r*")
    plt.savefig(save_path + "train_correct.png")
    fig.show()
    time.sleep(2)
    del fig

    ###############################################################################################################
    ###############################################################################################################
    # define network
    w = 128
    model_name = "{}-{}_ce_adam_rl1e-3_batch_512".format(
        w,
        w,
    )
    save_path = prediction_model_path + save_date + "/" + save_name + "/" + model_name + "/" + time.strftime("%Y-%m-%d-%H-%M-%S") + "/"
    create_folder(save_path + "model_saved/")
    get_acc = True
    get_acc = False
    # create model

    model = NeuralNetwork(input_shape, output_shape, w).to(device)
    print(model)
    loss_fn = nn.MSELoss()
    # loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # train
    time0 = time.time()
    x_epochs = list(range(epochs + 1))[1:]
    y_train_losses = []
    y_validation_losses = []
    y_train_correct = []
    y_validation_correct = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        tr_loss, tr_acc = train(train_dataloader, model, loss_fn, optimizer, get_correct=get_acc)
        vl_loss, vl_acc = test(validation_dataloader, model, loss_fn, get_correct=get_acc)
        y_train_losses.append(tr_loss)
        y_validation_losses.append(vl_loss)
        y_train_correct.append(tr_acc)
        y_validation_correct.append(vl_acc)

        torch.save(model, save_path + "model_saved/" + time.strftime("%Y-%m-%d-%H-%M-%S") + ".zip")
        print("time used:\t", time.time() - time0)

    print("Done!")

    y_train_losses = np.array(y_train_losses)
    y_validation_losses = np.array(y_validation_losses)
    losses_data = np.stack([y_train_losses, y_validation_losses])
    np.save(save_path + "losses_data.npy", losses_data)
    correct_data = np.stack([y_train_correct, y_validation_correct])
    np.save(save_path + "correct_data.npy", correct_data)

    fig = plt.figure(figsize=(10, 6), dpi=800)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title(save_name + "---" + model_name)
    plt.plot(x_epochs, y_train_losses, "b-", x_epochs, y_validation_losses, "r*")
    plt.savefig(save_path + "train_loss.png")
    fig.show()

    time.sleep(2)
    del model, fig

    fig = plt.figure(figsize=(10, 6), dpi=800)
    plt.xlabel("epochs")
    plt.ylabel("correct")
    plt.title(save_name + "---" + model_name)
    plt.plot(x_epochs, y_train_correct, "b-", x_epochs, y_validation_correct, "r*")
    plt.savefig(save_path + "train_correct.png")
    fig.show()
    time.sleep(2)
    del fig

    ###############################################################################################################
    ###############################################################################################################
    # define network
    w = 256
    model_name = "{}-{}_ce_adam_rl1e-3_batch_512".format(
        w,
        w,
    )
    save_path = prediction_model_path + save_date + "/" + save_name + "/" + model_name + "/" + time.strftime("%Y-%m-%d-%H-%M-%S") + "/"
    create_folder(save_path + "model_saved/")
    get_acc = True
    get_acc = False
    # create model

    model = NeuralNetwork(input_shape, output_shape, w).to(device)
    print(model)
    loss_fn = nn.MSELoss()
    # loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # train
    time0 = time.time()
    x_epochs = list(range(epochs + 1))[1:]
    y_train_losses = []
    y_validation_losses = []
    y_train_correct = []
    y_validation_correct = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        tr_loss, tr_acc = train(train_dataloader, model, loss_fn, optimizer, get_correct=get_acc)
        vl_loss, vl_acc = test(validation_dataloader, model, loss_fn, get_correct=get_acc)
        y_train_losses.append(tr_loss)
        y_validation_losses.append(vl_loss)
        y_train_correct.append(tr_acc)
        y_validation_correct.append(vl_acc)

        torch.save(model, save_path + "model_saved/" + time.strftime("%Y-%m-%d-%H-%M-%S") + ".zip")
        print("time used:\t", time.time() - time0)

    print("Done!")

    y_train_losses = np.array(y_train_losses)
    y_validation_losses = np.array(y_validation_losses)
    losses_data = np.stack([y_train_losses, y_validation_losses])
    np.save(save_path + "losses_data.npy", losses_data)
    correct_data = np.stack([y_train_correct, y_validation_correct])
    np.save(save_path + "correct_data.npy", correct_data)

    fig = plt.figure(figsize=(10, 6), dpi=800)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title(save_name + "---" + model_name)
    plt.plot(x_epochs, y_train_losses, "b-", x_epochs, y_validation_losses, "r*")
    plt.savefig(save_path + "train_loss.png")
    fig.show()

    time.sleep(2)
    del model, fig

    fig = plt.figure(figsize=(10, 6), dpi=800)
    plt.xlabel("epochs")
    plt.ylabel("correct")
    plt.title(save_name + "---" + model_name)
    plt.plot(x_epochs, y_train_correct, "b-", x_epochs, y_validation_correct, "r*")
    plt.savefig(save_path + "train_correct.png")
    fig.show()
    time.sleep(2)
    del fig

    ###############################################################################################################
    ###############################################################################################################
    # define network
    w = 512
    model_name = "{}-{}_ce_adam_rl1e-3_batch_512".format(
        w,
        w,
    )
    save_path = prediction_model_path + save_date + "/" + save_name + "/" + model_name + "/" + time.strftime("%Y-%m-%d-%H-%M-%S") + "/"
    create_folder(save_path + "model_saved/")
    get_acc = True
    get_acc = False
    # create model

    model = NeuralNetwork(input_shape, output_shape, w).to(device)
    print(model)
    loss_fn = nn.MSELoss()
    # loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # train
    time0 = time.time()
    x_epochs = list(range(epochs + 1))[1:]
    y_train_losses = []
    y_validation_losses = []
    y_train_correct = []
    y_validation_correct = []
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        tr_loss, tr_acc = train(train_dataloader, model, loss_fn, optimizer, get_correct=get_acc)
        vl_loss, vl_acc = test(validation_dataloader, model, loss_fn, get_correct=get_acc)
        y_train_losses.append(tr_loss)
        y_validation_losses.append(vl_loss)
        y_train_correct.append(tr_acc)
        y_validation_correct.append(vl_acc)

        torch.save(model, save_path + "model_saved/" + time.strftime("%Y-%m-%d-%H-%M-%S") + ".pth")
        print("time used:\t", time.time() - time0)

    print("Done!")

    y_train_losses = np.array(y_train_losses)
    y_validation_losses = np.array(y_validation_losses)
    losses_data = np.stack([y_train_losses, y_validation_losses])
    np.save(save_path + "losses_data.npy", losses_data)
    correct_data = np.stack([y_train_correct, y_validation_correct])
    np.save(save_path + "correct_data.npy", correct_data)

    fig = plt.figure(figsize=(10, 6), dpi=800)
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.title(save_name + "---" + model_name)
    plt.plot(x_epochs, y_train_losses, "b-", x_epochs, y_validation_losses, "r*")
    plt.savefig(save_path + "train_loss.png")
    fig.show()

    time.sleep(2)
    del model, fig

    fig = plt.figure(figsize=(10, 6), dpi=800)
    plt.xlabel("epochs")
    plt.ylabel("correct")
    plt.title(save_name + "---" + model_name)
    plt.plot(x_epochs, y_train_correct, "b-", x_epochs, y_validation_correct, "r*")
    plt.savefig(save_path + "train_correct.png")
    fig.show()
    time.sleep(2)
    del fig

    ###############################################################################################################
