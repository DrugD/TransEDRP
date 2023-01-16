import numpy as np
import pandas as pd
import sys
import os
from random import shuffle

from tqdm import tqdm
import torch
import torch.nn as nn


from models.graphdrp import GraphDRP
from models.graphtransdrp import GraTransDRP
from models.transformer_edge import TransE
from models.deepcdr import DeepCDR
from models.deepttc import DeepTTC
from models.tcnns import tCNNs

from utils import *
import datetime
import argparse
import random

from utils import copyfile

# training function at each epoch


def train(model, device, train_loader, optimizer, epoch, log_interval, args):
    print("Training on {} samples...".format(len(train_loader.dataset)))
    model.train()
    loss_fn = nn.MSELoss()
    avg_loss = []
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output.float(), data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())
        if batch_idx % log_interval == 0:
            print(
                "Train epoch: {} ({:.0f}%)\tLoss: {:.6f}".format(
                    epoch, 100.0 * batch_idx / len(train_loader), loss.item()
                )
            )
    return sum(avg_loss) / len(avg_loss)


def predicting(model, device, loader, loader_type, args):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print("Make prediction for {} samples...".format(len(loader.dataset)))
    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            data = data.to(device)

            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def dateStr():
    return (
        str(datetime.datetime.now())
        .replace(" ", "_")
        .replace("-", "_")
        .replace(":", "_")
        .split(".")[0]
        .replace("_", "")
    )


def main(model, config, yaml_path):

    train_batch = config["batch_size"]["train"]
    val_batch = config["batch_size"]["val"]
    test_batch = config["batch_size"]["test"]
    lr = config["lr"]
    num_epoch = config["num_epoch"]
    log_interval = config["log_interval"]
    cuda_name = config["cuda_name"]
    work_dir = config["work_dir"]

    date_info = ("_" + dateStr()) if config["work_dir"] != "test" else ""
    work_dir = "./exp/" + work_dir + date_info
    if not os.path.exists(work_dir):
        os.mkdir(work_dir)

    copyfile(yaml_path, work_dir + "/")

    model_st = config["model_name"]
    train_dataset, val_dataset, test_dataset = config["dataset_type"]
    dataset = config["dataset_name"]

    train_data = TestbedDataset(root="data", dataset=train_dataset)
    val_data = TestbedDataset(root="data", dataset=val_dataset)
    test_data = TestbedDataset(root="data", dataset=test_dataset)

    # make data PyTorch mini-batch processing ready
    train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=val_batch, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)
    print("CPU/GPU: ", torch.cuda.is_available())

    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_mse = 9999
    best_pearson = 1
    best_epoch = -1

    model_file_name = work_dir + "/" + model_st + ".model"
    result_file_name = work_dir + "/" + model_st + ".csv"
    loss_fig_name = work_dir + "/" + model_st + "_loss"
    pearson_fig_name = work_dir + "/" + model_st + "_pearson"

    train_losses = []
    val_losses = []
    val_pearsons = []

    for epoch in tqdm(range(num_epoch)):
        train_loss = train(
            model, device, train_loader, optimizer, epoch + 1, log_interval, config
        )
        G, P = predicting(model, device, val_loader, "val", config)
        ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P)]

        G_test, P_test = predicting(model, device, test_loader, "test", config)
        ret_test = [
            rmse(G_test, P_test),
            mse(G_test, P_test),
            pearson(G_test, P_test),
            spearman(G_test, P_test),
        ]

        train_losses.append(train_loss)
        val_losses.append(ret[1])
        val_pearsons.append(ret[2])

        if ret[1] < best_mse:
            torch.save(model.state_dict(), model_file_name)
            with open(result_file_name, "a") as f:
                f.write("\n" + str(epoch) + ":\n")
                f.write(",".join(map(str, ret_test)))
            best_epoch = epoch + 1
            best_mse = ret[1]
            best_pearson = ret[2]
            print(
                " rmse improved at epoch ",
                best_epoch,
                "; best_mse:",
                best_mse,
                model_st,
                dataset,
            )
        else:
            print(
                " no improvement since epoch ",
                best_epoch,
                "; best_mse, best pearson:",
                best_mse,
                best_pearson,
                model_st,
                dataset,
            )

        draw_loss(train_losses, val_losses, loss_fig_name)
        draw_pearson(val_pearsons, pearson_fig_name)


def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def getConfig():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="./config/Transformer_edge_concat_GDSCv2.yaml",
        help="",
    )
    args = parser.parse_args()
    import yaml

    with open(args.config, "r") as stream:
        config = yaml.safe_load(stream)
    return config, args.config


if __name__ == "__main__":
    config, yaml_path = getConfig()

    seed_torch(config["seed"])

    modeling = [TransE, GraphDRP, DeepCDR, tCNNs, DeepTTC, GraTransDRP][
        config["model_type"]
    ]
    model = modeling(config)

    main(model, config, yaml_path)
