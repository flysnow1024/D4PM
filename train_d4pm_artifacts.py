import argparse
import torch
import yaml
import os
import numpy as np
from Data_Preparation.data_for_eegdnet import prepare_data

from DDPM_joint import DDPM
from denoising_model_eegdnet_class_noise import DualBranchDenoisingModel_noise
from utils import train

from torch.utils.data import DataLoader, TensorDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="base.yaml")
    parser.add_argument('--device', default='cuda:0', help='Device')
    parser.add_argument('--n_type', type=str, default='d4pm', help='noise version')
    args = parser.parse_args()
    print(args)

    path = "config/" + args.config
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    foldername = "./check_points/Artifacts_" + args.n_type + "/"
    print('folder:', foldername)
    os.makedirs(foldername, exist_ok=True)

    [X_train, y_train, y_train_noise, label_train, X_val, y_val, y_val_noise, label_val] = prepare_data(combin_num=11, train_per=0.8, noise_type=args.n_type)

    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train_noise)
    label_train = torch.FloatTensor(label_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val_noise)
    label_val = torch.FloatTensor(label_val)

    train_set = TensorDataset(y_train, X_train, label_train)
    val_set = TensorDataset(y_val, X_val, label_val)

    train_loader = DataLoader(train_set, batch_size=config['train']['batch_size'],
                              shuffle=True, drop_last=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=config['train']['batch_size'], drop_last=True, num_workers=8)

    base_model = DualBranchDenoisingModel(config['train']['feats']).to(args.device)
    model = DDPM(base_model, config, args.device)

    train(model, config['train'], train_loader, args.device,
          valid_loader=val_loader, valid_epoch_interval=10, foldername=foldername)










