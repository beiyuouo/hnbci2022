#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   src\main_nn.py
# @Time    :   2022-03-10 11:02:50
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import time
import argparse
import numpy as np
from scipy import signal
import pandas as pd

import torch
import torch.nn as nn

from src.model import EEGNet

opt = argparse.ArgumentParser()
opt.add_argument('--data_path', type=str, default='../data/', help='data path')
opt.add_argument('--result_path', type=str, default='../result/', help='result path')
opt.add_argument('--batch_size', type=int, default=4, help='batch size')
opt.add_argument('--epochs', type=int, default=300, help='epochs')
opt.add_argument('--lr', type=float, default=0.001, help='learning rate')
opt.add_argument('--model_path', type=str, default='../model/', help='model path')
args = opt.parse_args()


def get_data(start_idx, end_idx):
    file_id = [
        '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14',
        '15', '16', '17', '18', '19', '20'
    ]
    train_data = None
    train_label = None
    test_data = None

    for i in range(start_idx, end_idx):
        train_data_temp = np.load(
            os.path.join(args.data_path, 'train_data_' + file_id[i] + '.npy'))
        train_label_temp = np.load(
            os.path.join(args.data_path, 'train_label_' + file_id[i] + '.npy'))
        test_data_temp = np.load(
            os.path.join(args.data_path, 'test_data_' + file_id[i] + '.npy'))

        if train_data is None:
            train_data = train_data_temp
            train_label = train_label_temp
            test_data = test_data_temp
        else:
            train_data = np.concatenate((train_data, train_data_temp), axis=0)
            train_label = np.concatenate((train_label, train_label_temp), axis=0)
            test_data = np.concatenate((test_data, test_data_temp), axis=0)

    print(train_data.shape, train_label.shape, test_data.shape)
    return train_data, train_label, test_data


def train_test(train_data, train_label, test_data, in_channels=30, in_feature=3750):
    model = EEGNet(classes_num=1, in_channels=in_channels, in_feature=in_feature, dropout=0.25)
    print(model)

    b, a = signal.butter(8, [8 * 2 / 250, 30 * 2 / 250], 'bandpass')
    train_data = signal.filtfilt(b, a, train_data)
    test_data = signal.filtfilt(b, a, test_data)

    train_X = train_data
    train_y = train_label.reshape(-1)
    test_X = test_data

    # normalize min max
    train_X = (train_X - train_X.min()) / (train_X.max() - train_X.min())
    test_X = (test_X - test_X.min()) / (test_X.max() - test_X.min())

    # print(train_y)

    # to one hot
    # train_y = np.eye(2)[train_y]

    # to tensor
    train_X = torch.from_numpy(train_X).float()
    train_y = torch.from_numpy(train_y).float()
    test_X = torch.from_numpy(test_X).float()

    print(train_X.shape, train_y.shape, test_X.shape)

    # data loader
    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    # train
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    milestones = [50, 100, 150, 200, 250]
    milestones = [a * len(train_loader) for a in milestones]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=milestones,
                                                     gamma=0.5)

    model.train()
    model = model.cuda()

    for epoch in range(args.epochs):
        running_loss = 0.0
        accuarcy = 0.0
        total = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs = model(inputs)
            # _, preds = torch.max(outputs, 1)
            # binary cross entropy
            outputs = torch.sigmoid(outputs)
            outputs = outputs.view(-1)
            # print(outputs.shape, labels.shape)
            # print(outputs, labels)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            # binary classification
            outputs = outputs.cpu().detach().numpy()
            outputs = np.where(outputs > 0.5, 1, 0)
            labels = labels.cpu().detach().numpy()
            accuarcy += np.sum(outputs == labels)
            total += len(labels)
            # print(accuarcy, total)

        if epoch % 1 == 0:
            print('Epoch [{}/{}], Loss: {:.4f} Accuarcy: {:.4f}'.format(
                epoch + 1, args.epochs, running_loss / len(train_loader), accuarcy / total))

    # test
    model.eval()
    test_pred = []
    with torch.no_grad():
        # test
        for i, (inputs) in enumerate(test_X):
            inputs = inputs.cuda()
            outputs = model(inputs)
            outputs = torch.sigmoid(outputs)
            outputs = outputs.view(-1)
            outputs = outputs.cpu().detach().numpy()
            outputs = np.where(outputs > 0.5, 1, 0)
            preds = outputs.tolist()
            test_pred.extend(preds)

    # save model
    torch.save(model.state_dict(),
               os.path.join(args.model_path, f'model_{in_feature}_{int(time.time())}.pt'))
    return test_pred


def main():
    os.makedirs(args.result_path, exist_ok=True)
    predict_label = []
    trial_id = [i for i in range(1, 1601)]

    train_data, train_label, test_data = get_data(0, 10)
    result = train_test(train_data, train_label, test_data, in_channels=30, in_feature=1500)
    predict_label.extend(result)

    train_data, train_label, test_data = get_data(10, 20)
    result = train_test(train_data, train_label, test_data, in_channels=30, in_feature=3750)
    predict_label.extend(result)

    dataframe = pd.DataFrame({'TrialId': trial_id, 'Label': predict_label})
    dataframe = dataframe.astype({'TrialId': 'int', 'Label': 'int'})
    dataframe.to_csv(os.path.join(args.result_path, "sample_submission.csv"),
                     index=False,
                     sep=',')


if __name__ == '__main__':
    main()