#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   src\main.py
# @Time    :   2022-02-22 00:13:29
# @Author  :   Bingjie Yan
# @Email   :   bj.yan.pa@qq.com
# @License :   Apache License 2.0

import os
import sys
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import argparse
import numpy as np
from scipy import signal
import pandas as pd
from mne.decoding import CSP
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import warnings
warnings.filterwarnings("ignore")

opt = argparse.ArgumentParser()
opt.add_argument('--data_path', type=str, default='data/', help='data path')
opt.add_argument('--result_path', type=str, default='result/', help='result path')


def train_eeg(args):

    file_id = [
        '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14',
        '15', '16', '17', '18', '19', '20'
    ]
    predict_label = []
    trial_id = [i for i in range(1, 1601)]

    reports = []

    csp = CSP(n_components=20, reg=None, log=False, norm_trace=False)
    svc = SVC(kernel='linear', probability=True, class_weight='balanced')
    clf = Pipeline([('CSP', csp), ('SVC', svc)])

    train_data_total = None
    train_label_total = None
    test_data_total = None

    for i in range(10):
        print(f'Processing {i}')
        train_data = np.load(os.path.join(args.data_path, 'train_data_' + file_id[i] + '.npy'))
        train_label = np.load(os.path.join(args.data_path,
                                           'train_label_' + file_id[i] + '.npy'))
        test_data = np.load(os.path.join(args.data_path, 'test_data_' + file_id[i] + '.npy'))

        b, a = signal.butter(8, [8 / 125, 30 / 125], 'bandpass')
        train_data = signal.filtfilt(b, a, train_data)
        test_data = signal.filtfilt(b, a, test_data)

        if train_data_total is None:
            train_data_total = train_data
            train_label_total = train_label
            test_data_total = test_data
        else:
            train_data_total = np.concatenate((train_data_total, train_data), axis=0)
            train_label_total = np.concatenate((train_label_total, train_label), axis=0)
            test_data_total = np.concatenate((test_data_total, test_data), axis=0)

    clf.fit(train_data_total, train_label_total)

    predict_label.extend(clf.predict(test_data_total))

    # reports.append(
    #     classification_report(train_data_total,
    #                           clf.predict(train_label_total),
    #                           target_names=['0', '1']))

    csp = CSP(n_components=20, reg=None, log=False, norm_trace=False)
    svc = SVC(kernel='linear', probability=True, class_weight='balanced')
    clf = Pipeline([('CSP', csp), ('SVC', svc)])

    train_data_total = np.array([])
    train_label_total = np.array([])
    test_data_total = np.zeros([])

    train_data_total = None
    train_label_total = None
    test_data_total = None

    for i in range(10, 20):
        print(f'Processing {i}')
        train_data = np.load(os.path.join(args.data_path, 'train_data_' + file_id[i] + '.npy'))
        train_label = np.load(os.path.join(args.data_path,
                                           'train_label_' + file_id[i] + '.npy'))
        test_data = np.load(os.path.join(args.data_path, 'test_data_' + file_id[i] + '.npy'))

        b, a = signal.butter(8, [8 / 125, 30 / 125], 'bandpass')
        train_data = signal.filtfilt(b, a, train_data)
        test_data = signal.filtfilt(b, a, test_data)

        if train_data_total is None:
            train_data_total = train_data
            train_label_total = train_label
            test_data_total = test_data
        else:
            train_data_total = np.concatenate((train_data_total, train_data), axis=0)
            train_label_total = np.concatenate((train_label_total, train_label), axis=0)
            test_data_total = np.concatenate((test_data_total, test_data), axis=0)

    clf.fit(train_data_total, train_label_total)

    predict_label.extend(clf.predict(test_data_total))

    print(train_data_total.shape, train_label_total.shape, test_data_total.shape)

    # reports.append(
    #     classification_report(train_data_total,
    #                           clf.predict(train_label_total),
    #                           target_names=['0', '1']))

    dataframe = pd.DataFrame({'TrialId': trial_id, 'Label': predict_label})
    dataframe.to_csv(os.path.join(args.result_path, "sample_submission.csv"),
                     index=False,
                     sep=',')

    for i in reports:
        print(i)


if __name__ == '__main__':
    args = opt.parse_args()

    os.makedirs(args.result_path, exist_ok=True)

    train_eeg(args)