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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier, BaggingClassifier, RandomTreesEmbedding, IsolationForest, VotingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

opt = argparse.ArgumentParser()
opt.add_argument('--data_path', type=str, default='data/', help='data path')
opt.add_argument('--result_path', type=str, default='result/', help='result path')


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


def train_test(train_data, train_label, test_data):
    b, a = signal.butter(8, [8 * 2 / 250, 30 * 2 / 250], 'bandpass')
    train_data = signal.filtfilt(b, a, train_data)
    test_data = signal.filtfilt(b, a, test_data)

    train_X = train_data
    train_y = train_label
    test_X = test_data

    # train_X = (train_data - np.mean(train_data, axis=0)) / np.max(train_data)
    # test_X = (test_data - np.mean(test_data, axis=0)) / np.max(train_data)

    # train_X = (train_data - np.mean(train_data, axis=0)) / np.std(train_data)
    # test_X = (test_data - np.mean(test_data, axis=0)) / np.std(train_data)

    csp = CSP(n_components=30, reg=None, log=False, norm_trace=False)
    rfc = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    efc = ExtraTreesClassifier(n_estimators=100, n_jobs=-1)
    abc = AdaBoostClassifier(n_estimators=100)
    gbc = GradientBoostingClassifier(n_estimators=100)
    bgc = BaggingClassifier(n_estimators=100)

    vc = VotingClassifier(estimators=[('rfc', rfc), ('efc', efc), ('abc', abc), ('gbc', gbc),
                                      ('bgc', bgc)],
                          voting='hard')

    rte = RandomTreesEmbedding(n_estimators=100)
    ifr = IsolationForest(n_estimators=100)

    # regression
    rfr = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    gbr = GradientBoostingRegressor(n_estimators=100)
    abr = AdaBoostRegressor(n_estimators=100)
    er = ExtraTreesRegressor(n_estimators=100, n_jobs=-1)
    br = BaggingRegressor(n_estimators=100)
    vr = VotingRegressor(estimators=[('rfr', rfr), ('gbr', gbr), ('abr',
                                                                  abr), ('er', er), ('br', br)])

    # classifier
    # lr = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000)
    # sgd = SGDClassifier(loss='log', max_iter=1000)
    # svc = SVC(kernel='rbf', probability=True)
    # svr = SVR(kernel='rbf')
    # omp = OrthogonalMatchingPursuit()
    # rlr = RandomizedLogisticRegression()

    # svc = SVC(C=1, kernel='rbf', probability=True)
    # clf = Pipeline([('CSP', csp), ('RFC', rfc)])
    # clf = Pipeline([('CSP', csp), ('EFC', efc)])
    # clf = Pipeline([('CSP', csp), ('ABC', abc)])
    # clf = Pipeline([('CSP', csp), ('GBC', gbc)])
    # clf = Pipeline([('CSP', csp), ('VC', vc)])
    # clf = Pipeline([('CSP', csp), ('BGC', bgc)])
    # clf = Pipeline([('CSP', csp), ('RTE', rte)])
    # clf = Pipeline([('CSP', csp), ('IFR', ifr)])
    # clf = Pipeline([('CSP', csp), ('RFR', rfr)])
    # clf = Pipeline([('CSP', csp), ('EFR', efr)])
    # clf = Pipeline([('CSP', csp), ('ABR', abr)])
    # clf = Pipeline([('CSP', csp), ('GBR', gbr)])
    # clf = Pipeline([('CSP', csp), ('BGR', bgr)])
    # clf = Pipeline([('CSP', csp), ('IFRR', ifrr)])
    # clf = Pipeline([('CSP', csp), ('SVC', svc)])
    # clf = Pipeline([('CSP', csp), ('SVR', svr)])
    # clf = Pipeline([('CSP', csp), ('OMP', omp)])
    clf = Pipeline([('CSP', csp), ('VR', vr)])

    clf.fit(train_X, train_y)
    train_pred = clf.predict(train_X)
    throld = 0.75
    print(train_pred[:10])
    train_pred[train_pred > throld] = '1'
    train_pred[train_pred <= throld] = '0'
    report = classification_report(train_y, train_pred)

    test_y = clf.predict(test_X)
    test_y[test_y > throld] = '1'
    test_y[test_y <= throld] = '0'

    return test_y, report


def train_eeg(args):

    predict_label = []
    trial_id = [i for i in range(1, 1601)]

    reports = []

    train_data, train_label, test_data = get_data(0, 10)
    result, report = train_test(train_data, train_label, test_data)
    print(report)
    predict_label.extend(result)

    train_data, train_label, test_data = get_data(10, 20)
    result, report = train_test(train_data, train_label, test_data)
    print(report)
    predict_label.extend(result)

    dataframe = pd.DataFrame({'TrialId': trial_id, 'Label': predict_label})
    dataframe = dataframe.astype({'TrialId': 'int', 'Label': 'int'})
    dataframe.to_csv(os.path.join(args.result_path, "sample_submission.csv"),
                     index=False,
                     sep=',')


if __name__ == '__main__':
    args = opt.parse_args()

    os.makedirs(args.result_path, exist_ok=True)

    train_eeg(args)