import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pytorch
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.metrics import Metric
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import QuantileTransformer
from sklearn import preprocessing
import torch.optim as optim
import torch.nn.functional as F
from sklearn import model_selection
import optuna
from optuna import create_study
from optuna.samplers import TPESampler

import pytorch_tabnet
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.metrics import Metric
from sklearn.metrics import roc_auc_score


from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import pipeline

from functools import partial

def optimize(trials, x, y):

    n_d = trial.suggest_categorical("n_d", [8, 16, 32, 64])
    n_a = trial.suggest_categorical("n_a", [8, 16, 32, 64])
    gamma = trial.suggest_int("gamma", 1.0, 2.0)
    momentum = trial.suggest_int("momentum", 0.01, 0.4)
    mask_type = trial.suggest_categorical("mask_type", ['sparsemax', 'entmax'])

    
    model = TabNetClassifier(
        n_d = n_d,
        n_a = n_a,
        gamma = gamma,
        momentum = momentum,
        mask_type = mask_type)

    kf = model_selection.StratifiedKFold(n_splits=5)
    accuracies = []

    for idx in kf.split(X=x, y=y):

        train_idx, test_idx = idx[0], idx[1]
        xtrain  = x[train_idx]
        ytrain = y[train_idx]

        xtest = x[test_idx]
        ytest = y[test_idx]

        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        fold_acc = metrics.accuracy_score(ytest, preds)
        accuracies.append(fold_acc)

    return -1.0 * np.mean(accuracies)

if __name__ == "__main__":

    df = pd.read_csv("./train.csv")
    X = df.drop("target", axis=1).values
    y = df.target.values

    optimization_function = partial(optimize, x=X, y=y)
    study = optuna.create_study(directions = "minimize")
    study.optimize(optimization_function, n_trials = 15)





