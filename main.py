import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn import preprocessing
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

def optimize(trial, x, y):

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
    roc_auc = []

    for idx in kf.split(X=x, y=y):

        train_idx, test_idx = idx[0], idx[1]
        xtrain  = x[train_idx]
        ytrain = y[train_idx]

        xtest = x[test_idx]
        ytest = y[test_idx]

        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        print(preds)
        fold_roc = metrics.roc_auc_score(ytest, preds, multi_class='ovr')
        roc_auc.append(fold_roc)

    
    return roc_auc





if __name__ == "__main__":

    path = sys.argv[1]
    test_path = sys.argv[2]
    output_path = sys.argv[3]

    df = pd.read_csv(path)
    X = df.drop("Output", axis=1).values
    y = df.Output.values

    df = pd.read_csv(test_path)
    X_test = df.values
    

    optimization_function = partial(optimize, x=X, y=y)
    study = optuna.create_study(direction = "minimize")
    study.optimize(optimization_function, n_trials = 5)

    #Instance with tuned hyperparameters
    optimised_tabnet = TabNetClassifier(n_d = study.best_params['n_d'],
        n_a = study.best_params['n_a'],
        gamma = study.best_params['gamma'],
        momentum = study.best_params['momentum'],
        mask_type = study.best_params['mask_type'],
                                     n_jobs=2)
    
    optimised_tabnet.fit(X,y)

    
    y_pred = optimised_tabnet.predict_proba(X_test)
    print(y_pred) 
    dout = y_pred.to_csv(output_path)




