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

def objective(trial, x, y):

    n_d = trial.suggest_int("n_d", low = 8, high = 64, step = 8)
    n_a = trial.suggest_int("n_a", low = 8, high = 64, step = 8)
    gamma = trial.suggest_float("gamma", 1.0, 2.0)
    momentum = trial.suggest_float("momentum", 0.01, 0.4)
    mask_type = trial.suggest_categorical("mask_type", ['sparsemax', 'entmax'])

    
    tabnet_params = dict(n_d=n_d, n_a=n_a, gamma=gamma,
                     momentum = momentum,
                     optimizer_fn=torch.optim.Adam,
                     optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
                     mask_type = mask_type,
                     scheduler_params=dict(mode="min",
                                           patience=trial.suggest_int("patienceScheduler",low=3,high=10), # changing sheduler patience to be lower than early stopping patience 
                                           min_lr=1e-5,
                                           factor=0.5,),
                     scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                     verbose=0,
                     ) #early stopping

    kf = model_selection.StratifiedKFold(n_splits=5)
    roc_auc = []

    for idx in kf.split(X=x, y=y):

        train_idx, test_idx = idx[0], idx[1]
        xtrain  = x[train_idx]
        ytrain = y[train_idx]
        
        xtest = x[test_idx]
        ytest = y[test_idx]
        
        classifier = TabNetClassifier(**tabnet_params)
        classifier.fit(xtrain, ytrain, max_epochs=trial.suggest_int('epochs', 1, 100))
        preds = classifier.predict_proba(xtest)
        # print(preds)
        fold_roc = metrics.roc_auc_score(ytest, preds, multi_class='ovr')
        roc_auc.append(fold_roc)

    
    return max(roc_auc)





if __name__ == "__main__":

    path = sys.argv[1]
    test_path = sys.argv[2]
    output_path = sys.argv[3]

    df = pd.read_csv(path)
    X = df.drop("Output", axis=1).values
    
    y = df.Output.values
    print(len(y))

    df = pd.read_csv(test_path)
    X_test = df.values
    

    
    study = optuna.create_study(direction = "minimize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=5)

    #Instance with tuned hyperparameters

    
    optimised_params = dict (n_d = study.best_params['n_d'],
        n_a = study.best_params['n_a'],
        gamma = study.best_params['gamma'],
        momentum = study.best_params['momentum'],
        mask_type = study.best_params['mask_type'],
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
        scheduler_params=dict(mode="min",
                              min_lr=1e-5,
                              factor=0.5,),
                              scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
                              verbose=0,
        )
    
    classifier = TabNetClassifier(**optimised_params)
    classifier.fit(X_train = X, y_train = y, max_epochs = study.best_params['epochs'])

    
    y_pred = classifier.predict(X_test)
    final_roc = metrics.roc_auc_score(y, classifier.predict_proba(X), multi_class='ovr')

    print('roc_auc_score:', final_roc)
    
    # print(y_pred) 
    pd.DataFrame(y_pred).to_csv(output_path, index=False)




