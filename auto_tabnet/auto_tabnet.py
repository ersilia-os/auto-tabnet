import pandas as pd
import numpy as np
import torch
from typing import Dict


from sklearn import model_selection
import optuna
from optuna import create_study

import pytorch_tabnet
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.metrics import Metric

from sklearn import metrics
from sklearn import model_selection


class AutoTabnetClassifier:
    def __init__(
        self, X_train: pd.DataFrame, y_train: pd.DataFrame, X_test: pd.DataFrame
    ) -> None:
        """AutoTabnet class to perform the classfication on tabular data using Google's TabNet.

        Args:
            X_train (pd.DataFrame): Training features in a pandas DataFrame for training the TabNetClassifier.
            y_train (pd.DataFrame): Output of training data in a pandas DataFrame for training the TabNetClassifier.
            X_test (pd.DataFrame): Test data in a pandas DataFrame to get the prediction from Optimised TabnetClassifier.
        """

        self.X_train = X_train.values
        self.y_train = y_train.values
        self.X_test = X_test.values
        self.y_pred = None
        self.final_roc = 0
        self.predicted = False
        self.best_params = dict()

    def _objective(self, trial):
        """Optimises the hyperparameters for the training the TabNet classifier.

        Returns:
            _Float_: Best roc_auc_score in each trial of the Optuna study.
        """

        x = self.X_train
        y = self.y_train

        n_d = trial.suggest_int("n_d", low=8, high=64, step=8)
        n_a = trial.suggest_int("n_a", low=8, high=64, step=8)
        gamma = trial.suggest_float("gamma", 1.0, 2.0)
        momentum = trial.suggest_float("momentum", 0.01, 0.4)
        mask_type = trial.suggest_categorical("mask_type", ["sparsemax", "entmax"])

        tabnet_params = dict(
            n_d=n_d,
            n_a=n_a,
            gamma=gamma,
            momentum=momentum,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
            mask_type=mask_type,
            scheduler_params=dict(
                mode="min",
                patience=trial.suggest_int(
                    "patienceScheduler", low=3, high=10
                ),  # changing sheduler patience to be lower than early stopping patience
                min_lr=1e-5,
                factor=0.5,
            ),
            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
            verbose=0,
        )

        kf = model_selection.StratifiedKFold(n_splits=5)
        roc_auc = []

        for idx in kf.split(X=x, y=y):
            train_idx, test_idx = idx[0], idx[1]
            xtrain = x[train_idx]
            ytrain = y[train_idx]

            xtest = x[test_idx]
            ytest = y[test_idx]

            classifier = TabNetClassifier(**tabnet_params)
            classifier.fit(
                xtrain, ytrain, max_epochs=trial.suggest_int("epochs", 1, 100)
            )
            preds = classifier.predict_proba(xtest)
            fold_roc = metrics.roc_auc_score(ytest, preds, multi_class="ovr")
            roc_auc.append(fold_roc)

        return max(roc_auc)

    def _predict(self):
        """Predicts the roc_auc_score and classified results on test data by studying best hyperparameters."""

        X = self.X_train
        y = self.y_train

        X_test = self.X_test
        # Creating the optuna study

        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: self._objective(trial), n_trials=5)

        # TabNet Instance with tuned hyperparameters

        optimised_params = dict(
            n_d=study.best_params["n_d"],
            n_a=study.best_params["n_a"],
            gamma=study.best_params["gamma"],
            momentum=study.best_params["momentum"],
            mask_type=study.best_params["mask_type"],
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
            scheduler_params=dict(
                mode="min",
                min_lr=1e-5,
                factor=0.5,
            ),
            scheduler_fn=torch.optim.lr_scheduler.ReduceLROnPlateau,
            verbose=0,
        )

        self.best_params = optimised_params

        classifier = TabNetClassifier(**optimised_params)
        classifier.fit(X_train=X, y_train=y, max_epochs=study.best_params["epochs"])

        self.y_pred = classifier.predict(X_test)
        self.final_roc = metrics.roc_auc_score(
            y, classifier.predict_proba(X), multi_class="ovr"
        )

    def predict(self) -> np.ndarray:
        """Returns the classification results of test data

        Returns:
            np.ndarray: Output as predicted classes
        """

        if not self.predicted:
            self._predict()
            self.predicted = True

        return self.y_pred

    def get_roc_auc_score(self) -> float:
        """Returns the roc_auc_score of the optimised TabnetClassifier

        Returns:
            Float: roc_auc score
        """

        if not self.predicted:
            self._predict()
            self.predicted = True

        return self.final_roc

    def get_best_params(self) -> Dict:
        """Best Parameters obtained after optimising the hyperparameters

        Returns:
            Dict: best parameters in a dict form
        """

        if not self.predicted:
            self._predict()
            self.predicted = True

        return self.best_params
