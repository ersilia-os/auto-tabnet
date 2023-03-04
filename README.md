# Auto-Tabnet
Auto-TabNet is the implementation of Google's TabNet model using dreamquark-ai's pytorch implementation along with hyperparameter search with Optuna.

Google's TabNet was proposed in 2019 with the idea of effectively using deep neural networks for tabular data.

TabNet is a complex model composed of a feature transformer, attentive transformer, and feature masking, that soft feature selection with controllable sparsity in end-to-end learning. The reason for the high performance of TabNet is that it focuses on the most important features that have been considered by the Attentive Transformer. The Attentive Transformer performs feature selection to select which model features to reason from at each step in the model, and a Feature Transformer processes feature into more useful representations and learn complex data patterns, which improve interpretability and help it learn more accurate models.



## Requirements

-
 ```sh
python 3.7 >
```

- 
```sh
pip (python package manager)
```


## Installation

With pip:

```sh
pip install auto-tabnet
```

## Source Code

If you want to use it locally within a pip virtualenv:

- Clone the repository

```sh
git clone https://github.com/Femme-js/auto-tabnet.git
```
- Create a pip virtual environment.

```sh
virtualenv env
```
- Install the dependencies from requirements.txt file.

```sh
pip install -r requirements.txt
```


## Usage

```sh
from auto_tabnet import AutoTabnetClassifier

clf = AutoTabnetClassifier(X, y, X_test)

```
To get the prediction on test data.

```sh
results = clf.predict()

```
To get the auc_roc_score:

```sh
results = clf.get_roc_auc_score()

```
To get the best hyperparamters tuned by optuna:

```sh
results = clf.get_best_params()

```

The targets on y_train should contain a unique type (e.g. they must all be strings or integers).


