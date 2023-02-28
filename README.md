# auto-tabnet
Auto-TabNet is the implementation of Google's TabNet model using dreamquark-ai's pytorch implementation along with hyperparameter search with Optuna.

Google's TabNet was proposed in 2019 with the idea of effectively using deep neural networks for tabular data.

TabNet is a complex model composed of a feature transformer, attentive transformer, and feature masking, that soft feature selection with controllable sparsity in end-to-end learning. The reason for the high performance of TabNet is that it focuses on the most important features that have been considered by the Attentive Transformer. The Attentive Transformer performs feature selection to select which model features to reason from at each step in the model, and a Feature Transformer processes feature into more useful representations and learn complex data patterns, which improve interpretability and help it learn more accurate models.

The key benefit TabNet has is its explainability, as it uses instance-wise feature selection using masks in its encoder. This allows the model’s learning capacity to be focused on important features. It can capitalize on the forthcoming visualization of masks (as it provides explainability) which can be experimented and which features are being used at a prediction level can be explored.
