# Configurations

To create your own configuration files, please carefully read the details of this documents.

## Overview

The configuration files are all written in YAML format.

## Locations

All the default/provided configuration files are stored in the `configs` folder, in which

- `DeepCov.yaml` (default configuration) is the configuration file of model 1;
- `ResPRE.yaml` (DeepCov configuration), is the configuration file of model 2;
- `NLResPRE.yaml` (ResPRE configuration), is the configuration file of model 3;

You can refer to the given configuration file to set up your own configurations. In the next sections we will introduce the possible items in the configuration of each model.

## DeepCov (Model 1) Configurations

Here are the possible items in the configuration file of DeepCov model (model 1).
- `batch_size`: int, optional, default: 2, the batch size of the training/evaluation stage of the DeepCov model;
- `zipped`: bool, optional, default: True, whether the data files are zipped;
- `multigpu`: bool, optional, default: False, whether to multiple GPUs in training/inference;
- `max_epoch`: int, optional, default: 50, the maximum of epoch num;
- `adam_beta1`: float, optional, default: 0.9, the `beta1` parameter of AdamW optimizer;
- `adam_beta2`: float, optional, default: 0.999, the `beta2` parameter of AdamW optimizer;
- `adam_weight_decay`: float, optional, default: 0.01, the `weight_decay` parameter of AdamW optimizer;
- `adam_eps`: float, optional, default: 0.000001, the `eps` parameter of AdamW optimizer;
- `learning_rate`: float, optional, default: 0.005, the initial learning rate of AdamW optimizer;
- `milestones`: list of int, optional, default: [15, 25, 35, 45], the `milestones` parameter of learning rate scheduler;
- `gamma`: float, optional, default: 0.2, the `gamma` parameter of learning rate scheduler;
- `checkpoint_dir`: str, optional, default: 'checkpoint', the folder name where checkpoint file will be stored;
- `temp_dir`: str, optional, default: 'temp', the folder name where temporary files will be stored;
- `softmax`: bool, optional, default: True, whether the output will be activated by softmax function;
- `network`:
    - `name`: str, optional, default: 'DeepCov', the name of network;
    - `input_channel`: int, optional, default: 441, the dimension of input;
    - `output_channel`: int, optional, default: 10, the dimension of output;
    - `hidden_channel`: int, optional, default: 64, the dimension of hidden layer;
    - `blocks`: int, optional, default: 10, the number of convolutional layers;

## ResPRE (Model 2) Configurations

Here are the possible items in the configuration file of Bert model (model 2).

- `max_epoch`: int, optional, default: 2000, the maximum of epoch num;
- `bert_cased`: bool, optional, default: False, whether the bert model is case-sensitive;
- `multigpu`: bool, optional, default: False, whether to multiple GPUs in training/inference;
- `adam_beta1`: float, optional, default: 0.9, the `beta1` parameter of AdamW optimizer;
- `adam_beta2`: float, optional, default: 0.999, the `beta2` parameter of AdamW optimizer;
- `adam_weight_decay`: float, optional, default: 0.01, the `weight_decay` parameter of AdamW optimizer;
- `adam_eps`: float, optional, default: 1e-6, the `eps` parameter of AdamW optimizer.
- `learning_rate`: float, optional, default: 0.01, the initial learning rate of Adam optimizer;
- `batch_size`: int, optional, default: 16, the batch size of the training/evaluation stage of the bert model;
- `max_length`: int, optional, default: 512, the maximum length of context input to Bert model;
- `seq_len`: int, optional, default: 50, the maximum length of citation text;
- `end_year`: int, optional, default: 2020, the end year of the papers to train and evaluate;
- `frequency`: int, optional, default: 5, the minimum citations of a referenced papers to be counted;
- `recall_K`, list of int, optional, default: [5, 10, 30, 50, 80], the Ks of the Recall@K metrics;
- `K`: int, optional, default: 10, the number of searching items in inference stage;
- `stats_dir`: str, optional, default: 'stats/vgae', the directory of the statistics (including checkpoints and others);
- `data_path`: str, optional, default: 'data/citation.csv', the path to the data.

## NLResPRE (Model 3) Configurations

Here are the possible items in the configuration file of Citation-Bert model (model 3).

- `max_epoch`: int, optional, default: 2000, the maximum of epoch num;
- `multigpu`: bool, optional, default: False, whether to multiple GPUs in training/inference;
- `embedding_dim`: int, optional, default: 768, the embedding dimension of each node;
- `cosine_softmax_S`: float, optional, default: 1, the coefficient of the score combining process;
- `bert_cased`: bool, optional, default: False, whether the bert model is case-sensitive;
- `adam_beta1`: float, optional, default: 0.9, the `beta1` parameter of AdamW optimizer;
- `adam_beta2`: float, optional, default: 0.999, the `beta2` parameter of AdamW optimizer;
- `adam_weight_decay`: float, optional, default: 0.01, the `weight_decay` parameter of AdamW optimizer;
- `adam_eps`: float, optional, default: 1e-6, the `eps` parameter of AdamW optimizer.
- `learning_rate`: float, optional, default: 0.01, the initial learning rate of Adam optimizer;
- `batch_size`: int, optional, default: 16, the batch size of the training/evaluation stage of the bert model;
- `max_length`: int, optional, default: 512, the maximum length of context input to Bert model;
- `seq_len`: int, optional, default: 50, the maximum length of citation text;
- `end_year`: int, optional, default: 2020, the end year of the papers to train and evaluate;
- `frequency`: int, optional, default: 5, the minimum citations of a referenced papers to be counted;
- `recall_K`, list of int, optional, default: [5, 10, 30, 50, 80], the Ks of the Recall@K metrics;
- `K`: int, optional, default: 10, the number of searching items in inference stage;
- `stats_dir`: str, optional, default: 'stats/vgae', the directory of the statistics (including checkpoints and others);
- `data_path`: str, optional, default: 'data/citation.csv', the path to the data;
- `embedding_path`: str, optional, default: 'stats/vgae/specter_embedding.npy', the path to the embedding file of papers.