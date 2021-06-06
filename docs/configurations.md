# Configurations

To create your own configuration files, please carefully read the details of this documents.

## Overview

The configuration files are all written in YAML format.

## Locations

All the default/provided configuration files are stored in the `configs` folder, in which

- `DeepCov.yaml` (DeepCov configuration) is the configuration file of model 1;
- `ResPRE.yaml` (ResPRE configuration), is the configuration file of model 2;
- `FCResPRE.yaml` (FCResPRE configuration), is the configuration file of model 2.5;
- `CbamResPRE.yaml` (CbamResPRE configuration), is the configuration file of model 3;
- `CbamFCResPRE.yaml` (CbamFCResPRE configuration), is the configuration file of model 3.5;
- `SEResPRE.ymal` (SEResPRE configuration), is the configuration file of model 4;
- `SEFCResPRE.yaml` (SEFCResPRE configuration), is the configuration file of model 4.5;
- `NLResPRE.yaml` (NLResPRE configuration), is the configuration file of model 5;
- `HaloResPRE.yaml` (HaloResPRE configuration), is the configuration file of model 6;

You can refer to the given configuration file to set up your own configurations. In the next sections we will introduce the possible items in the configuration of each model.

## DeepCov (Model 1) Configurations

Here are the possible items in the configuration file of DeepCov model (model 1).
- `batch_size`: int, optional, default: 4, the batch size of the training/evaluation stage of the DeepCov model;
- `zipped`: bool, optional, default: True, whether the data files are zipped;
- `multigpu`: bool, optional, default: True, whether to multiple GPUs in training/inference;
- `max_epoch`: int, optional, default: 30, the maximum of epoch num;
- `adam_beta1`: float, optional, default: 0.9, the `beta1` parameter of AdamW optimizer;
- `adam_beta2`: float, optional, default: 0.999, the `beta2` parameter of AdamW optimizer;
- `adam_weight_decay`: float, optional, default: 0.01, the `weight_decay` parameter of AdamW optimizer;
- `adam_eps`: float, optional, default: 1e-6, the `eps` parameter of AdamW optimizer;
- `learning_rate`: float, optional, default: 0.001, the initial learning rate of AdamW optimizer;
- `milestones`: list of int, optional, default: [15], the `milestones` parameter of learning rate scheduler;
- `gamma`: float, optional, default: 0.1, the `gamma` parameter of learning rate scheduler;
- `checkpoint_dir`: str, optional, default: 'checkpoint', the folder name where checkpoint file will be stored;
- `temp_dir`: str, optional, default: 'temp', the folder name where temporary files will be stored;
- `softmax`: bool, optional, default: True, whether the output will be activated by softmax function;
- `network`:
    - `name`: str, optional, default: 'SampleNet', the name of network;
    - `input_channel`: int, optional, default: 441, the dimension of input;
    - `output_channel`: int, optional, default: 10, the dimension of output;
    - `hidden_channel`: int, optional, default: 64, the dimension of hidden layer;
    - `blocks`: int, optional, default: 10, the number of convolutional layers;

## ResPRE (Model 2) Configurations

Here are the possible items in the configuration file of ResPRE model (model 2).

- `batch_size`, `zipped`, `multigpu`, `max_epoch`, `adam_beta1`, `adam_beta2`, `adam_weight_decay`, `adam_eps`, `learning_rate`, `milestones`, `gamma`, `checkpoint_dir`, `temp_dir`, `softmax` parameters have already described in model 1.
- `network`:
    - `name`, `input_channel`, `output_channel`, `hidden_channel`, `blocks` parameters have already described in model 1.
    - `droprate`: float, optional, default: 0.2, the `dropout rate` parameter of residual basic block;

## FCResPRE (Model 2.5) Configurations

It is the same as configurations of model 2.

## CbamResPRE (Model 3) Configurations

Here are the possible items in the configuration file of CbamResPRE (model 3).

- `batch_size`, `zipped`, `multigpu`, `max_epoch`, `adam_beta1`, `adam_beta2`, `adam_weight_decay`, `adam_eps`, `learning_rate`, `milestones`, `gamma`, `checkpoint_dir`, `temp_dir`, `softmax` parameters have already described in model 2.
- `network`:
  - `name`, `input_channel`, `output_channel`, `hidden_channel`, `blocks` parameters have already described in model 2.
  - `block_type`: str, optional, default:　'BasicBlock', the type of residual basic block



## CbamFCResPRE (Model 3.5) Configurations

It is the same as configurations of model 3.

## SEResPRE (Model 4) Configurations

Here are the possible items in the configuration file of SEResPRE (model 4).

- `batch_size`, `zipped`, `multigpu`, `max_epoch`, `adam_beta1`, `adam_beta2`, `adam_weight_decay`, `adam_eps`, `learning_rate`, `milestones`, `gamma`, `checkpoint_dir`, `temp_dir`, `softmax` parameters have already described in model 3.
- `network`:
  - `name`, `input_channel`, `output_channel`, `hidden_channel`, `blocks`, `block_type` parameters have already described in model 3.

  -`reduction`: int, optional, default: 16, the reduction rate of SE　attention module.


## SEFCResPRE (Model 4.5) Configurations

It is the same as configurations of model 4.


## NLResPRE (Model 5) Configurations

Here are the possible items in the configuration file of NLResPRE (model 5).

- `batch_size`, `zipped`, `multigpu`, `max_epoch`, `adam_beta1`, `adam_beta2`, `adam_weight_decay`, `adam_eps`, `learning_rate`, `milestones`, `gamma`, `checkpoint_dir`, `temp_dir`, `softmax` parameters have already described in model 2.
- `network`:
  - `name`, `input_channel`, `output_channel`, `hidden_channel`, `blocks` parameters have already described in model 2.


## HaloResPRE (Model 6) Configurations

Here are the possible items in the configuration file of HaloResPRE (model 5).

- `batch_size`, `zipped`, `multigpu`, `max_epoch`, `adam_beta1`, `adam_beta2`, `adam_weight_decay`, `adam_eps`, `learning_rate`, `milestones`, `gamma`, `checkpoint_dir`, `temp_dir`, `softmax` parameters have already described in model 2.
- `network`:
  - `name`, `input_channel`, `output_channel`, `hidden_channel`, `blocks` parameters have already described in model 2.
  - `block_size`: int, optional, default: 8, <font color=red>懵逼</font>
  - `halo_size`: int, optional, default: 4, <font color=red>懵逼</font>
  - `dim_head`: int, optional, default: 16, <font color=red>懵逼</font>
  - `heads`: int, optional, default: 4, <font color=red>懵逼</font>
