# Configurations

To create your own configuration files, please carefully read the details of this documents.

## Overview

The configuration files are all written in YAML format.

## Locations

All the provided default configuration files are stored in the `configs` folder, in which

- `DeepCov.yaml` is the default configuration file of DeepCov model;
- `ResPRE.yaml` is the default configuration file of ResPRE model;
- `CbamResPRE.yaml` is the default configuration file of Cbam-ResPRE model;
- `SEResPRE.ymal` is the default configuration file of SE-ResPRE model;
- `HaloResPRE.yaml` is the default configuration file of Halo-ResPRE model;
- `NLResPRE.yaml` is the default configuration file of NL-ResPRE model;
- `FCResPRE.yaml` is the default configuration file of FC-ResPRE model;
- `CbamFCResPRE.yaml` is the default configuration file of Cbam-FC-ResPRE model;
- `SEFCResPRE.yaml` is the default configuration file of SE-FC-ResPRE model;
- `DilatedResnet34.yaml` is the default configuration file of Dilated Resnet 34 model.

You can refer to the given configuration file to set up your own configurations. In the next sections we will introduce the possible items in the configuration of each model.

## Basic Configurations

Here are the basic items in the configuration file.

- `batch_size`: int, optional, default: 4, the batch size of the training/evaluation stage of the DeepCov model.
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
- `network`: dict, optional, default {}, which is the configuration of the network.

## Network Configuration: DeepCov

Here are the items that describes the DeepCov network in the configuration file.

- `name`: "DeepCov".
- `input_channel`: int, optional, default: 441, the dimension of input;
- `output_channel`: int, optional, default: 10, the dimension of output;
- `hidden_channel`: int, optional, default: 64, the dimension of hidden layer;
- `blocks`: int, optional, default: 10, the number of convolutional layers;

## Network Configurations: ResPRE

Here are the items that describes the ResPRE network in the configuration file.

- `name`: "ResPRE".
- `input_channel`: int, optional, default: 441, the dimension of input;
- `output_channel`: int, optional, default: 10, the dimension of output;
- `hidden_channel`: int, optional, default: 64, the dimension of hidden layer;
- `blocks`: int, optional, default: 22, the number of convolutional layers;
- `droprate`: float, optional, default: 0.2, the dropout rate in the residual block.

## Network Configuraions: Cbam-ResPRE

Here are the items that describes the Cbam-ResPRE network in the configuration file.

- `name`: "CbamResPRE".
- `input_channel`: int, optional, default: 441, the dimension of input;
- `output_channel`: int, optional, default: 10, the dimension of output;
- `hidden_channel`: int, optional, default: 64, the dimension of hidden layer;
- `blocks`: int, optional, default: 22, the number of convolutional layers;
- `droprate`: float, optional, default: 0.2, the dropout rate in the residual block.

## Network Configuraions: SE-ResPRE

Here are the items that describes the SE-ResPRE network in the configuration file.

- `name`: "SEResPRE".
- `input_channel`: int, optional, default: 441, the dimension of input;
- `output_channel`: int, optional, default: 10, the dimension of output;
- `hidden_channel`: int, optional, default: 64, the dimension of hidden layer;
- `blocks`: int, optional, default: 22, the number of convolutional layers;
- `droprate`: float, optional, default: 0.2, the dropout rate in the residual block;
- `reduction`: int, optional, defaulr: 4, the reduction rate of the squeeze-and-excitation module.

## Network Configuraions: NL-ResPRE

Here are the items that describes the NL-ResPRE network in the configuration file.

- `name`: "NLResPRE".
- `input_channel`: int, optional, default: 441, the dimension of input;
- `output_channel`: int, optional, default: 10, the dimension of output;
- `hidden_channel`: int, optional, default: 64, the dimension of hidden layer;
- `blocks`: int, optional, default: 22, the number of convolutional layers;
- `droprate`: float, optional, default: 0.2, the dropout rate in the residual block.

## Network Configuraions: Halo-ResPRE

Here are the items that describes the Halo-ResPRE network in the configuration file.

- `name`: "HaloResPRE".
- `input_channel`: int, optional, default: 441, the dimension of input;
- `output_channel`: int, optional, default: 10, the dimension of output;
- `hidden_channel`: int, optional, default: 64, the dimension of hidden layer;
- `blocks`: int, optional, default: 22, the number of convolutional layers;
- `droprate`: float, optional, default: 0.2, the dropout rate in the residual block;
- `block_size`: int, optional, default: 8, the neighborhood block size (feature map must be divisible by this);
- `halo_size`: int, optional, default: 4, the halo size (block receptive field);
- `dim_head`: int, optional, default: 16, the dimension of each head;
- `heads`: int, optional, default: 4, the number of attention heads

## Network Configuration: FC-ResPRE

Here are the items that describes the FC-ResPRE network in the configuration file.

- `name`: "FCResPRE".
- `input_channel`: int, optional, default: 441, the dimension of input;
- `output_channel`: int, optional, default: 10, the dimension of output;
- `hidden_channel`: int, optional, default: 64, the dimension of hidden layer;
- `blocks`: int, optional, default: 22, the number of convolutional layers;
- `droprate`: float, optional, default: 0.2, the dropout rate in the residual block.

## Network Configuraions: Cbam-FC-ResPRE

Here are the items that describes the Cbam-FC-ResPRE network in the configuration file.

- `name`: "CbamFCResPRE".
- `input_channel`: int, optional, default: 441, the dimension of input;
- `output_channel`: int, optional, default: 10, the dimension of output;
- `hidden_channel`: int, optional, default: 64, the dimension of hidden layer;
- `blocks`: int, optional, default: 22, the number of convolutional layers;
- `droprate`: float, optional, default: 0.2, the dropout rate in the residual block.

## Network Configuraions: SE-FC-ResPRE

Here are the items that describes the SE-FC-ResPRE network in the configuration file.

- `name`: "SEFCResPRE".
- `input_channel`: int, optional, default: 441, the dimension of input;
- `output_channel`: int, optional, default: 10, the dimension of output;
- `hidden_channel`: int, optional, default: 64, the dimension of hidden layer;
- `blocks`: int, optional, default: 22, the number of convolutional layers;
- `droprate`: float, optional, default: 0.2, the dropout rate in the residual block;
- `reduction`: int, optional, defaulr: 4, the reduction rate of the squeeze-and-excitation module.

## Network Configurations: DilatedResnet34

Here are the items that describes the DilatedResnet34 network in the configuration file.

- `name`: "DilatedResnet34".
- `input_channel`: int, optional, default: 441, the dimension of input;
- `output_channel`: int, optional, default: 10, the dimension of output;
- `arch_config`: list of list, optional, default value is stated in `models/DilatedResnet34.py` file, the architecture of the dilated residual network. Each item of the list is also a list that contains exactly 5 elements `[a, b, c, d, e]`, where `a` is the number of the residual blocks, `b` is the input channel of the layer, `c` is the output channel of the layer, `d` is the kernel size of the layer and `e` is the dilation size of the layer.
