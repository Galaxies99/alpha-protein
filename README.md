# Alpha-Protein

## Requirements

- torch
- numpy

Version to be specified.

## Data Preprocessing

Download the features and the labels into the `data` folder, then execute the following commands.

```bash
python utils/preprocessing.py
```

## Models

* Model 1: DeepCov model for protein contact map prediction using fully convolutional neural networks with precision matrix as input. (Baseline)
* Model 2: ResPRE model for protein contact map prediction using deep residual neural networks with precision matrix as input. (Baseline)
* Model 3: NLResPRE model for protein contact map prediction using improved ResPRE model with precision matrix as input. (Ours)

Please see [docs/models.md](docs/models.md) for details.

## Configurations

Before training, evaluation and inference of the models, please set up your own configurations correctly. Please see [docs/configurations.md](docs/configurations.md) for details.


## Training

```bash
python train.py --cfg [Config File]
```

`[Config File]` is `config/default.yaml` by default.

## Testing

```bash
python test.py --cfg [Config File]
```

`[Config File]` is `config/default.yaml` by default.
