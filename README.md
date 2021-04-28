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
