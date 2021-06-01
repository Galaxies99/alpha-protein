![](imgs/alpha-protein.png)

# Alpha-Protein: Protein Contact-map Prediction Networks Implementations in PyTorch

This is the official repository of alpha-protein, protein contact-map networks implementations in PyTorch framework. In the repository, we implement 3 main algorithm: DeepCov[1], ResPRE[2] and our proposed NLResPRE (non-local ResPRE).

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

## Reference

[1]  Jones, David T., and Shaun M. Kandathil. "High precision in protein contact prediction using fully convolutional neural networks and minimal sequence features." Bioinformatics 34.19 (2018): 3308-3315.
[2]  Li, Yang, et al. "ResPRE: high-accuracy protein contact prediction by coupling precision matrix with deep residual neural networks." Bioinformatics 35.22 (2019): 4647-4655.

## Citation

If you found our work useful, please cite the following items.

```bibtex
@article{jones2018high,
  title =        {High precision in protein contact prediction using fully convolutional neural networks and minimal sequence features},
  author =       {Jones, David T and Kandathil, Shaun M},
  journal =      {Bioinformatics},
  volume =       {34},
  number =       {19},
  pages =        {3308--3315},
  year =         {2018},
  publisher =    {Oxford University Press}
}

@article{li2019respre,
  title =        {ResPRE: high-accuracy protein contact prediction by coupling precision matrix with deep residual neural networks},
  author =       {Li, Yang and Hu, Jun and Zhang, Chengxin and Yu, Dong-Jun and Zhang, Yang},
  journal =      {Bioinformatics},
  volume =       {35},
  number =       {22},
  pages =        {4647--4655},
  year =         {2019},
  publisher =    {Oxford University Press}
}

@misc{fang2021alphaprotein,
  author =       {Hongjie Fang, Zhanda Zhu, Peishen Yan and Hao Yin},
  title =        {Alpha Protein: Protein Contact-map Prediction Networks Implementations in PyTorch},
  howpublished = {\url{https://github.com/Galaxies99/alpha-protein}},
  year =         {2021}
}
```
