![](imgs/alpha-protein.png)

# Alpha-Protein: Protein Contact-map Prediction Networks Implementations in PyTorch

This is the official repository of alpha-protein, protein contact-map networks implementations in PyTorch framework. In the repository, we implement the following algorithms: DeepCov[1], ResPRE[2], our proposed NLResPRE (non-local ResPRE) based on CBAM[3], HaloNet based on Halo attention module[4] and DilatedResnet34[5].

## Requirements

```bash
pip install -r requirements.txt
```

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

1. Jones, David T., and Shaun M. Kandathil. "High precision in protein contact prediction using fully convolutional neural networks and minimal sequence features." Bioinformatics 34.19 (2018): 3308-3315.

2. Li, Yang, et al. "ResPRE: high-accuracy protein contact prediction by coupling precision matrix with deep residual neural networks." Bioinformatics 35.22 (2019): 4647-4655.

3. Vaswani, Ashish, et al. "Scaling local self-attention for parameter efficient visual backbones." arXiv preprint arXiv:2103.12731 (2021).

4. Jiayan Xu. "Protein Contact Map Prediction Using Deep Convolutional Neural Network." master thesis of Shanghai Jiao Tong University (2019).

5. Woo, Sanghyun, et al. "Cbam: Convolutional block attention module." Proceedings of the European conference on computer vision (ECCV). 2018.

## Citation

If you found our work useful, please cite the following items.

```bibtex
@misc{fang2021alphaprotein,
  author =       {Hongjie Fang, Zhanda Zhu, Peishen Yan and Hao Yin},
  title =        {Alpha Protein: Protein Contact-map Prediction Networks Implementations in PyTorch},
  howpublished = {\url{https://github.com/Galaxies99/alpha-protein}},
  year =         {2021}
}
```
