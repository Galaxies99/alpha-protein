![](imgs/alpha-protein.png)

# Alpha-Protein: Protein Contact-map Prediction Boosted by Attention

This is the official repository of "Alpha-Protein: Protein Contact-map Prediction Boosted by Attention". In the repository, we implement the following classic models:

- DeepCov [1];
- ResPRE [2];
  
as well as our proposed models

- CbamResPRE based on Cbam attention mechanism [3];
- SEResPRE based on squeeze-and-excitation attention mechanism [4];
- HaloResPRE based on Halo block attention mechanism [5];
- NLResPRE based on non-local attention mechanism [6].

We also provides models for ablation studies, namely FCResPRE, CbamFCResPRE and SEFCResPRE, which are based on ResPRE, CbamResPRE and SEResPRE respectively except only changing the kernel size of the final convolution block to 1x1 according to the idea of FCN [7]. A dilated residual network named DilatedResnet34 [8] is also provided for ablation studies in order to see whether the dilation process may improve the performances.

## Requirements

Execute the following commands to install the required packages.

```bash
pip install -r requirements.txt
```

## Data Preprocessing

Download the features and the labels into the `data` folder, then execute the following commands.

```bash
python utils/preprocessing.py
```

## Models

This repository provides totally 10 models, namely

- DeepCov model;
- ResPRE model;
- Cbam-ResPRE model;
- SE-ResPRE model;
- Halo-ResPRE model;
- NL-ResPRE model;
- FC-ResPRE model used for ablation studies;
- Cbam-FC-ResPRE model used for ablation studies;
- SE-FC-ResPRE model used for ablation studies;
- DilatedResnet34 model used for ablation studies.

Please see [docs/models.md](docs/models.md) for details.

## Configurations

Before training, evaluation and inference of the models, please set up your own configurations correctly. Please see [docs/configurations.md](docs/configurations.md) for details.


## Training

Execute the following command to begin training.

```bash
python train.py --cfg [Config File] 
                (--clean_cache)
```

where `[Config File]` is `config/default.yaml` by default, which points to a sample network (SampleNet). The optional `--clean_cache` will automatically clean the caches after every epochs to save the GPU memory.

## Testing

Execute the following command to test the saved model.

```bash
python test.py --cfg [Config File] 
               (--clean_cache)
```

where `[Config File]` is `config/default.yaml` by default, which points to a sample network (SampleNet). The optional `--clean_cache` will automatically clean the caches after every epochs to save the GPU memory.

If you want to test all checkpoints (the training results after each epoch) of a model, please execute the following command.

```bash
python full_test.py --cfg [Config File] 
                    --path [File Saving Path] 
                    (--clean_cache)
```

which will automatically test all the checkpoints and save the result into a csv file in `[File Saving Path]` named `test_[Network Name].csv`, which includes the accuracy and scores of every checkpoint of the model. 

## Citation

If you found our work useful, please cite the following items.

```bibtex
@misc{fang2021alphaprotein,
  author =       {Hongjie Fang, Zhanda Zhu, Peishen Yan and Hao Yin},
  title =        {Alpha Protein: Protein Contact-map Prediction Boosted by Attention},
  howpublished = {\url{https://github.com/Galaxies99/alpha-protein}},
  year =         {2021}
}
```

## References

1. Jones, David T., and Shaun M. Kandathil. "High precision in protein contact prediction using fully convolutional neural networks and minimal sequence features." Bioinformatics 34.19 (2018): 3308-3315.

2. Li, Yang, et al. "ResPRE: high-accuracy protein contact prediction by coupling precision matrix with deep residual neural networks." Bioinformatics 35.22 (2019): 4647-4655.

3. Woo, Sanghyun, et al. "Cbam: Convolutional block attention module." Proceedings of the European conference on computer vision (ECCV). 2018.

4. Hu, Jie, Li Shen, and Gang Sun. "Squeeze-and-excitation networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.
  
5. Vaswani, Ashish, et al. "Scaling local self-attention for parameter efficient visual backbones." arXiv preprint arXiv:2103.12731 (2021).

6. Wang, Xiaolong, et al. "Non-local neural networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.

7. Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks for semantic segmentation." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.

8. Jiayan Xu. "Protein Contact Map Prediction Using Deep Convolutional Neural Network." master thesis of Shanghai Jiao Tong University (2019).
