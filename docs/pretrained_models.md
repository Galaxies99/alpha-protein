# Pretrained Models

Pretrained models are already released at [Google Drive](https://drive.google.com/drive/folders/10n0MPoP6MEXmNliQpJkCdKJTShV5YCOv?usp=sharing) or [Baidu Netdisk](https://pan.baidu.com/s/1_Wh3ORfXH5faYkcEXx4_Ng) (Extract Code: jl6n). After your download the pretrained models, put it in the `checkpoint` folder under the main directory, then you can test them or continue fine-tuning them.

The checkpoint folder should have the following structure, where `checkpoint_xxx.tar` are the pretrained models that you have downloaded.

```
checkpoint
├── checkpoint_DeepCov.tar
├── checkpoint_ResPRE.tar
└── ...
```

Notice that we provide the pretrained models of all models that we have implemented except NL-ResPRE, since it is too large for our GPUs to train.

**Note**. If you want to train your own model from begining, please keep `checkpoint` folder clean (at least it does not contain `checkpoint_[Your Network Name].tar`, where `[Your Network Name]` is the name of the network that you want to train), otherwise the training scripts will automatically load the checkpoints in the `checkpoint` and continue training. If you want to fine-tune the model, you can put the checkpoints in the `checkpoint` folder and then continue fine-tuning it.
