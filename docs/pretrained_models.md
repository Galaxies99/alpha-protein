# Pretrained Models

Pretrained models are already released at [?](?). 
After your download the pretrained models, put it in the main folder, then execute the following commands to unzip it.

```bash
unzip checkpoint.zip
```

The `checkpoint` folder should have the following structure.

```
checkpoint
├── checkpoint_DeepCov.tar
├── checkpoint_ResPRE.tar
├── checkpoint_CbamResPRE.tar
├── checkpoint_SEResPRE.tar
└── checkpoint_HaloResPRE.tar

```

where the `checkpoint_*.tar` is the pretrained models' checkpoints. 

**Note**. If you want to train your own model from begining, please keep `checkpoint` folder clean, otherwise the training scripts will automatically load the checkpoints in the `checkpoint` and continue training. If you want to fine-tune the model, you can put the checkpoints in the `checkpoint` folder and then continue fine-tuning it.