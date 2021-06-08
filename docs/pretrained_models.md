# Pretrained Models

Pretrained models are already released at [?](?). After your download the pretrained models, put it in the `checkpoint` folder under the main directory, then you can test them or continue fine-tuning them.

**Note**. If you want to train your own model from begining, please keep `checkpoint` folder clean (at least it does not contain `checkpoint_[Your Network Name].tar`, where `[Your Network Name]` is the name of the network that you want to train), otherwise the training scripts will automatically load the checkpoints in the `checkpoint` and continue training. If you want to fine-tune the model, you can put the checkpoints in the `checkpoint` folder and then continue fine-tuning it.
