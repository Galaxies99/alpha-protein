import torch
import torch.nn as nn

class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, res, gt, mask):
        loss = self.cross_entropy(res, gt)
        loss = loss * mask
        sample_loss = loss.sum(dim = [1, 2]) / mask.sum(dim = [1, 2])
        mean_batch_loss = torch.mean(sample_loss)
        return mean_batch_loss
