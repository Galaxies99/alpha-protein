import os
import yaml
import torch
import random
import argparse
import numpy as np
from torch import optim
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils.loss import MaskedCrossEntropyLoss
from torch.optim.lr_scheduler import MultiStepLR
from dataset import ProteinDataset, collate_fn
from models.SampleNet import SampleNet
from models.DeepCov import DeepCov
from models.ResPRE import ResPRE
from models.NLResPRE import NLResPRE
from models.GANcon import GANcon

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default = 'configs/default.yaml', help = 'Config File', type = str)
FLAGS = parser.parse_args()
CFG_FILE = FLAGS.cfg

with open(CFG_FILE, 'r') as cfg_file:
    cfg_dict = yaml.load(cfg_file, Loader=yaml.FullLoader)
    
BATCH_SIZE = cfg_dict.get('batch_size', 4)
MULTIGPU = cfg_dict.get('multigpu', True)
MAX_EPOCH = cfg_dict.get('max_epoch', 30)
ADAM_BETA1 = cfg_dict.get('adam_beta1', 0.9)
ADAM_BETA2 = cfg_dict.get('adam_beta2', 0.999)
LEARNING_RATE = cfg_dict.get('learning_rate', 0.001)
MILESTONES = cfg_dict.get('milestones', [int(MAX_EPOCH / 2)])
GAMMA = cfg_dict.get('gamma', 0.1)
CHECKPOINT_DIR = cfg_dict.get('checkpoint_dir', 'checkpoint')
NETWORK = cfg_dict.get('network', {})
if "name" not in NETWORK.keys():
    NETWORK["name"] = "SampleNet"

# Load data & Build dataset
train_dataset = ProteinDataset('data/train/feature', 'data/train/label')
train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn = collate_fn)

val_dataset = ProteinDataset('data/val/feature', 'data/val/label')
val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn = collate_fn)

# Build model from configs
if NETWORK["name"] == "SampleNet":
    model = SampleNet(NETWORK)
elif NETWORK["name"] == "DeepCov":
    model = DeepCov(NETWORK)
elif NETWORK["name"] == "ResPRE":
    model = ResPRE(NETWORK)
elif NETWORK["name"] == "NLResPRE":
    model = NLResPRE(NETWORK)
elif NETWORK["name"] == "GANcon":
    model = GANcon(NETWORK)
else:
    raise AttributeError("Invalid Network.")


if MULTIGPU is False:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cpu'):
        raise EnvironmentError('No GPUs, cannot initialize multigpu training.')
    model.to(device)

# Define optimizer
optimizer = optim.Adam(model.parameters(), betas = (ADAM_BETA1, ADAM_BETA2), lr = LEARNING_RATE)

# Define Criterion
criterion = MaskedCrossEntropyLoss()

# Define Scheduler
lr_scheduler = MultiStepLR(optimizer, milestones = MILESTONES, gamma = GAMMA)

# Read checkpoints
start_epoch = 0
if os.path.exists(CHECKPOINT_DIR) == False:
    os.mkdir(CHECKPOINT_DIR)
checkpoint_file = CHECKPOINT_DIR + '/checkpoint.tar'
if os.path.isfile(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    lr_scheduler.load_state_dict(checkpoint['scheduler'])
    print("Load checkpoint {} (epoch {})".format(checkpoint_file, start_epoch))

if MULTIGPU is True:
    model = torch.nn.DataParallel(model)


def train_one_epoch():
    model.train()
    for idx, data in enumerate(train_dataloader):
        optimizer.zero_grad()
        feature, label, mask = data
        feature = feature.to(device)
        label = label.to(device)
        mask = mask.to(device)
        # Forward
        result = model(feature)
        # Backward
        loss = criterion(result, label, mask)
        loss.backward()
        optimizer.step()

        print('--------------- Train Batch %d ---------------' % (idx + 1))
        print('loss: %.12f' % loss.item())


def train(start_epoch):
    global cur_epoch
    for epoch in range(start_epoch, MAX_EPOCH):
        cur_epoch = epoch
        print('**************** Epoch %d ****************' % (epoch + 1))
        print('learning rate: %f' % (lr_scheduler.get_last_lr()[0]))
        train_one_epoch()
        loss = eval_one_epoch()
        lr_scheduler.step()
        if MULTIGPU is False:
            save_dict = {'epoch': epoch + 1, 'loss': loss,
                         'optimizer_state_dict': optimizer.state_dict(),
                         'model_state_dict': model.state_dict(),
                         'scheduler': lr_scheduler.state_dict()
                         }
        else:
            save_dict = {'epoch': epoch + 1, 'loss': loss,
                         'optimizer_state_dict': optimizer.state_dict(),
                         'model_state_dict': model.module.state_dict(),
                         'scheduler': lr_scheduler.state_dict()
                         }
        torch.save(save_dict, os.path.join(CHECKPOINT_DIR, 'checkpoint.tar'))
        torch.save(save_dict, os.path.join(CHECKPOINT_DIR, 'checkpoint' + str(epoch) + '.tar'))
        print('mean eval loss: %.12f' % loss)


def eval_one_epoch():
    model.eval()
    mean_loss = 0
    count = 0
    for idx, data in enumerate(val_dataloader):
        feature, label, mask = data
        feature = feature.to(device)
        label = label.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            result = model(feature)
        # Compute loss
        with torch.no_grad():
            loss = criterion(result, label, mask)
        print('--------------- Eval Batch %d ---------------' % (idx + 1))
        print('loss: %.12f' % loss.item())
        mean_loss += loss.item()
        count += 1

    mean_loss = mean_loss / count
    return mean_loss


if __name__ == '__main__':
    train(start_epoch)