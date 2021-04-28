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
NETWORK_NAME = NETWORK["name"]

# Load data & Build dataset
test_dataset = ProteinDataset('data/test/feature', 'data/test/label')
test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn = collate_fn)

# Build model from configs
if NETWORK_NAME == "SampleNet":
    model = SampleNet(NETWORK)
elif NETWORK_NAME == "DeepCov":
    model = DeepCov(NETWORK)
elif NETWORK_NAME == "ResPRE":
    model = ResPRE(NETWORK)
elif NETWORK_NAME == "NLResPRE":
    model = NLResPRE(NETWORK)
elif NETWORK_NAME == "GANcon":
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

if MULTIGPU is True:
    model = torch.nn.DataParallel(model)

# Read model from checkpoints
checkpoint_file = os.path.join(CHECKPOINT_DIR, 'checkpoint_{}.tar'.format(NETWORK_NAME))
if os.path.isfile(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    lr_scheduler.load_state_dict(checkpoint['scheduler'])
    print("Load checkpoint {} (epoch {})".format(checkpoint_file, start_epoch))
else:
    raise AttributeError('No checkpoint file!')


def test_one_epoch():
    model.eval()
    mean_loss = 0
    count = 0
    for idx, data in enumerate(test_dataloader):
        feature, label, mask = data
        feature = feature.to(device)
        label = label.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            result = model(feature)
        # Compute loss
        with torch.no_grad():
            loss = criterion(result, label, mask)
        print('--------------- Test Batch %d ---------------' % (idx + 1))
        print('loss: %.12f' % loss.item())
        mean_loss += loss.item()
        count += 1

    mean_loss = mean_loss / count
    return mean_loss


if __name__ == '__main__':
    result = test_one_epoch()
    print('--------------- Test Result ---------------')
    print('test mean loss: %.12f' % result)