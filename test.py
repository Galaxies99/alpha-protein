import os
import yaml
import torch
import random
import argparse
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from utils.loss import MaskedCrossEntropyLoss
from utils.criterion import calc_batch_acc
from dataset import ProteinDataset, collate_fn
from models.SampleNet import SampleNet
from models.DeepCov import DeepCov
from models.ResPRE import ResPRE
from models.NLResPRE import NLResPRE

# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default = os.path.join('configs', 'default.yaml'), help = 'Config File', type = str)
FLAGS = parser.parse_args()
CFG_FILE = FLAGS.cfg

with open(CFG_FILE, 'r') as cfg_file:
    cfg_dict = yaml.load(cfg_file, Loader=yaml.FullLoader)
    
BATCH_SIZE = cfg_dict.get('batch_size', 4)
MULTIGPU = cfg_dict.get('multigpu', True)
CHECKPOINT_DIR = cfg_dict.get('checkpoint_dir', 'checkpoint')
NETWORK = cfg_dict.get('network', {})
if "name" not in NETWORK.keys():
    NETWORK["name"] = "SampleNet"
NETWORK_NAME = NETWORK["name"]

# Load data & Build dataset
TEST_DIR = os.path.join('data', 'test')
TEST_FEATURE_DIR = os.path.join(TEST_DIR, 'feature')
TEST_LABEL_DIR = os.path.join(TEST_DIR, 'label')
test_dataset = ProteinDataset(TEST_FEATURE_DIR, TEST_LABEL_DIR)
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

# Define Criterion
criterion = MaskedCrossEntropyLoss()

if MULTIGPU is True:
    model = torch.nn.DataParallel(model)

# Read model from checkpoints
checkpoint_file = os.path.join(CHECKPOINT_DIR, 'checkpoint_{}.tar'.format(NETWORK_NAME))
if os.path.isfile(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("Load checkpoint {} (epoch {})".format(checkpoint_file, start_epoch))
else:
    raise AttributeError('No checkpoint file!')


def test_one_epoch():
    model.eval()
    mean_loss = 0
    count = 0
    acc = np.zeros((2, 4))
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
        acc_batch, batch_size = calc_batch_acc(label.numpy(), mask.numpy(), result.numpy())
        print('--------------- Test Batch %d ---------------' % (idx + 1))
        print('loss: %.12f' % loss.item())
        print('acc: ', acc_batch)
        acc += acc_batch * batch_size
        mean_loss += loss.item() * batch_size
        count += batch_size

    mean_loss = mean_loss / count
    acc = acc / count
    return mean_loss, acc


if __name__ == '__main__':
    loss, acc = test_one_epoch()
    print('--------------- Test Result ---------------')
    print('test mean loss: %.12f' % loss)
    print('test mean acc: ', acc)