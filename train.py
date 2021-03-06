import os
import yaml
import torch
import warnings
import argparse
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from time import perf_counter
from torch import optim
from torch.optim import AdamW
from torch.utils.data import DataLoader
from utils.loss import MaskedCrossEntropyLoss
from utils.logger import ColoredLogger
from utils.criterion import calc_batch_acc, calc_score
from torch.optim.lr_scheduler import MultiStepLR
from dataset import ProteinDataset, ProteinCollator
from models.SampleNet import SampleNet
from models.DeepCov import DeepCov
from models.ResPRE import ResPRE
from models.FCResPRE import FCResPRE
from models.CbamResPRE import CbamResPRE
from models.CbamFCResPRE import CbamFCResPRE
from models.NLResPRE import NLResPRE
from models.SEResPRE import SEResPRE
from models.SEFCResPRE import SEFCResPRE
from models.HaloResPRE import HaloResPRE
from models.DilatedResnet34 import DilatedResnet34


logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--cfg', default = os.path.join('configs', 'default.yaml'), help = 'Config File', type = str)
parser.add_argument('--clean_cache', action = 'store_true', help = 'whether to clean the cache of GPU while training, evaluation and testing')
FLAGS = parser.parse_args()
CFG_FILE = FLAGS.cfg
CLEAN_CACHE = FLAGS.clean_cache

with open(CFG_FILE, 'r') as cfg_file:
    cfg_dict = yaml.load(cfg_file, Loader=yaml.FullLoader)
    
BATCH_SIZE = cfg_dict.get('batch_size', 4)
ZIPPED = cfg_dict.get('zipped', True)
MULTIGPU = cfg_dict.get('multigpu', True)
MAX_EPOCH = cfg_dict.get('max_epoch', 30)
ADAM_BETA1 = cfg_dict.get('adam_beta1', 0.9)
ADAM_BETA2 = cfg_dict.get('adam_beta2', 0.999)
ADAM_WEIGHT_DECAY = cfg_dict.get('adam_weight_decay', 0.01)
ADAM_EPS = cfg_dict.get('adam_eps', 1e-6)
LEARNING_RATE = cfg_dict.get('learning_rate', 0.001)
MILESTONES = cfg_dict.get('milestones', [int(MAX_EPOCH / 2)])
GAMMA = cfg_dict.get('gamma', 0.1)
CHECKPOINT_DIR = cfg_dict.get('checkpoint_dir', 'checkpoint')
DATA_DIR = cfg_dict.get('data_dir', 'data')
TEMP_DIR = cfg_dict.get('temp_dir', 'temp')
SOFTMAX = cfg_dict.get('softmax', True)
NETWORK = cfg_dict.get('network', {})
if "name" not in NETWORK.keys():
    NETWORK["name"] = "SampleNet"
NETWORK_NAME = NETWORK["name"]
TEMP_PATH = os.path.join(TEMP_DIR, NETWORK_NAME)
if os.path.exists(TEMP_PATH) is False:
    os.makedirs(TEMP_PATH)

if NETWORK_NAME == "HaloResPRE":
    BLOCK_SIZE = NETWORK.get('block_size', 8)
else:
    BLOCK_SIZE = 1
collator = ProteinCollator(block_size = BLOCK_SIZE)

# Load data & Build dataset
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TRAIN_FEATURE_DIR = os.path.join(TRAIN_DIR, 'feature')
TRAIN_LABEL_DIR = os.path.join(TRAIN_DIR, 'label')
train_dataset = ProteinDataset(TRAIN_FEATURE_DIR, TRAIN_LABEL_DIR, TEMP_PATH, ZIPPED)
train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn = collator, num_workers = 16)

VAL_DIR = os.path.join(DATA_DIR, 'val')
VAL_FEATURE_DIR = os.path.join(VAL_DIR, 'feature')
VAL_LABEL_DIR = os.path.join(VAL_DIR, 'label')
val_dataset = ProteinDataset(VAL_FEATURE_DIR, VAL_LABEL_DIR, TEMP_PATH, ZIPPED)
val_dataloader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True, collate_fn = collator, num_workers = 16)

# Build model from configs
if NETWORK_NAME == "SampleNet":
    model = SampleNet(NETWORK)
elif NETWORK_NAME == "DeepCov":
    model = DeepCov(NETWORK)
elif NETWORK_NAME == "ResPRE":
    model = ResPRE(NETWORK)
elif NETWORK_NAME == "FCResPRE":
    model = FCResPRE(NETWORK)
elif NETWORK_NAME == "NLResPRE":
    model = NLResPRE(NETWORK)
elif NETWORK_NAME == "CbamResPRE":
    model = CbamResPRE(NETWORK)
elif NETWORK_NAME == "CbamFCResPRE":
    model = CbamFCResPRE(NETWORK)
elif NETWORK_NAME == "SEResPRE":
    model = SEResPRE(NETWORK)
elif NETWORK_NAME == "SEFCResPRE":
    model = SEFCResPRE(NETWORK)
elif NETWORK_NAME == "HaloResPRE":
    model = HaloResPRE(NETWORK)
elif NETWORK_NAME == "DilatedResnet34":
    model = DilatedResnet34(NETWORK)
else:
    raise AttributeError("Invalid Network.")

# Data Parallelism
if MULTIGPU is False:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cpu'):
        raise EnvironmentError('No GPUs, cannot initialize multigpu training.')
    model.to(device)

# Define optimizer
optimizer = optim.AdamW(model.parameters(), betas = (ADAM_BETA1, ADAM_BETA2), lr = LEARNING_RATE, weight_decay = ADAM_WEIGHT_DECAY, eps = ADAM_EPS)

# Define Criterion
criterion = MaskedCrossEntropyLoss(with_softmax = SOFTMAX)

# Define Scheduler
lr_scheduler = MultiStepLR(optimizer, milestones = MILESTONES, gamma = GAMMA)

# Read checkpoints
start_epoch = 0
if os.path.exists(CHECKPOINT_DIR) == False:
    os.mkdir(CHECKPOINT_DIR)
checkpoint_file = os.path.join(CHECKPOINT_DIR, 'checkpoint_{}.tar'.format(NETWORK_NAME))
if os.path.isfile(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    lr_scheduler.last_epoch = start_epoch - 1
    logger.info("Checkpoint {} (epoch {}) loaded.".format(checkpoint_file, start_epoch))

if MULTIGPU is True:
    model = torch.nn.DataParallel(model)


def train_one_epoch(epoch):
    logger.info('Start training process in epoch {}.'.format(epoch + 1))
    model.train()
    tot_batch = len(train_dataloader)
    for idx, data in enumerate(train_dataloader):
        if CLEAN_CACHE and device != torch.device('cpu'):
            torch.cuda.empty_cache()
        start_time = perf_counter()
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
        logger.info('Train epoch {}/{} batch {}/{}, time: {:.4f}s, loss: {:.12f}'.format(epoch + 1, MAX_EPOCH, idx + 1, tot_batch, perf_counter() - start_time, loss.item()))
    logger.info('Finish training process in epoch {}.'.format(epoch + 1))


def eval_one_epoch(epoch):
    logger.info('Start evaluation process in epoch {}.'.format(epoch + 1))
    model.eval()
    mean_loss = 0
    count = 0
    acc = np.zeros((2, 4))
    tot_batch = len(val_dataloader)
    for idx, data in enumerate(val_dataloader):
        if CLEAN_CACHE and device != torch.device('cpu'):
            torch.cuda.empty_cache()
        start_time = perf_counter()
        feature, label, mask = data
        feature = feature.to(device)
        label = label.to(device)
        mask = mask.to(device)
        with torch.no_grad():
            result = model(feature)
        # Compute loss
        with torch.no_grad():
            loss = criterion(result, label, mask)
        result = F.softmax(result, dim = 1)
        acc_batch, batch_size = calc_batch_acc(label.cpu().detach().numpy(), mask.cpu().detach().numpy(), result.cpu().detach().numpy())
        logger.info('Eval epoch {}/{} batch {}/{}, time: {:.4f}s, loss: {:.12f}'.format(epoch + 1, MAX_EPOCH, idx + 1, tot_batch, perf_counter() - start_time, loss.item()))
        acc += acc_batch * batch_size
        mean_loss += loss.item() * batch_size
        count += batch_size
    mean_loss = mean_loss / count
    acc = acc / count
    score = calc_score(acc)
    logger.info('Finish evaluation process in epoch {}. Now calculating metrics ...'.format(epoch + 1))
    logger.info('Mean evaluation loss: {:.12f}'.format(mean_loss))
    logger.info('Mean acc: {}'.format(acc))
    logger.info('Score: {:.6f}'.format(score))
    return mean_loss, score


def train(start_epoch):
    global cur_epoch
    max_score = 0
    for epoch in range(start_epoch, MAX_EPOCH):
        cur_epoch = epoch
        logger.info('--> Epoch {}/{}, learning rate: {}'.format(epoch + 1, MAX_EPOCH, lr_scheduler.get_last_lr()[0]))
        train_one_epoch(epoch)
        loss, score = eval_one_epoch(epoch)
        lr_scheduler.step()
        if MULTIGPU is False:
            save_dict = {
                'epoch': epoch + 1, 
                'loss': loss,
                'optimizer_state_dict': optimizer.state_dict(),
                'model_state_dict': model.state_dict(),
                'scheduler': lr_scheduler.state_dict()
            }
        else:
            save_dict = {
                'epoch': epoch + 1, 
                'loss': loss,
                'optimizer_state_dict': optimizer.state_dict(),
                'model_state_dict': model.module.state_dict(),
                'scheduler': lr_scheduler.state_dict()
            }
        torch.save(save_dict, os.path.join(CHECKPOINT_DIR, 'checkpoint_{}_{}.tar'.format(NETWORK_NAME, epoch)))
        if score > max_score:
            max_score = score
            torch.save(save_dict, os.path.join(CHECKPOINT_DIR, 'checkpoint_{}.tar'.format(NETWORK_NAME)))


if __name__ == '__main__':
    train(start_epoch)