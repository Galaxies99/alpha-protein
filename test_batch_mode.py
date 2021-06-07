import os
import yaml
import torch
import random
import argparse
import logging
import warnings
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from time import perf_counter
from torch import optim
from torch.utils.data import DataLoader
from utils.logger import ColoredLogger
from utils.loss import MaskedCrossEntropyLoss
from utils.criterion import calc_batch_acc, calc_score
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

MAX_EPOCH = cfg_dict.get('max_epoch', 50)
BATCH_SIZE = cfg_dict.get('batch_size', 1)
ZIPPED = cfg_dict.get('zipped', True)
MULTIGPU = cfg_dict.get('multigpu', True)
CHECKPOINT_DIR = cfg_dict.get('checkpoint_dir', 'checkpoint')
TEMP_DIR = cfg_dict.get('temp_dir', 'temp')
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
TEST_DIR = os.path.join('data', 'test')
TEST_FEATURE_DIR = os.path.join(TEST_DIR, 'feature')
TEST_LABEL_DIR = os.path.join(TEST_DIR, 'label')
test_dataset = ProteinDataset(TEST_FEATURE_DIR, TEST_LABEL_DIR, TEMP_PATH, ZIPPED)
test_dataloader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False, collate_fn = collator, num_workers = 16)

def test(epoch_id):
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

    # Define Criterion
    criterion = MaskedCrossEntropyLoss()

    # Read model from checkpoints
    checkpoint_network_name = NETWORK_NAME if NETWORK_NAME != 'FCResPRE' else 'ResPRE'
    checkpoint_file = os.path.join(CHECKPOINT_DIR, NETWORK_NAME, 'checkpoint_{}_{}.tar'.format(checkpoint_network_name, epoch_id))
    # logger.info(f'Loading checkpoint: {checkpoint_file}')
    if os.path.isfile(checkpoint_file):
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        logger.info("Checkpoint {} (epoch {}) loaded.".format(checkpoint_file, start_epoch))
    else:
        raise AttributeError('No checkpoint file!')

    if MULTIGPU is True:
        model = torch.nn.DataParallel(model)

    def test_one_epoch():
        logger.info('Start testing process ...')
        model.eval()
        mean_loss = 0
        count = 0
        acc = np.zeros((2, 4))
        tot_batch = len(test_dataloader)
        for idx, data in enumerate(test_dataloader):
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
            logger.info('Test Epoch {}, batch {}/{}, time: {:.4f}, loss: {:.12f}'.format(epoch_id, idx + 1, tot_batch, perf_counter() - start_time, loss.item()))
            acc += acc_batch * batch_size
            mean_loss += loss.item() * batch_size
            count += batch_size

        mean_loss = mean_loss / count
        acc = acc / count
        score = calc_score(acc)
        logger.info('Finish testing process. Now calculating metrics ...')
        logger.info('Epoch: {}, Mean evaluation loss: {:.12f}'.format(epoch_id, mean_loss))
        logger.info('Epoch: {}, Mean acc: {}'.format(epoch_id, acc))
        logger.info('Epoch: {}, Score: {:.6f}'.format(epoch_id, score))
        return mean_loss, acc, score

    mean_loss, acc, score = test_one_epoch()
    acc = acc.reshape(-1).tolist()

    return mean_loss, acc, score


if __name__ == '__main__':
    losses, accs, scores = [], [], []
    for i in range(MAX_EPOCH):
        loss, acc, score = test(i)
        losses.append(loss)
        accs.append(acc)
        scores.append(score)
        logger.info(f"* Epoch: {i}, loss: {loss:.6f}, acc: {acc}, score: {score:.6f}")
    
    import pandas as pd
    output = [[losses[i], accs[i], scores[i]] for i in range(MAX_EPOCH)]
    df = pd.DataFrame(output)
    df.to_csv(f'test_{NETWORK_NAME}.csv', index=True, index_label='epoch', header=['Losses', 'Accs', 'Scores'])
