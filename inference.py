import os
import yaml
import torch
import argparse
import logging
import warnings
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils.logger import ColoredLogger
from dataset import ProteinInferenceDataset
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

# Load data & Build dataset
INFERENCE_FEATURE_DIR = os.path.join('data', 'inference', 'feature')
inference_dataset = ProteinInferenceDataset(INFERENCE_FEATURE_DIR, TEMP_PATH, ZIPPED)

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

# Read model from checkpoints
checkpoint_file = os.path.join(CHECKPOINT_DIR, 'checkpoint_{}.tar'.format(NETWORK_NAME))
if os.path.isfile(checkpoint_file):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    logger.info("Checkpoint {} (epoch {}) loaded.".format(checkpoint_file, start_epoch))
else:
    raise AttributeError('No checkpoint file!')

if MULTIGPU is True:
    model = torch.nn.DataParallel(model)

inference_dir = 'inference_' + NETWORK_NAME
if os.path.exists(inference_dir) == False:
    os.makedirs(inference_dir)

def inference():
    logger.info('Start inference process ...')
    model.eval()
    total_samples = len(inference_dataset)
    for idx in tqdm(range(total_samples)):
        if CLEAN_CACHE and device != torch.device('cpu'):
            torch.cuda.empty_cache()
        name, feature = inference_dataset[idx]
        L = feature.shape[1]
        newL = int(np.ceil(L / BLOCK_SIZE) * BLOCK_SIZE)
        new_feature = torch.zeros((1, feature.shape[0], newL, newL), dtype = torch.float32)
        new_feature[0, :, 0:L, 0:L] = feature
        new_feature = new_feature.to(device)
        del feature
        with torch.no_grad():
            result = model(new_feature)
            result = F.softmax(result, dim = 1)
            result = result.detach().cpu().numpy()
            result = result[0, :, 0:L, 0:L]
            result = result.transpose(2, 0, 1)
            np.save(os.path.join(inference_dir, name + '.npy'), result)

    logger.info('Finish inference process.')


if __name__ == '__main__':
    inference()
