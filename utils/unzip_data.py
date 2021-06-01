import os
import sys
import shutil
import tarfile
import logging
from tqdm import tqdm
import numpy as np
sys.path.append(os.path.dirname(sys.path[0]))
from utils.logger import ColoredLogger


logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)


def rm_all_extracted_files(path):
    file_list = os.listdir(path)
    for file in file_list:
        file = os.path.abspath(os.path.join(path, file))
        if os.path.isfile(file) is False:
            shutil.rmtree(file)
            logger.info('Remove {}'.format(file))


def unzip_all_files(path):
    file_list = os.listdir(path)
    for file in tqdm(file_list):
        name = os.path.join(path, file)
        if name.find(".npy.gz") == -1:
            continue
        f_name = name.replace(".npy.gz", "")
        g_file = tarfile.open(name)
        g_file.extractall(f_name)
        dir_ = os.listdir(f_name)
        tmp_feature = np.load(os.path.join(f_name, dir_[0]))
        tmp_feature = np.transpose(tmp_feature, (2, 0, 1))
        shutil.rmtree(f_name)
        np.save(f_name + ".npy", tmp_feature)


if __name__ == '__main__':
    rm_all_extracted_files(os.path.join('data', 'train', 'feature'))
    rm_all_extracted_files(os.path.join('data', 'val', 'feature'))
    rm_all_extracted_files(os.path.join('data', 'test', 'feature'))
    unzip_all_files(os.path.join('data', 'train', 'feature'))
    unzip_all_files(os.path.join('data', 'val', 'feature'))
    unzip_all_files(os.path.join('data', 'test', 'feature'))