import os
import os.path as osp
import numpy as np
import torch
import random
import tarfile
import shutil
from torch.utils.data import Dataset


class ProteinDataset(Dataset):
    '''
    Protein dataset
    '''
    def __init__(self, feature_dir, label_dir):
        '''
        Construct protein dataset

        Parameters
        ----------
        feature_dir: the feature directory.
        label_dir: the label directory.
        '''
        super(ProteinDataset, self).__init__()
        self.label_dir = label_dir
        self.feature_dir = feature_dir
        self.proteins = os.listdir(self.label_dir)

    def __getitem__(self, index):
        prot_name = self.proteins[index]
        prot_name = prot_name[:prot_name.find('.')]
        dist = self.get_label(os.path.join(self.label_dir, prot_name + '.npy'))
        feature = self.get_feature(os.path.join(self.feature_dir, prot_name + '.npy.gz'))
        mask = np.where(dist == -1, 0, 1)
        label = np.zeros(dist.shape)
        label += np.where((dist >= 4) & (dist < 6), np.ones_like(label), np.zeros_like(label))
        label += np.where((dist >= 6) & (dist < 8), np.ones_like(label) * 2, np.zeros_like(label))
        label += np.where((dist >= 8) & (dist < 10), np.ones_like(label) * 3, np.zeros_like(label))
        label += np.where((dist >= 10) & (dist < 12), np.ones_like(label) * 4, np.zeros_like(label))
        label += np.where((dist >= 12) & (dist < 14), np.ones_like(label) * 5, np.zeros_like(label))
        label += np.where((dist >= 14) & (dist < 16), np.ones_like(label) * 6, np.zeros_like(label))
        label += np.where((dist >= 16) & (dist < 18), np.ones_like(label) * 7, np.zeros_like(label))
        label += np.where((dist >= 18) & (dist < 20), np.ones_like(label) * 8, np.zeros_like(label))
        label += np.where((dist >= 20), np.ones_like(label) * 9, np.zeros_like(label))
        return feature.astype(np.float32), label.astype(np.int), mask.astype(np.bool)

    def __len__(self):
        return len(self.proteins)

    def get_label(self, name):
        tmp_label = np.load(name)
        return tmp_label

    @staticmethod
    def get_feature(name):
        f_name = name.replace(".npy.gz", "")
        g_file = tarfile.open(name)
        g_file.extractall(f_name)
        dir_ = os.listdir(f_name)
        tmp_feature = np.load(os.path.join(f_name, dir_[0]))
        tmp_feature = np.transpose(tmp_feature, (2, 0, 1))
        shutil.rmtree(f_name)
        return tmp_feature


def collate_fn(data):
    assert len(data) > 0
    max_m = np.max([x.shape[1] for (x, _, _) in data])
    batch_size = len(data)
    channel_size = data[0][0].shape[0]

    features = np.zeros((batch_size, channel_size, max_m, max_m)).astype(np.float32)
    labels = np.zeros((batch_size, max_m, max_m)).astype(np.int)
    masks = np.zeros((batch_size, max_m, max_m)).astype(np.bool)
    for i, piece_data in enumerate(data):
        feature, label, mask = piece_data
        m = feature.shape[1]
        features[i, :, 0:m, 0:m] = feature
        labels[i, 0:m, 0:m] = label
        masks[i, 0:m, 0:m] = mask
    
    return torch.from_numpy(features), torch.from_numpy(labels), torch.from_numpy(masks)