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
    def __init__(self, feature_dir, label_dir, temp_dir, zipped):
        '''
        Construct protein dataset
        Parameters
        ----------
        feature_dir: the feature directory;
        label_dir: the label directory;
        zipped: whether the data is zipped.
        '''
        super(ProteinDataset, self).__init__()
        self.label_dir = label_dir
        self.feature_dir = feature_dir
        self.temp_dir = temp_dir
        if os.path.exists(self.temp_dir) == False:
            os.makedirs(self.temp_dir)
        self.proteins = os.listdir(self.label_dir)
        self.zipped = zipped

    def __getitem__(self, index):
        prot_name = self.proteins[index]
        prot_name = prot_name[:prot_name.find('.')]
        dist = self.get_label(self.label_dir, prot_name)
        feature = self.get_feature(self.feature_dir, prot_name, self.zipped)
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
        return torch.from_numpy(feature.astype(np.float)), \
               torch.from_numpy(label.astype(np.int)), \
               torch.from_numpy(mask.astype(np.bool))

    def __len__(self):
        return len(self.proteins)

    def get_label(self, dir, name):
        tmp_label = np.load(os.path.join(dir, name + ".npy"))
        return tmp_label

    def get_feature(self, dir, name, zipped = False):
        if zipped:
            name = os.path.join(dir, name)
            g_file = tarfile.open(name + ".npy.gz")
            extract_dir = os.path.join(self.temp_dir, name)
            g_file.extractall(extract_dir)
            file = os.listdir(extract_dir)
            tmp_feature = np.load(os.path.join(extract_dir, file[0]))
            tmp_feature = np.transpose(tmp_feature, (2, 0, 1))
            shutil.rmtree(extract_dir)
        else:
            name = os.path.join(dir, name + ".npy")
            tmp_feature = np.load(name)
        return tmp_feature


class ProteinCollator(object):
    def __init__(self, block_size = 1):
        super(ProteinCollator, self).__init__()
        self.block_size = block_size
    
    def __call__(self, data):
        assert len(data) > 0
        max_m = np.max([x.shape[1] for (x, _, _) in data])
        max_m = int(np.ceil(max_m / self.block_size) * self.block_size)
        batch_size = len(data)
        channel_size = data[0][0].shape[0]

        features = torch.zeros((batch_size, channel_size, max_m, max_m), dtype = torch.float32)
        labels = torch.zeros((batch_size, max_m, max_m), dtype = torch.int64)
        masks = torch.zeros((batch_size, max_m, max_m), dtype = torch.bool)
        for i, piece_data in enumerate(data):
            feature, label, mask = piece_data
            m = feature.shape[1]
            features[i, :, 0:m, 0:m] = feature
            labels[i, 0:m, 0:m] = label
            masks[i, 0:m, 0:m] = mask
        
        return features, labels, masks


class ProteinInferenceDataset(Dataset):
    '''
    Protein dataset
    '''
    def __init__(self, feature_dir, temp_dir, zipped):
        '''
        Construct protein dataset
        Parameters
        ----------
        feature_dir: the feature directory;
        label_dir: the label directory;
        zipped: whether the data is zipped.
        '''
        super(ProteinInferenceDataset, self).__init__()
        self.feature_dir = feature_dir
        self.temp_dir = temp_dir
        if os.path.exists(self.temp_dir) == False:
            os.makedirs(self.temp_dir)
        self.proteins = os.listdir(self.feature_dir)
        self.zipped = zipped

    def __getitem__(self, index):
        prot_name = self.proteins[index]
        prot_name = prot_name[:prot_name.find('.')]
        feature = self.get_feature(self.feature_dir, prot_name, self.zipped)
        return prot_name, torch.FloatTensor(feature)

    def __len__(self):
        return len(self.proteins)

    def get_feature(self, dir, name, zipped = False):
        if zipped:
            name = os.path.join(dir, name)
            g_file = tarfile.open(name + ".npy.gz")
            extract_dir = os.path.join(self.temp_dir, name)
            g_file.extractall(extract_dir)
            file = os.listdir(extract_dir)
            tmp_feature = np.load(os.path.join(extract_dir, file[0]))
            tmp_feature = np.transpose(tmp_feature, (2, 0, 1))
            shutil.rmtree(extract_dir)
        else:
            name = os.path.join(dir, name + ".npy")
            tmp_feature = np.load(name)
        return tmp_feature