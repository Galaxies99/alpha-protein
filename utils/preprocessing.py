import os
import json
import random
import shutil
from tqdm import tqdm


def divideDataset(source, target, reference = None):
    '''
    Divide the dataset into training set, validating set and testing set.
    The ratio among them is approximately 8 : 1 : 1.

    Parameters
    ----------
    source: the source directory of data, must contain two folders named 'feature' and 'label'
    target: the target directory of data
    reference: str, optional, default: None, the referencing file list
    '''
    FEATURE_DIR = os.path.join(source, 'feature')
    LABEL_DIR = os.path.join(source, 'label')
    TRAIN_DIR = os.path.join(target, 'train')
    VAL_DIR = os.path.join(target, 'val')
    TEST_DIR = os.path.join(target, 'test')
    TRAIN_FEATURE_DIR = os.path.join(TRAIN_DIR, 'feature')
    TRAIN_LABEL_DIR = os.path.join(TRAIN_DIR, 'label')
    VAL_FEATURE_DIR = os.path.join(VAL_DIR, 'feature')
    VAL_LABEL_DIR = os.path.join(VAL_DIR, 'label')
    TEST_FEATURE_DIR = os.path.join(TEST_DIR, 'feature')
    TEST_LABEL_DIR = os.path.join(TEST_DIR, 'label')

    if not os.path.exists(target):
        os.mkdir(target)
    if not os.path.exists(TRAIN_DIR):
        os.mkdir(TRAIN_DIR)
    if not os.path.exists(VAL_DIR):
        os.mkdir(VAL_DIR)
    if not os.path.exists(TEST_DIR):
        os.mkdir(TEST_DIR)
    if not os.path.exists(TRAIN_FEATURE_DIR):
        os.mkdir(TRAIN_FEATURE_DIR)
    if not os.path.exists(TRAIN_LABEL_DIR):
        os.mkdir(TRAIN_LABEL_DIR)
    if not os.path.exists(VAL_FEATURE_DIR):
        os.mkdir(VAL_FEATURE_DIR)
    if not os.path.exists(VAL_LABEL_DIR):
        os.mkdir(VAL_LABEL_DIR)
    if not os.path.exists(TEST_FEATURE_DIR):
        os.mkdir(TEST_FEATURE_DIR)
    if not os.path.exists(TEST_LABEL_DIR):
        os.mkdir(TEST_LABEL_DIR)

    label_list = os.listdir(LABEL_DIR)
    ALL_SAMPLES = len(label_list)
    TRAINING_SAMPLES = int(0.8 * ALL_SAMPLES)
    VALIDATING_SAMPLES = int(0.1 * ALL_SAMPLES)
    TESTING_SAMPLES = ALL_SAMPLES - TRAINING_SAMPLES - VALIDATING_SAMPLES

    if reference is None:
        random.shuffle(label_list)
    else:
        try:
            with open(reference, 'r') as f:
                label_list = json.load(f)['order']
        except Exception:
            raise AttributeError('Reference file invalid!')
        
    for i, label_filename in tqdm(enumerate(label_list)):
        if i < TRAINING_SAMPLES:
            target_folder = TRAIN_DIR
        elif i < TRAINING_SAMPLES + VALIDATING_SAMPLES:
            target_folder = VAL_DIR
        else:
            target_folder = TEST_DIR
        id, _ = os.path.splitext(label_filename)
        target_feature = os.path.join(target_folder, 'feature')
        target_label = os.path.join(target_folder, 'label')
        feature_filename = id + '.npy.gz'
        shutil.copy(os.path.join(LABEL_DIR, label_filename), target_label)
        shutil.copy(os.path.join(FEATURE_DIR, feature_filename), target_feature)
    
    if reference is None:
        out_dict = {'order': label_list}
        with open(os.path.join(target, 'reference.json'), 'w') as f:
            json.dump(out_dict, f)
    

if __name__ == '__main__':
    divideDataset('data', 'data', os.path.join('data', 'reference.json'))