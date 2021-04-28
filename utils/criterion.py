import numpy as np
from sklearn.metrics import precision_score


def cal_acc(label, mask, large_20, pred):
    result = np.zeros([4])

    mask_tmp = np.triu(mask, 0)

    L = label.shape[0]
    effect = np.where(mask_tmp.reshape(-1) != 0)
    label = label.reshape(-1)
    label = label[effect]
    pred = pred.reshape(-1)
    pred = pred[effect]
    large_20 = large_20.reshape(-1)
    large_20 = large_20[effect]
    order = np.argsort(large_20)[::-1]

    topL_10 = int(np.ceil(L/10))
    result[0] = precision_score(label[order[:topL_10]], pred[order[:topL_10]], average='weighted')
    topL_5 = int(np.ceil(L/5))
    result[1] = precision_score(label[order[:topL_5]], pred[order[:topL_5]], average='weighted')
    topL_2 = int(np.ceil(L/2))
    result[2] = precision_score(label[order[:topL_2]], pred[order[:topL_2]], average='weighted')
    topL = int(np.ceil(L))
    result[3] = precision_score(label[order[:topL]], pred[order[:topL]], average='weighted')

    return result


def cal_top(label, mask, pred):  
    acc = np.zeros([2, 4])
    large_20 = pred[9, :, :]
    large_20 = 1 - large_20
    pred_final = np.argmax(pred, 0)
    trunc_mat = np.zeros(label.shape)
    nn = label.shape[0]
    for kk in range(7):
        if kk != 0:
            trunc_mat = trunc_mat + np.diag(np.ones(nn - kk), kk) + np.diag(np.ones(nn - kk), -kk)
        else:
            trunc_mat = trunc_mat + np.diag(np.ones(nn - kk), kk)

    trunc_mat_tmp = np.ones(trunc_mat.shape) - trunc_mat
    mask = mask * trunc_mat_tmp
    acc[0, :] = cal_acc(label, mask, large_20, pred_final)

    trunc_mat = np.zeros(label.shape)
    nn = label.shape[0]
    for kk in range(25):
        if kk != 0:
            trunc_mat = trunc_mat + np.diag(np.ones(nn - kk), kk) + np.diag(np.ones(nn - kk), -kk)
        else:
            trunc_mat = trunc_mat + np.diag(np.ones(nn - kk), kk)

    trunc_mat_tmp = np.ones(trunc_mat.shape) - trunc_mat
    mask = mask * trunc_mat_tmp
    acc[1, :] = cal_acc(label, mask, large_20, pred_final)
    return acc


def calc_batch_acc(label, mask, pred):
    acc_batch = np.zeros((2, 4))
    batch_size = label.shape[0]
    for i in range(batch_size):
        acc_sample = cal_top(label[i], mask[i], pred[i])
        acc_batch += acc_sample
    acc_batch = acc_batch / batch_size
    return acc_batch, batch_size
