import torch
import numpy as np
from sklearn.metrics import confusion_matrix


def calculate_acc(predict, labels):
    pred_labels = torch.max(predict, dim=1)[1].int()
    return torch.sum(pred_labels == labels.int()) * 100 / predict.shape[0]


def calculate_iou_single_shape(predict, labels, n_parts):
    pred_labels = torch.max(predict, dim=1)[1]
    Confs = confusion_matrix(labels.detach().cpu().numpy(), pred_labels.detach().cpu().numpy(), np.arange(n_parts))

    # Objects IoU
    IoUs = IoU_from_confusions(Confs)
    return IoUs


def calculate_iou(predict, labels, stack_lengths, n_parts):
    start_ind = 0
    iou_list = []
    for length in stack_lengths:
        iou = calculate_iou_single_shape(predict[start_ind:start_ind + length], labels[start_ind:start_ind + length], n_parts)
        iou_list.append(iou)
        start_ind += length
    iou_list = np.array(iou_list)
    return np.array(iou_list).mean(axis=0) * 100


def IoU_from_confusions(confusions):
    """
    Computes IoU from confusion matrices.
    :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
    the last axes. n_c = number of classes
    :param ignore_unclassified: (bool). True if the the first class should be ignored in the results
    :return: ([..., n_c] np.float32) IoU score
    """

    # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
    # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
    TP = np.diagonal(confusions, axis1=-2, axis2=-1)
    TP_plus_FN = np.sum(confusions, axis=-1)
    TP_plus_FP = np.sum(confusions, axis=-2)

    # Compute IoU
    IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

    # Compute mIoU with only the actual classes
    mask = TP_plus_FN < 1e-3
    counts = np.sum(1 - mask, axis=-1, keepdims=True)
    mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

    # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
    IoU += mask * mIoU

    return IoU
