# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

import numpy as np
import sklearn
import os
import tensorflow as tf


def setup_system(gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for sel_gpu in gpus:
                tf.config.experimental.set_memory_growth(sel_gpu, True)
        except RuntimeError as e:
            print(e)


def get_class_weigths(np_train):
    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    pos = np_train[1][:, 1].sum()
    neg = np_train[1][:, 0].sum()
    total = pos + neg
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))
    return class_weight


def evaluate_model_simple(pred, truth, is_one_hot=True, return_dict=True):
    """
    Evaluate the prediction results of a model with 7 different metrics
    Metrics:
        Confusion Matrix
        F1 Macro
        F1 Micro
        F1 Weighted
        Precision
        Recall
        Kappa (sklearn.metrics.cohen_kappa_score)

    Parameters:
        pred
            predictions made by the model

        truth
            the ground-truth labels

        is_one_hot=True
            whether the predictions and ground-truth labels are one-hot encoded or not

        return_dict=True
            whether to return the results in dictionary form (return a tuple if False)

    Return:
        results
            dictionary with 7 entries if return_dict=True
            tuple of size 7 if return_dict=False
    """

    if is_one_hot:
        truth_argmax = np.argmax(truth, axis=1)
        pred_argmax = np.argmax(pred, axis=1)
    else:
        truth_argmax = truth
        pred_argmax = pred

    test_cm = sklearn.metrics.confusion_matrix(truth_argmax, pred_argmax)
    test_auroc = sklearn.metrics.roc_auc_score(truth, pred, average="micro", multi_class='ovr')
    test_auprc_micro = sklearn.metrics.average_precision_score(truth, pred, average="micro")
    test_auprc_macro = sklearn.metrics.average_precision_score(truth, pred, average="macro")
    test_acc = sklearn.metrics.accuracy_score(truth_argmax, pred_argmax)
    test_f1 = sklearn.metrics.f1_score(truth_argmax, pred_argmax, average='macro')
    test_precision = sklearn.metrics.precision_score(truth_argmax, pred_argmax, average='macro')
    test_recall = sklearn.metrics.recall_score(truth_argmax, pred_argmax, average='macro')
    test_kappa = sklearn.metrics.cohen_kappa_score(truth_argmax, pred_argmax)
    test_balanced_acc = sklearn.metrics.balanced_accuracy_score(truth_argmax, pred_argmax)

    test_f1_micro = sklearn.metrics.f1_score(truth_argmax, pred_argmax, average='micro')
    test_f1_weighted = sklearn.metrics.f1_score(truth_argmax, pred_argmax, average='weighted')

    if return_dict:
        return {
            'Accuracy': test_acc,
            'Balanced Accuracy': test_balanced_acc,
            'AUROC': test_auroc,
            'AUPRC Macro': test_auprc_macro,
            'AUPRC Micro': test_auprc_micro,
            'Confusion Matrix': test_cm,
            'F1 Macro': test_f1,
            'F1 Micro': test_f1_micro,
            'F1 Weighted': test_f1_weighted,
            'Precision': test_precision,
            'Recall': test_recall,
            'Kappa': test_kappa
        }
    else:
        return (test_acc, test_auroc, test_cm, test_f1, test_f1_micro, test_f1_weighted, test_precision, test_recall,
                test_kappa)