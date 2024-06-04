# © 2024 Nokia
# Licensed under the BSD 3 Clause Clear License
# SPDX-License-Identifier: BSD-3-Clause-Clear

from aif360.datasets import StandardDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score, f1_score, precision_score, recall_score, \
    balanced_accuracy_score

# Evaluation with bootstrapping
np.random.seed(1234)
rng = np.random.RandomState(1234)


def prepare_dataset_aif360(protected_attribute, model_predictions):
    labels = model_predictions['y_true'].reset_index(drop=True)  # ground truth labels
    predictions = model_predictions['y_pred'].reset_index(drop=True)  # predicted labels
    attribute_column = model_predictions[protected_attribute]

    # Concatenate
    true_values = pd.concat([labels, attribute_column], axis=1)

    # drop nans -> there are no nans in the prediction files since we have replaced them with the majority class in
    nan_idx = true_values.loc[pd.isna(true_values[protected_attribute]), :].index
    true_values = true_values[true_values[protected_attribute].notna()]
    predictions.drop(nan_idx, inplace=True)

    # convert protected attribute from boolean to int
    # true_values[protected_attribute] = true_values[protected_attribute].astype(int)

    # reset index
    true_values = true_values.reset_index(drop=True)
    predictions = predictions.reset_index(drop=True)
    return true_values, predictions


def aif360_model(true_values, predictions, protected_attribute, privileged_class, favorable_class=1):
    # An aif360 library class
    dataset = StandardDataset(true_values,  # the real values dataset
                              label_name='y_true',
                              favorable_classes=[favorable_class],
                              protected_attribute_names=[protected_attribute],
                              privileged_classes=[privileged_class])

    # Dataset containing predictions
    dataset_pred = dataset.copy()
    dataset_pred.labels = predictions
    return get_aif360_metrics(dataset, dataset_pred)


def print_aif360_result(metric_pred, classified_metric):
    # evaluating in terms of fairness
    # definitions = https://aif360.mybluemix.net/ - keep only disparate impact ratio here for testing
    result = {
        'statistical_parity_difference': metric_pred.statistical_parity_difference(),  # The difference of the
        # rate of favorable outcomes received by the unprivileged group to the privileged group (WAE view)
        'disparate_impact': metric_pred.disparate_impact(),  # the proportion of the unprivileged group that
        # received the positive outcome divided by the proportion of the privileged group that received the
        # positive outcome (WAE view)
        'equal_opportunity_difference': classified_metric.equal_opportunity_difference(),  # The difference of
        # true positive rates between the unprivileged and the privileged groups (between the two worldviews,
        # i.e., WYSIWYG and WAE).
        'average_absolute_odds_difference': classified_metric.average_abs_odds_difference(),  # fair near 0
        'error_rate_difference': classified_metric.error_rate_difference(),
        'error_rate_ratio': classified_metric.error_rate_ratio(),
        'false_discovery_rate_ratio': classified_metric.false_discovery_rate_ratio(),
        'false_negative_rate_ratio': classified_metric.false_negative_rate_ratio(),
        'false_omission_rate_ratio': classified_metric.false_omission_rate_ratio(),
        'false_positive_rate_ratio': classified_metric.false_positive_rate_ratio(),
        # 'performance_measures': classified_metric.performance_measures(privileged=True),
        'true_positive_rate_difference': classified_metric.true_positive_rate_difference(),
        # 'accuracy': classified_metric.accuracy()
    }

    for r in result.keys():
        print("{}: {}".format(r, result.get(r)))
    return result


def get_aif360_metrics(dataset, dataset_pred):
    # for attr in dataset_pred.protected_attribute_names:
    attr = dataset_pred.protected_attribute_names[0]

    idx = dataset_pred.protected_attribute_names.index(attr)
    privileged_groups = [{attr: dataset_pred.privileged_protected_attributes[idx][0]}]
    unprivileged_groups = [{attr: dataset_pred.unprivileged_protected_attributes[idx][0]}]

    classified_metric = ClassificationMetric(dataset, dataset_pred, unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)

    metric_pred = BinaryLabelDatasetMetric(dataset_pred, unprivileged_groups=unprivileged_groups,
                                           privileged_groups=privileged_groups)

    return metric_pred, classified_metric


def print_aif360_accuracy_metrics(classified_metric):
    print("Privileged Group: {}".format(classified_metric.binary_confusion_matrix(privileged=True)))
    print("Unprivileged Group: {}".format(classified_metric.binary_confusion_matrix(privileged=False)))
    print()
    print("Overall Accuracy: {}".format(classified_metric.accuracy()))
    print("Privileged Group: {}".format(classified_metric.accuracy(privileged=True)))
    print("Unprivileged Group: {}".format(classified_metric.accuracy(privileged=False)))
    print()
    print("Overall Error Rate: {}".format(classified_metric.error_rate()))
    print("Privileged Group: {}".format(classified_metric.error_rate(privileged=True)))
    print("Unprivileged Group: {}".format(classified_metric.error_rate(privileged=False)))


# https://sites.google.com/site/lisaywtang/tech/python/scikit/auc-conf-interval
def get_ci_auc(y_true, y_pred, metric='AUROC'):
    from scipy.stats import sem
    from sklearn.metrics import roc_auc_score
    n_bootstraps = 500
    bootstrapped_scores = []
    for i in range(n_bootstraps):
        # bootstrap by sampling with replacement on the prediction indices
        indices = rng.random_integers(0, len(y_pred) - 1, len(y_pred))
        # indices = rng.randint(0, len(y_pred) + 1)

        if len(np.unique(y_true[indices])) < 2:
            # We need at least one positive and one negative sample for ROC AUC
            # to be defined: reject the sample
            continue

        try:
            roc_auc_score(y_true[indices,1], y_pred[indices,1])
        except ValueError:
            continue

        if metric == 'AUROC':
            score = roc_auc_score(y_true[indices,1], y_pred[indices,1], average='micro')
        elif metric == 'AUPRC Micro':
            score = average_precision_score(y_true[indices,1], y_pred[indices,1], average='micro')
        elif metric == 'AUPRC Macro':
            score = average_precision_score(y_true[indices,1], y_pred[indices,1], average='macro')
        elif metric == 'Accuracy':
            score = accuracy_score(np.argmax(y_true[indices], axis=1), np.argmax(y_pred[indices], axis=1))
        elif metric == 'F1 Macro':
            score = f1_score(np.argmax(y_true[indices], axis=1), np.argmax(y_pred[indices], axis=1), average='macro')
        elif metric == 'F1 Micro':
            score = f1_score(np.argmax(y_true[indices], axis=1), np.argmax(y_pred[indices], axis=1), average='micro')
        elif metric == 'F1 Weighted':
            score = f1_score(np.argmax(y_true[indices], axis=1), np.argmax(y_pred[indices], axis=1), average='weighted')
        elif metric == 'Precision':
            score = precision_score(np.argmax(y_true[indices], axis=1), np.argmax(y_pred[indices], axis=1), average='macro')
        elif metric == 'Recall':
            score = recall_score(np.argmax(y_true[indices], axis=1), np.argmax(y_pred[indices], axis=1), average='macro')
        elif metric == 'Balanced Accuracy':
            score = balanced_accuracy_score(np.argmax(y_true[indices], axis=1), np.argmax(y_pred[indices], axis=1))
        else:
            raise ValueError('Please provide correct metric parameter.')

        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

    return confidence_lower, confidence_upper
