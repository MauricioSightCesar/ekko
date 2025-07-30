import time
import gc
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

import numpy as np

def log_resource_metrics(logger, metrics):
    logger.info(f"--------- Resource Performance --------")
    logger.info(f"CPU Inference Time: \t{(metrics['cpu_inference_time'] or 0):.6f} ms")
    logger.info(f"GPU Inference Time: \t{(metrics['gpu_inference_time'] or 0):.6f} ms")
    logger.info(f"MPS Inference Time: \t{(metrics['mps_inference_time'] or 0):.6f} ms")
    logger.info(f"Model size: \t\t{metrics['model_size_mb']:.6f} MB")
    logger.info('')

def __log_metrics(logger, metrics, cm):
    # Overall metrics
    logger.info('')
    logger.info("--- Overall Performance --")
    logger.info("AUCROC: \t\t\t{:.4f}".format(metrics['AUCROC']))
    logger.info("Accuracy: \t\t\t{:.4f}".format(metrics['Accuracy']))
    logger.info("FPR: \t\t\t\t{:.4f}".format(metrics['FPR']))
    logger.info("TPR: \t\t\t\t{:.4f}".format(metrics['TPR']))
    logger.info("Precision: \t\t\t{:.4f}".format(metrics['Precision']))
    logger.info("F1-score: \t\t\t{:.4f}".format(metrics['F1-score']))
    logger.info("Threshold: \t\t\t{:.4f}".format(metrics['optimal_threshold']))
    logger.info('')

    # TPR per label
    if 'tpr_per_label' in metrics:
        logger.info("------- TPR per Label -------")
        for label, tpr in metrics['tpr_per_label'].items():
            label = label + ':\t\t\t' if len(label) < 15 else label + ':\t'
            logger.info(f"{label}{tpr:.4f}")
        logger.info('')

    if 'aucroc_per_label' in metrics:
        logger.info("------- AUCROC per Label -------")
        for label, tpr in metrics['aucroc_per_label'].items():
            label = label + ':\t\t\t' if len(label) < 15 else label + ':\t'
            logger.info(f"{label}{tpr:.4f}")
        logger.info('')

    # Confusion matrix
    logger.info("------ Confusion Matrix ------")
    tn, fp, fn, tp = cm
    total = tn + fp + fn + tp
    tn_str = f'{tn} ({tn / (total) * 100:.2f}%)'
    fp_str = f'{fp} ({fp / (total) * 100:.2f}%)'
    fn_str = f'{fn} ({fn / (total) * 100:.2f}%)'
    tp_str = f'{tp} ({tp / (total) * 100:.2f}%)'
    logger.info("TN: {} \tFP: {}".format(tn_str, fp_str))
    logger.info("FN: {} \tTP: {}".format(fn_str, tp_str))
    logger.info('')


def get_sentence_results(y_pred, y_true, reduction='sum'):
    y_scores = []
    for sentence in y_pred:
        sentence_scores = 0

        for label, score in sentence:
            sentence_scores += score
        
        if reduction == 'mean':
            y_scores.append(sentence_scores / len(sentence))
        elif reduction == 'sum':
            y_scores.append(sentence_scores)

    y_true_binary = []
    for sentence in y_true:
        sentence_label = 0
        for label in sentence:
            sentence_label = 1 if label != '0' else sentence_label
        y_true_binary.append(sentence_label)

    y_scores = np.array(y_scores)
    y_true_binary = np.array(y_true_binary)

    return y_scores, y_true_binary


def flatten_data(y_pred, y_true):
    y_scores = []
    y_pred_labels = []
    for sentence in y_pred:
        for label, score in sentence:
            y_scores.append(score)
            y_pred_labels.append(label)

    y_true_binary = []
    y_true_labels = []
    for sentence in y_true:
        for label in sentence:
            y_true_binary.append(1 if label != '0' else 0)
            y_true_labels.append(label)

    y_pred_labels = np.array(y_pred_labels)
    y_scores = np.array(y_scores)
    y_true_labels = np.array(y_true_labels)
    y_true_binary = np.array(y_true_binary)
    return y_pred_labels, y_scores, y_true_labels, y_true_binary

def get_threshold_youden_index(y_true, y_pred):
    # Calculate Youden index to determine optimal threshold
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    youden_index = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[youden_index]
    return optimal_threshold


def get_overall_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    acc = (tp+tn)/(tp+tn+fp+fn)
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    precision = tp/(tp+fp)
    f1 = (2*tpr*precision)/(tpr+precision)
    return {'Accuracy':acc,'TPR':tpr,'FPR':fpr,'Precision':precision,'F1-score':f1}

def roc_auc_score_each_label(y_pred_labels, y_scores, y_true_labels):
    unique_labels = set(y_true_labels)
    auc_scores = {}
    
    for label in unique_labels:
        if label == '0':
            continue

        y_true_idx = [i for i, l in enumerate(y_true_labels) if l == label]
        y_pred_idx = [i for i, l in enumerate(y_pred_labels) if l == label]

        curr_y_true = []
        curr_y_scores = []

        for i in range(len(y_true_labels)):
            if i in y_true_idx and i in y_pred_idx:
                curr_y_true.append(0 if y_true_labels[i] == '0' else 1)
                curr_y_scores.append(y_scores[i])
            
            elif i in y_true_idx:
                curr_y_true.append(0 if y_true_labels[i] == '0' else 1)
                curr_y_scores.append(0)

            elif i in y_pred_idx:
                curr_y_true.append(0)
                curr_y_scores.append(y_scores[i])


        auc_score = roc_auc_score(curr_y_true, curr_y_scores)
        auc_scores[label] = float(auc_score)
    
    return auc_scores

def get_tpr_per_label(y_labels, y_pred):
    aux_df = pd.DataFrame({'label':y_labels,'prediction':y_pred})
    total_per_label = aux_df['label'].value_counts().to_dict()
    correct_predictions_per_label = aux_df.query('prediction == label').groupby('label').size().to_dict()
    tpr_per_label = {}
    for label_label, total in total_per_label.items():
      tp = correct_predictions_per_label[label_label] if label_label in correct_predictions_per_label else 0
      tpr = tp/total
      tpr_per_label[label_label] = tpr
    return tpr_per_label

def get_metrics_by_token(y_pred, y_true, logger=None):
    y_pred_labels, y_scores, y_true_labels, y_true_binary = flatten_data(y_pred, y_true)

    aucroc = roc_auc_score(y_true_binary, y_scores)
    # aucroc_per_label = roc_auc_score_each_label(y_pred_labels, y_scores, y_true_labels)

    threshold = get_threshold_youden_index(y_true_binary, y_scores)

    overall_result = get_overall_metrics(y_true_binary, y_scores >= threshold)

    results_per_label = get_tpr_per_label(y_true_labels, np.where(y_scores >= threshold, y_pred_labels, '0'))

    result = {'AUCROC': aucroc, **overall_result, 'optimal_threshold': threshold}
    metrics_serializable = {k: float(v) for k, v in result.items()}
    metrics_serializable['tpr_per_label'] = results_per_label
    # metrics_serializable['aucroc_per_label'] = aucroc_per_label

    if logger is not None:
        logger.info('----- RESULTS BY TOKEN IN SENTENCES ------')
        __log_metrics(logger, metrics_serializable, confusion_matrix(y_true_binary, y_scores > threshold).ravel())

    return metrics_serializable

def get_metrics_by_sentence(y_pred, y_true, logger=None):
    y_scores, y_true_binary = get_sentence_results(y_pred, y_true)

    aucroc = roc_auc_score(y_true_binary, y_scores)

    threshold = get_threshold_youden_index(y_true_binary, y_scores)

    overall_result = get_overall_metrics(y_true_binary, y_scores >= threshold)

    result = {'AUCROC': aucroc, **overall_result, 'optimal_threshold': threshold}
    metrics_serializable = {k: float(v) for k, v in result.items()}

    if logger is not None:
        logger.info('----- RESULTS BY SENTENCES ------')
        __log_metrics(logger, metrics_serializable, confusion_matrix(y_true_binary, y_scores > threshold).ravel())

    return metrics_serializable

def get_metrics(y_pred, y_true, logger=None):
    start_time = time.time()
    if logger is not None:
        logger.info("Starting metrics calculation...")
        logger.info(f"Processing {len(y_pred)} predictions and {len(y_true)} ground truth labels")

    metrics_by_token = get_metrics_by_token(y_pred, y_true, logger)
    metrics_by_sentence = get_metrics_by_sentence(y_pred, y_true, logger)

    return { 'metrics_by_token': metrics_by_token, 'metrics_by_sentence': metrics_by_sentence }
