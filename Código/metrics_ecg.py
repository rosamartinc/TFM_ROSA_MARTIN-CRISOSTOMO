import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, hamming_loss
import torch

def compute_metrics_from_logits(logits, targets, thresh=0.5):
    probs = torch.sigmoid(logits).cpu().numpy()
    y_true = targets.cpu().numpy().astype(np.float32)
    preds  = (probs >= thresh).astype(np.float32)
    n_classes = y_true.shape[1]

    tp = (preds * y_true).sum(axis=0)
    fp = (preds * (1 - y_true)).sum(axis=0)
    fn = ((1 - preds) * y_true).sum(axis=0)

    precision_c = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp+fp)!=0)
    recall_c    = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp+fn)!=0)
    f1_c        = np.divide(2*tp, 2*tp + fp + fn, out=np.zeros_like(tp), where=(2*tp+fp+fn)!=0)

    precision_macro = float(np.nanmean(precision_c))
    recall_macro    = float(np.nanmean(recall_c))
    f1_macro        = float(np.nanmean(f1_c))

    auroc_c = np.full(n_classes, np.nan, dtype=np.float32)
    ap_c    = np.full(n_classes, np.nan, dtype=np.float32)
    for j in range(n_classes):
        yj = y_true[:, j]; pj = probs[:, j]
        if yj.min() != yj.max():
            try: auroc_c[j] = roc_auc_score(yj, pj)
            except ValueError: pass
        if yj.sum() > 0:
            try: ap_c[j] = average_precision_score(yj, pj)
            except ValueError: pass

    auroc_macro = float(np.nanmean(auroc_c)) if np.any(~np.isnan(auroc_c)) else np.nan
    ap_macro    = float(np.nanmean(ap_c))    if np.any(~np.isnan(ap_c))    else np.nan

    hamming = float(hamming_loss(y_true, preds))
    subset  = float(accuracy_score(y_true, preds))

    return dict(
        f1_macro=f1_macro, precision_macro=precision_macro, recall_macro=recall_macro,
        auroc_macro=auroc_macro, ap_macro=ap_macro,
        hamming_loss=hamming, subset_accuracy=subset,
        f1_per_class=f1_c, precision_per_class=precision_c, recall_per_class=recall_c,
        auroc_per_class=auroc_c, ap_per_class=ap_c
    )

