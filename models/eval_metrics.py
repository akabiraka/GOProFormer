import sys
sys.path.append("../GOProFormer")

import numpy as np
import pandas as pd
import math


import sklearn.metrics as metrics
import matplotlib.pyplot as plt





def evaluate_annotations(go, real_annots, pred_annots):
    total = 0
    p = 0.0
    r = 0.0
    p_total= 0
    ru = 0.0
    mi = 0.0
    fps = []
    fns = []
    for i in range(len(real_annots)):
        if len(real_annots[i]) == 0:
            continue
        tp = set(real_annots[i]).intersection(set(pred_annots[i]))
        fp = pred_annots[i] - tp
        fn = real_annots[i] - tp
        for go_id in fp:
            mi += go.get_ic(go_id)
        for go_id in fn:
            ru += go.get_ic(go_id)
        fps.append(fp)
        fns.append(fn)
        tpn = len(tp)
        fpn = len(fp)
        fnn = len(fn)
        total += 1
        recall = tpn / (1.0 * (tpn + fnn))
        r += recall
        if len(pred_annots[i]) > 0:
            p_total += 1
            precision = tpn / (1.0 * (tpn + fpn))
            p += precision
    ru /= total
    mi /= total
    r /= total
    if p_total > 0:
        p /= p_total
    f = 0.0
    if p + r > 0:
        f = 2 * p * r / (p + r)
    s = math.sqrt(ru * ru + mi * mi)
    return f, p, r, s, ru, mi, fps, fns



def Fmax_Smin_AUPR(pred_scores, test_dataset, idx_to_term_dict, go_rels, terms_set, test_annotations):
    # pred_scores = np.random.rand(869, 244)
    fmax = 0.0
    tmax = 0.0
    precisions = []
    recalls = []
    smin = 1000000.0
    rus = []
    mis = []
    for t in range(1, 101): # the range in this loop has influence in the AUPR output
        threshold = t / 100.0
        preds = []
        for i, (uniprot_id, annots) in enumerate(test_dataset):
            pred_terms_indies = np.where(pred_scores[i] > threshold)[0]
            annots = set([idx_to_term_dict.get(j) for j in pred_terms_indies])

            new_annots = set()
            for go_id in annots:
                ancestors = go_rels.get_anchestors(go_id)
                ancestors = set(ancestors).intersection(terms_set) # taking ancestors only in the studied terms
                new_annots = new_annots | ancestors # set union
            preds.append(new_annots) # list of sets

        fscore, prec, rec, s, ru, mi, fps, fns = evaluate_annotations(go_rels, test_annotations, preds)
        
        precisions.append(prec)
        recalls.append(rec)
        
        if fmax < fscore:
            fmax = fscore
            tmax = threshold
        if smin > s:
            smin = s


    print(f'    threshold: {tmax}')
    print(f'    Fmax: {fmax:0.4f}')
    print(f'    Smin: {smin:0.4f}')
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    sorted_index = np.argsort(recalls)
    recalls = recalls[sorted_index]
    precisions = precisions[sorted_index]
    aupr = np.trapz(precisions, recalls)
    print(f'    AUPR: {aupr:0.4f}')
    
    return tmax, fmax, smin, aupr


# def apply_true_path_rule_on_pred_scores(pred_scores, th):#, idx_to_term_dict:dict, term_to_idx_dict:dict, terms_set:set, go_rels:Ontology):
#     ext_pred_scores = np.zeros(pred_scores.shape)
#     rows, cols = pred_scores.shape
#     for i in range(rows):
#         for j in range(cols):
#             if not pred_scores[i, j] > th: continue

#             ext_pred_scores[i, j] = 1.0

#             ancestors = go_rels.get_anchestors(idx_to_term_dict.get(j))
#             ancestors = set(ancestors).intersection(terms_set) # taking ancestors those are in the studied terms
#             for ancestor in ancestors:
#                 ext_pred_scores[i, term_to_idx_dict.get(ancestor)] = 1.0
    
#     return ext_pred_scores

# # TPR: true path rule
# def MicroAvgF1_TPR(true_scores:np.ndarray, pred_scores:np.ndarray):#, idx_to_term_dict:dict, term_to_idx_dict:dict, terms_set:set, go_rels:Ontology):
#     best_micro_avg_f1 = 0.0
#     for t in range(1, 101):
#         decision_th = t/100
#         ext_pred_scores = apply_true_path_rule_on_pred_scores(pred_scores, decision_th)
#         micro_avg_f1 = metrics.f1_score(true_scores, ext_pred_scores, average="micro")
#         if micro_avg_f1 > best_micro_avg_f1:
#             best_micro_avg_f1 = micro_avg_f1
    
#     print(f"    FmMicroAvgF1_TPR: {best_micro_avg_f1:0.4f} at decision_th: {decision_th}")
#     return best_micro_avg_f1


def MicroAvgF1(true_scores:np.ndarray, pred_scores:np.ndarray):
    pred_labels = np.where(pred_scores > 0.5, 1, 0)
    micro_avg_f1 = metrics.f1_score(true_scores, pred_labels, average="micro")
    print(f"    FmMicroAvgF1: {micro_avg_f1:0.4f}")
    return micro_avg_f1



def MicroAvgPrecision(true_scores:np.ndarray, pred_scores:np.ndarray):
    micro_avg_prec = metrics.average_precision_score(true_scores, pred_scores, average="micro")
    print(f'    MicroAvgPrecision: {micro_avg_prec:0.4f}')
    return micro_avg_prec



def Fmax(y_true:np.ndarray, y_scores:np.ndarray):
    fmax=0.0
    decision_th = 0.0

    for t in range(1, 101):
        th = t/100.0 # according to Article1
        y_pred = np.where(y_scores>th, 1, 0)

        prec = metrics.precision_score(y_true, y_pred, average="micro", zero_division=1)
        rec = metrics.recall_score(y_true, y_pred, average="micro", zero_division=1)

        f = (2*prec*rec) / (prec+rec)
        if f > fmax: 
            fmax = f
            decision_th = th
            
    print(f"    Fmax: {fmax:0.4f} at decision_th: {decision_th}")
    return fmax



def AUROC(y_true:np.ndarray, y_scores:np.ndarray, pltpath=None):
    y_true, y_scores = y_true.flatten(), y_scores.flatten()
    fpr, tpr, t= metrics.roc_curve(y_true, y_scores)
    auroc = metrics.auc(fpr, tpr)
    # auroc = metrics.roc_auc_score(y_true, y_scores) #same as previous 2 lines
    print(f"    AUROC: {auroc:0.4f}")

    # plot_area_under_curve(fpr, tpr, auroc, "True-positive rate", "False-positive rate", pltpath)
    return auroc



def AUPR(y_true:np.ndarray, y_scores:np.ndarray, pltpath=None):
    y_true, y_scores = y_true.flatten(), y_scores.flatten()
    prec, rec, t = metrics.precision_recall_curve(y_true, y_scores)
    aupr = metrics.auc(rec, prec)
    print(f"    AUPR: {aupr:0.4f}")

    # plot_area_under_curve(rec, prec, aupr, "Recall", "Precision", pltpath)
    return aupr


def plot_area_under_curve(x, y, area_value, xlabel, ylabel, pltpath=None):
    # i.e x=recalls, y=precissions
    plt.figure()
    lw = 2
    plt.plot(x, y, color='darkorange', lw=lw, label=f'Area={area_value:0.2f}')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if pltpath is not None: plt.savefig(pltpath, dpi=300, format="pdf", bbox_inches='tight', pad_inches=0.0)
    else: plt.show()
