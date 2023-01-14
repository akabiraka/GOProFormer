import sys
sys.path.append("../GOProFormer")

import models.MultimodalTransformer as MultimodalTransformer
import models.eval_metrics as eval_metrics
from transformer.config import Config
from data_preprocess.GO import Ontology
import utils as Utils
from torch.utils.data import DataLoader
from models.Dataset import SeqAssociationDataset

config = Config()

# for evaluation purposes
go_rels_val = Ontology('data/downloads/go.obo', with_rels=True)
term_to_idx_dict_val = Utils.load_pickle(f"data/goa/{config.species}/train_val_test_set/{config.data_generation_process}/{config.GO}/studied_terms.pkl")
idx_to_term_dict_val = {i:term for term, i in term_to_idx_dict_val.items()}
terms_set_val = set(term_to_idx_dict_val.keys())

train_dataset_val = Utils.load_pickle(f"data/goa/{config.species}/train_val_test_set/{config.data_generation_process}/{config.GO}/train.pkl") # list of uniprot_id, set([terms])
print(f"Length of train set: {len(train_dataset_val)}")

val_set = Utils.load_pickle(f"data/goa/{config.species}/train_val_test_set/{config.data_generation_process}/{config.GO}/val.pkl")
print(f"Length of eval set: {len(val_set)}")


val_annotations = [annots for uniprot_id, annots in val_set]
train_annotations_val = [annots for uniprot_id, annots in train_dataset_val]
go_rels_val.calculate_ic(train_annotations_val + val_annotations)

print("Log: finished computing ic")

val_dataset = SeqAssociationDataset(config.species, config.GO, config.data_generation_process, dataset="val")
val_loader = DataLoader(val_dataset, config.batch_size, shuffle=False)
print(f"val batches: {len(val_loader)}")

def run_val(model, terms_graph, label_pred_criterion):
    val_loss, true_scores, pred_scores = MultimodalTransformer.val(model, val_loader, terms_graph, label_pred_criterion, config.device)

    tmax, fmax, smin, aupr = eval_metrics.Fmax_Smin_AUPR(pred_scores, val_set, idx_to_term_dict_val, go_rels_val, terms_set_val, val_annotations)
    # tmax, fmax, smin, aupr = eval_metrics.Fmax_Smin_AUPR(pred_scores)
    # micro_avg_f1 = eval_metrics.MicroAvgF1_TPR(true_scores, pred_scores)
    # micro_avg_f1 = eval_metrics.MicroAvgF1(true_scores, pred_scores)
    # micro_avg_precision = eval_metrics.MicroAvgPrecision(true_scores, pred_scores)
    # fmax = eval_metrics.Fmax(true_scores, pred_scores)

    return val_loss, tmax, fmax, smin, aupr
