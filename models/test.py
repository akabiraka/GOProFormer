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
go_rels = Ontology('data/downloads/go.obo', with_rels=True)
term_to_idx_dict = Utils.load_pickle(f"data/goa/{config.species}/train_val_test_set/{config.data_generation_process}/{config.GO}/studied_terms.pkl")
idx_to_term_dict = {i:term for term, i in term_to_idx_dict.items()}
terms_set = set(term_to_idx_dict.keys())

train_dataset = Utils.load_pickle(f"data/goa/{config.species}/train_val_test_set/{config.data_generation_process}/{config.GO}/train.pkl") # list of uniprot_id, set([terms])
print(f"Length of train set: {len(train_dataset)}")

test_set = Utils.load_pickle(f"data/goa/{config.species}/train_val_test_set/{config.data_generation_process}/{config.GO}/test.pkl")
print(f"Length of eval set: {len(test_set)}")


test_annotations = [annots for uniprot_id, annots in test_set]
train_annotations = [annots for uniprot_id, annots in train_dataset]
go_rels.calculate_ic(train_annotations + test_annotations)

print("Log: finished computing ic")

test_dataset = SeqAssociationDataset(config.species, config.GO, config.data_generation_process, dataset="test")
test_loader = DataLoader(test_dataset, config.batch_size, shuffle=False)
print(f"test batches: {len(test_loader)}")

def run_test(model, terms_graph, label_pred_criterion):
    test_loss, true_scores, pred_scores = MultimodalTransformer.val(model, test_loader, terms_graph, label_pred_criterion, config.device)

    tmax, fmax, smin, aupr = eval_metrics.Fmax_Smin_AUPR(pred_scores, test_set, idx_to_term_dict, go_rels, terms_set, test_annotations)
    # tmax, fmax, smin, aupr = eval_metrics.Fmax_Smin_AUPR(pred_scores)
    # micro_avg_f1 = eval_metrics.MicroAvgF1_TPR(true_scores, pred_scores)
    # micro_avg_f1 = eval_metrics.MicroAvgF1(true_scores, pred_scores)
    # micro_avg_precision = eval_metrics.MicroAvgPrecision(true_scores, pred_scores)
    # fmax = eval_metrics.Fmax(true_scores, pred_scores)

    return test_loss, tmax, fmax, smin, aupr
