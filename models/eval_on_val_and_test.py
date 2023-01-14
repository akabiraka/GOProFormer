import sys
sys.path.append("../GOProFormer")

import torch
torch.cuda.empty_cache()

from transformer.config import Config
from models.Dataset import SeqAssociationDataset, TermsGraph, get_class_weights
import models.MultimodalTransformer as MultimodalTransformer

from models.val import run_val
from models.test import run_test

config = Config()
out_filename = config.get_model_name()
out_filename = out_filename+"_perf" # _loss, _perf
print(f"Running test: {out_filename}")


# loading model, criterion
model = MultimodalTransformer.Model(config=config).to(config.device)
class_weights = get_class_weights(config.species, config.GO, config.data_generation_process).to(config.device)
label_pred_criterion = torch.nn.BCEWithLogitsLoss(class_weights)

# loading learned weights
checkpoint = torch.load(f"outputs/models/{out_filename}.pth")
model.load_state_dict(checkpoint['model_state_dict'])


# loading dataset
terms_graph = TermsGraph(config.species, config.GO, config.data_generation_process, config.n_samples_from_pool)

val_loss, val_tmax, val_fmax, val_smin, val_aupr = run_val(model, terms_graph, label_pred_criterion)
test_loss, test_tmax, test_fmax, test_smin, test_aupr = run_test(model, terms_graph, label_pred_criterion)

