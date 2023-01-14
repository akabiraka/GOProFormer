import sys
sys.path.append("../GOProFormer")

import torch
from torch.utils.data import DataLoader
torch.cuda.empty_cache()
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from transformer.config import Config
from models.Dataset import SeqAssociationDataset, TermsGraph, get_class_weights, get_positive_class_weights
import models.MultimodalTransformer as MultimodalTransformer

import models.eval_metrics as eval_metrics
from models.val import run_val
from models.test import run_test

config = Config()
out_filename = config.get_model_name()
print(out_filename)


# loading dataset
terms_graph = TermsGraph(config.species, config.GO, config.data_generation_process, config.n_samples_from_pool)
train_dataset = SeqAssociationDataset(config.species, config.GO, config.data_generation_process, "train")
train_loader = DataLoader(train_dataset, config.batch_size, shuffle=True)
print(f"train batches: {len(train_loader)}")



# loading model, criterion, optimizer, summarywriter
model = MultimodalTransformer.Model(config=config).to(config.device)
class_weights = get_class_weights(config.species, config.GO, config.data_generation_process).to(config.device)
# pos_class_weights = get_positive_class_weights(config.species, config.GO).to(config.device)
label_pred_criterion = torch.nn.BCEWithLogitsLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
writer = SummaryWriter(f"outputs/tensorboard_runs/{out_filename}")
print("log: model loaded")
print(f"n_trainable_weights: {MultimodalTransformer.count_parameters(model)}")


best_loss, best_fmax = np.inf, 0.0
for epoch in range(config.n_epochs+1):
    train_loss = MultimodalTransformer.train(model, train_loader, terms_graph, label_pred_criterion, optimizer, config.device)
    print(f"Epoch: {epoch:03d}, train loss: {train_loss:.4f}")
    
    if epoch%10 != 0: continue

    val_loss, val_tmax, val_fmax, val_smin, val_aupr = run_val(model, terms_graph, label_pred_criterion)
    print(f"    val_loss: {val_loss:.4f}")
    test_loss, test_tmax, test_fmax, test_smin, test_aupr = run_test(model, terms_graph, label_pred_criterion)
    print(f"    test_loss: {test_loss:.4f}")
    
    

    writer.add_scalar('TrainLoss', train_loss, epoch)
    writer.add_scalar('ValLoss', val_loss, epoch)
    writer.add_scalar('TestLoss', test_loss, epoch)

    writer.add_scalar('ValFmax', val_fmax, epoch)
    writer.add_scalar('TestFmax', test_fmax, epoch)
    
    writer.add_scalar('ValTh', val_tmax, epoch)
    writer.add_scalar('TestTh', test_tmax, epoch)


    # save model dict based on loss
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    }, f"outputs/models/{out_filename}_loss.pth")


    # save model dict based on performance
    if val_fmax > best_fmax:
        best_fmax = val_fmax
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    }, f"outputs/models/{out_filename}_perf.pth")


# saving the last model
# torch.save({'epoch': config.n_epochs,
#             'model_state_dict': model.state_dict(),
#             }, f"outputs/models/{out_filename}_last.pth")    
