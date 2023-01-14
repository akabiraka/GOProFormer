import sys
sys.path.append("../GOProFormer")

import pandas as pd
import torch
from torch.utils.data import Dataset
import utils as Utils
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import random

class SeqAssociationDataset(Dataset):
    def __init__(self, species, GO, data_generation_process, dataset="train") -> None:
        super(SeqAssociationDataset, self).__init__()
        self.species = species
        self.GO = GO
        
        self.dataset_annots = Utils.load_pickle(f"data/goa/{species}/train_val_test_set/{data_generation_process}/{GO}/{dataset}.pkl")
        self.terms_dict = Utils.load_pickle(f"data/goa/{species}/train_val_test_set/{data_generation_process}/{GO}/studied_terms.pkl")
        

    def __len__(self):
        return len(self.dataset_annots)


    def generate_true_label(self, annots):
        y_true = torch.zeros(len(self.terms_dict), dtype=torch.float32)
        for term in annots:
            y_true[self.terms_dict[term]] = 1.
        return y_true

    def get_seq_representation(self, uniprot_id):
        seq_rep = Utils.load_pickle(f"data/uniprotkb/{self.species}_sequences_rep_mean/{uniprot_id}.pkl") # shape: [esm1b_embed_dim]
        return seq_rep

    def __getitem__(self, i):
        uniprot_id, annots = self.dataset_annots[i]
        # print(uniprot_id, annots)
        
        y_true = self.generate_true_label(annots) # shape: [n_terms]
        seq_rep = self.get_seq_representation(uniprot_id) # shape: [esm1b_embed_dim]

        return uniprot_id, seq_rep, y_true


    


class TermsGraph(object):
    def __init__(self, species, GO, data_generation_process, n_samples_from_pool=5) -> None:
        self.species = species
        self.GO = GO
        self.n_samples = n_samples_from_pool

        self.terms_dict = Utils.load_pickle(f"data/goa/{species}/train_val_test_set/{data_generation_process}/{GO}/studied_terms.pkl")
        self.train_annots = Utils.load_pickle(f"data/goa/{species}/train_val_test_set/{data_generation_process}/{GO}/train.pkl")

        self.GOid_vs_uniprotids_dict = self.terms_annotated_to()
        # self.terms_ancestors = Utils.load_pickle(f"data/goa/{species}/train_val_test_set/{data_generation_process}/{GO}/ancestors.pkl")
        self.terms_adjacency = Utils.load_pickle(f"data/goa/{species}/train_val_test_set/{data_generation_process}/{GO}/adjacency.pkl")
    

    def terms_annotated_to(self):
        GOid_vs_uniprotids_dict = {}
        for term in self.terms_dict.keys():
            uniprotid_list = []
            for uniprot_id, annots in self.train_annots:
                if term in annots:
                    uniprotid_list.append(uniprot_id)

            GOid_vs_uniprotids_dict[term] = set(uniprotid_list)
        return GOid_vs_uniprotids_dict


    def get(self, crnt_uniprotid_list):
        # the pool excludes crnt_uniprot_ids
        nodes = []
        for term, id in self.terms_dict.items():
            # print(term, id)
            uniprotids_set = self.GOid_vs_uniprotids_dict[term]
            
            term_seq_features = self.get_term_seq_features(uniprotids_set, crnt_uniprotid_list) # shape: [n_samples, esm1b_embed_dim]
            # print(term_seq_features.shape)
            nodes.append(term_seq_features)
            # break

        data = {}
        data["nodes"] = torch.stack(nodes)
        # data["ancestors_rel_matrix"] = torch.tensor(self.terms_ancestors, dtype=torch.float32)# dtype=torch.bool) # torch.logical_not()
        # data["adjacency_rel_matrix"] = torch.tensor(self.terms_adjacency, dtype=torch.float32)

        # data["ancestors_rel_matrix"] = torch.logical_not(torch.tensor(self.terms_ancestors, dtype=torch.bool))
        data["adjacency_rel_matrix"] = torch.logical_not(torch.tensor(self.terms_adjacency, dtype=torch.bool))
        return data    


    def get_term_seq_features(self, uniprotids, crnt_uniprotid_list):
        uniprotids = set(uniprotids) - set(crnt_uniprotid_list) # removing current uniprotids from seq-feature pool
        uniprotids = random.sample(uniprotids, self.n_samples)
        features = []
        for uniprot_id in uniprotids:
            seq_rep = Utils.load_pickle(f"data/uniprotkb/{self.species}_sequences_rep_mean/{uniprot_id}.pkl") # shape: [esmb_embed_dim]
            features.append(seq_rep)
        
        features = torch.stack(features)
        # print(features.shape) # shape: [n_samples, esm1b_embed_dim]
        return features


# sample usage
# val_dataset = SeqAssociationDataset("yeast", "BP", "time_series_no_knowledge", dataset="test")
# print(val_dataset.__len__())
# uniprot_id, seq_rep, y_true = val_dataset.__getitem__(0)
# print(uniprot_id, seq_rep.shape, y_true.shape) # ie: P25639 torch.Size([768]) torch.Size([245])

# terms_graph = TermsGraph("yeast", "CC", n_samples_from_pool=5).get([])
# print(terms_graph["nodes"].shape, terms_graph["ancestors_rel_matrix"].shape) # ie: torch.Size([244, 5, 768]) torch.Size([244, 244])



# deprecated
# def get_terms_dataset(species, GO):
#     GO_dict = Utils.load_pickle(f"data/goa/{species}/studied_GO_id_to_index_dicts/{GO}.pkl")

#     data = {}
#     data["nodes"] = torch.tensor(list(GO_dict.values())) # node embeddings from 0 to vocab_size-1
    
#     ancestors = Utils.load_pickle(f"data/goa/{species}/studied_GO_terms_relation_matrix/{GO}_ancestors.pkl")
#     data["ancestors_rel_matrix"] = torch.logical_not(torch.tensor(ancestors, dtype=torch.bool))

#     children = Utils.load_pickle(f"data/goa/{species}/studied_GO_terms_relation_matrix/{GO}_children.pkl")
#     data["children_rel_matrix"] = torch.tensor(children, dtype=torch.float32)


#     print(f"#-terms: {data['nodes'].shape}")
#     print(f"ancestors_rel_matrix: {data['ancestors_rel_matrix'].shape}")
#     print(f"children_rel_matrix: {data['children_rel_matrix'].shape}")
#     return data

    

def get_class_weights(species, GO, data_generation_process):
    # computing class weights from the train data
    terms_dict = Utils.load_pickle(f"data/goa/{species}/train_val_test_set/{data_generation_process}/{GO}/studied_terms.pkl")
    train_annots = Utils.load_pickle(f"data/goa/{species}/train_val_test_set/{data_generation_process}/{GO}/train.pkl")
    
    classes = np.array([key for key, value in terms_dict.items()])
    all_labels = np.hstack([list(annots) for unitprot_id, annots in train_annots])

    class_weights = compute_class_weight("balanced", classes=classes, y=all_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    return class_weights

# sample usage
# class_weights = get_class_weights("yeast", "BP")
# print(class_weights)


def get_positive_class_weights(species, GO, data_generation_process):
    terms_dict = Utils.load_pickle(f"data/goa/{species}/train_val_test_set/{data_generation_process}/{GO}/studied_terms.pkl")
    train_df = pd.read_pickle(f"data/goa/{species}/train_val_test_set/{data_generation_process}/{GO}/train.pkl")

    def generate_true_label(GO_terms):
        y_true = np.zeros(len(terms_dict), dtype=np.int32)
        for term in GO_terms:
            y_true[terms_dict.get(term)] = 1
        return y_true


    all_labels = []
    for i, row in train_df.iterrows():
        GO_terms = row["GO_id"]
        y_true = generate_true_label(GO_terms)
        all_labels.append(y_true)

    all_labels = np.array(all_labels)

    positive_cls_weights = []
    for i in range(all_labels.shape[1]):
        n_pos = (all_labels[:, i]==1).sum()
        n_neg = (all_labels[:, i]==0).sum()
        weight = n_neg / n_pos
        positive_cls_weights.append(weight)


    return torch.tensor(positive_cls_weights, dtype=torch.float32)

# get_positive_class_weights("yeast", "BP")
