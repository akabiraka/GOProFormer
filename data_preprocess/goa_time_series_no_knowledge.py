import sys
sys.path.append("../GOProFormer")

import numpy as np
# np.random.seed(123456)
from sklearn.model_selection import train_test_split

import utils as Utils
from helpers import *

data_generation_process = "time_series_no_knowledge"

t0 = 20000101 # 1 Jan 2000
t1 = 20220114 # 14 Jan 2022

timeline = list(range(t0, t1, 3000))
timeseries = [(timeline[i], timeline[i+1]) for i in range(len(timeline)-1)]
# print(timeseries)
train_timeseries, test_timeseries = train_test_split(timeseries, test_size=0.20)
train_timeseries, val_timeseries = train_test_split(train_timeseries, test_size=0.10)
print(len(train_timeseries), len(val_timeseries), len(test_timeseries))
# print(train_timeseries, val_timeseries, test_timeseries)



def check_timeseries(date:int):
    for (start, end) in train_timeseries:
        if date>=start and date<end:
            return "train"
    for (start, end) in val_timeseries:
        if date>=start and date<end:
            return "val"
    for (start, end) in test_timeseries:
        if date>=start and date<end:
            return "test"
        

def generate_dataset(GOname="BP", GO_terms_set=bp_set, cutoff_value=125, atleast_n_annots=0):
    train_set, val_set, test_set = {}, {}, {} # uniprot_id, set of annots

    f = open(f"data/downloads/{species}_goa.gpa", "r")
    for i, line in enumerate(f.readlines()):
        # print(f"line no: {i}")

        do_continue, uniprot_id, GO_id, date, evidence = validate_line(i, line)
        if do_continue: continue

        if GO_id in GO_terms_set and date<=t1: 
            if check_timeseries(date) == "train": train_set = update_annot_dict(uniprot_id, GO_id, train_set)
            elif check_timeseries(date) == "val": val_set = update_annot_dict(uniprot_id, GO_id, val_set)
            elif check_timeseries(date) == "test": test_set = update_annot_dict(uniprot_id, GO_id, test_set)
        
        # if i==32: break # for debugging

    print(t0,t1)
    print(f"#-prots in train, val, test: {len(train_set)}, {len(val_set)}, {len(test_set)}")
    print_summary(list(train_set.items()))

    train_set, val_set = remove_dev_uniprotids_from_test(train_set, val_set) # inplace operation
    train_set, test_set = remove_dev_uniprotids_from_test(train_set, test_set) # inplace operation
    val_set, test_set = remove_dev_uniprotids_from_test(val_set, test_set) # inplace operation
    print(f"#-prots in train, val, test after keeping only no-knowledge proteins in val and test: {len(train_set)}, {len(val_set)}, {len(test_set)}")
    print_summary(list(train_set.items())) # the number of annotations is increased in train_set, becasue no-knowledge proteins annotations are be added from val/test into train.


    train_set = apply_true_path_rule(train_set)
    val_set = apply_true_path_rule(val_set)
    test_set = apply_true_path_rule(test_set)
    print("\nSummary of sets after applying true-path-rule: ")
    print_summary(list(train_set.items()))
    print_summary(list(val_set.items()))
    print_summary(list(test_set.items()))


    studied_terms = compute_studied_terms(train_set, cutoff_value)
    save_studied_terms(list(studied_terms), GOname, data_generation_process)
    create_terms_relation_matrix(species, GOname, data_generation_process, relation="adjacency")
    print(f"\n#-of studied terms: {len(studied_terms)}")

    train_set = update_annots_with_studied_terms(train_set, studied_terms)
    val_set = update_annots_with_studied_terms(val_set, studied_terms)
    test_set = update_annots_with_studied_terms(test_set, studied_terms)
    print("\nSummary of sets after updating annotations with studied GO terms: ")
    print_summary(list(train_set.items()))
    print_summary(list(val_set.items()))
    print_summary(list(test_set.items()))

    remove_proteins_annotated_to_n_or_less_terms(train_set, n=atleast_n_annots)
    remove_proteins_annotated_to_n_or_less_terms(val_set, n=atleast_n_annots)
    remove_proteins_annotated_to_n_or_less_terms(test_set, n=atleast_n_annots)
    print("\nSummary of sets after removing proteins having <=n annotations: ")
    print_summary(list(train_set.items()))
    print_summary(list(val_set.items()))
    print_summary(list(test_set.items()))

    remove_nonexist_uniprotids_from_dev_test(train_set) # inplace operation
    remove_nonexist_uniprotids_from_dev_test(val_set) # inplace operation
    remove_nonexist_uniprotids_from_dev_test(test_set) # inplace operation
    print("\nSummary of sets after removing nonexist uniprotids: ")
    print_summary(list(train_set.items()))
    print_summary(list(val_set.items()))
    print_summary(list(test_set.items()))

    Utils.save_as_pickle(list(train_set.items()), f"data/goa/{species}/train_val_test_set/{data_generation_process}/{GOname}/train.pkl")
    Utils.save_as_pickle(list(val_set.items()), f"data/goa/{species}/train_val_test_set/{data_generation_process}/{GOname}/val.pkl")
    Utils.save_as_pickle(list(test_set.items()), f"data/goa/{species}/train_val_test_set/{data_generation_process}/{GOname}/test.pkl")


generate_dataset(GOname="BP", GO_terms_set=bp_set, cutoff_value=150, atleast_n_annots=0)
generate_dataset(GOname="CC", GO_terms_set=cc_set, cutoff_value=25, atleast_n_annots=0)
generate_dataset(GOname="MF", GO_terms_set=mf_set, cutoff_value=25, atleast_n_annots=0)

