import sys
sys.path.append("../GOProFormer")

from sklearn.model_selection import train_test_split

import utils as Utils
from helpers import *
data_generation_process = "time_delay_no_knowledge"

t0 = 20200811 # 11 Aug 2020, dev deadline
t1 = 20220114 # 14 Jan 2022, test deadline

# def do(dev_annots, test_annots, terms_cutoff_value, n_annots, go):
def generate_dataset(GOname="BP", GO_terms_set=bp_set, cutoff_value=125, atleast_n_annots=0):
    dev_set, test_set = {}, {} # uniprot_id, set of annots

    f = open(f"data/downloads/{species}_goa.gpa", "r")
    for i, line in enumerate(f.readlines()):
        # print(f"line no: {i}")

        do_continue, uniprot_id, GO_id, date, evidence = validate_line(i, line)
        if do_continue: continue


        # separating annotations according to the GO={BP, CC, MF} types
        if GO_id in GO_terms_set: 
            if date<=t0: dev_set = update_annot_dict(uniprot_id, GO_id, dev_set)  
            elif date>t0 and date<=t1: test_set = update_annot_dict(uniprot_id, GO_id, test_set) 
        
        # if i==32: break # for debugging


    print(f"#-prots in dev, test: {len(dev_set)}, {len(test_set)}")
    print_summary(list(dev_set.items()))

    dev_set, test_set = remove_dev_uniprotids_from_test(dev_set, test_set) # inplace operation
    print(f"#-prots in dev, test after keeping only no-knowledge proteins in test: {len(dev_set)}, {len(test_set)}")
    print_summary(list(dev_set.items())) 

    dev_set = apply_true_path_rule(dev_set)
    test_set = apply_true_path_rule(test_set)
    print("\nSummary of sets after applying true-path-rule: ")
    print_summary(list(dev_set.items()))
    print_summary(list(test_set.items()))

    studied_terms = compute_studied_terms(dev_set, cutoff_value)
    save_studied_terms(list(studied_terms), GOname, data_generation_process)
    create_terms_relation_matrix(species, GOname, data_generation_process, relation="adjacency")
    print(f"\n#-of studied terms: {len(studied_terms)}")

    dev_set = update_annots_with_studied_terms(dev_set, studied_terms)
    test_set = update_annots_with_studied_terms(test_set, studied_terms)
    print("\nSummary of sets after updating annotations with studied GO terms: ")
    print_summary(list(dev_set.items()))
    print_summary(list(test_set.items()))

    remove_proteins_annotated_to_n_or_less_terms(dev_set, n=atleast_n_annots)
    remove_proteins_annotated_to_n_or_less_terms(test_set, n=atleast_n_annots)
    print("\nSummary of sets after removing proteins having <=n annotations: ")
    print_summary(list(dev_set.items()))
    print_summary(list(test_set.items()))

    remove_nonexist_uniprotids_from_dev_test(dev_set) # inplace operation
    remove_nonexist_uniprotids_from_dev_test(test_set) # inplace operation
    print("\nSummary of sets after removing nonexist uniprotids: ")
    print_summary(list(dev_set.items()))
    print_summary(list(test_set.items()))

    train_set, val_set = train_test_split(list(dev_set.items()), test_size=0.10)
    print("\nSummary of sets after train/val split: ")
    print_summary(train_set)
    print_summary(val_set)
    print_summary(list(test_set.items()))

    
    

    Utils.save_as_pickle(train_set, f"data/goa/{species}/train_val_test_set/{data_generation_process}/{GOname}/train.pkl")
    Utils.save_as_pickle(val_set, f"data/goa/{species}/train_val_test_set/{data_generation_process}/{GOname}/val.pkl")
    Utils.save_as_pickle(list(test_set.items()), f"data/goa/{species}/train_val_test_set/{data_generation_process}/{GOname}/test.pkl")
    

    

# applying multiview-GCN cutoff values
# generate_dataset(GOname="BP", GO_terms_set=bp_set, cutoff_value=125, atleast_n_annots=0)
# generate_dataset(GOname="CC", GO_terms_set=cc_set, cutoff_value=25, atleast_n_annots=0)
# generate_dataset(GOname="MF", GO_terms_set=mf_set, cutoff_value=25, atleast_n_annots=0)


# applying DeepGO cutoff values
# generate_dataset(GOname="BP", GO_terms_set=bp_set, cutoff_value=250, atleast_n_annots=0)
# generate_dataset(GOname="CC", GO_terms_set=cc_set, cutoff_value=50, atleast_n_annots=0)
# generate_dataset(GOname="MF", GO_terms_set=mf_set, cutoff_value=50, atleast_n_annots=0)


# mine
generate_dataset(GOname="BP", GO_terms_set=bp_set, cutoff_value=150, atleast_n_annots=0)
generate_dataset(GOname="CC", GO_terms_set=cc_set, cutoff_value=25, atleast_n_annots=0)
generate_dataset(GOname="MF", GO_terms_set=mf_set, cutoff_value=25, atleast_n_annots=0)
