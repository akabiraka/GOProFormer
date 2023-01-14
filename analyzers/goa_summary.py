import sys
sys.path.append("../GOProFormer")
import pandas as pd

from data_preprocess.GO import Ontology, NAMESPACES

species = "yeast" # human, yeast
# for GO in ["BP", "CC", "MF"]:
#     dataset = "test"  # "dev", "test"
#     which_set = "dev_test_set_expanded" #dev_test_set, dev_test_set_expanded, dev_test_set_cutoff

#     # filepath = f"data/goa/{species}/separated_annotations/{GO}.csv" # for separated annotations
#     filepath = f"data/goa/{species}/{which_set}/{GO}/{dataset}.csv" # for other dev-test set
    

#     df = pd.read_csv(filepath)
#     # print(df.columns) #['line_no', 'uniprot_id', 'GO_id', 'date']

#     print(f"for {GO}")
#     print(f"    #-of annotations: {df.shape[0]}")
#     print(f"    #-of unique proteins: {len(df['uniprot_id'].unique())}")
#     print(f"    #-of unique GO terms: {len(df['GO_id'].unique())}")
#     print()



go_rels = Ontology('data/downloads/go.obo', with_rels=True)
bp_set = go_rels.get_namespace_terms(NAMESPACES["bp"])
cc_set = go_rels.get_namespace_terms(NAMESPACES["cc"])
mf_set = go_rels.get_namespace_terms(NAMESPACES["mf"])

print("\nSummary of terms:")
print(len(bp_set)) #30365
print(len(cc_set)) #4423
print(len(mf_set)) #12360
# no intersetion among these sets

GO_terms_set = mf_set

f = open(f"data/downloads/{species}_goa.gpa", "r")
all_uniprotids, GO_specific_uniprotids, terms = set(), set(), set()

for i, line in enumerate(f.readlines()):
    if not line.startswith("UniProtKB"): continue

    line_items = line.split()
    uniprot_id = line_items[1]
    GO_id = line_items[3]
    
    all_uniprotids = all_uniprotids | set([uniprot_id])

    if GO_id in GO_terms_set:
        GO_specific_uniprotids = GO_specific_uniprotids | set([uniprot_id])
        terms = terms | set([GO_id])


print("\nNumber of all unique proteins:")
print(len(all_uniprotids)) # num of unique proteins

print("\nThe number of proteins with the number of GO terms used to annotate these proteins:")
print(len(GO_specific_uniprotids)) # number of unique proteins annotated to GO-type (BP, CC, MF)
print(len(terms)) # number of terms annotated to those proteins
    