import sys
sys.path.append("../GOProFormer")

import numpy as np
import utils as Utils
species = "yeast"
GO = "CC"

# checking what children relation matrix encodes. 
# not very dense/sparse matrix
relation_matrix = Utils.load_pickle(f"data/goa/{species}/studied_GO_terms_relation_matrix/{GO}_children.pkl")
n_rows = relation_matrix.shape[0]
for i in range(0, 50):
    if 0 in relation_matrix[i]:
        print(relation_matrix[i])
        print(i)
print(np.where(relation_matrix==0)[0].shape)


# checking what parents relation matrix encodes
# very sparse matrix 
relation_matrix = Utils.load_pickle(f"data/goa/{species}/studied_GO_terms_relation_matrix/{GO}_parents.pkl")
n_rows = relation_matrix.shape[0]
for i in range(0, 20):
    print(relation_matrix[i])


# checking what ancestors relation matrix encodes. 
# not very dense/sparse matrix
relation_matrix = Utils.load_pickle(f"data/goa/{species}/studied_GO_terms_relation_matrix/{GO}_ancestors.pkl")
n_rows = relation_matrix.shape[0]
for i in range(200, 240):
    if 0 in relation_matrix[i]:
        print(relation_matrix[i])