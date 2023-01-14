# GOProFormer
This repository corresponds to the article titled as "GOProFormer: A Multi-Modal Transformer Method for Gene Ontology Protein Function Prediction".


## Data pre-processing steps:
* `python data_preprocess/create_uniprot_species_dict.py`

* `python data_preprocess/goa_random_split_leakage.py`

* `python data_preprocess/goa_time_delay_no_knowledge.py`

* `python data_preprocess/goa_time_series_no_knowledge.py`

* To compute the sequence representation:
    * Install ESM-1b (particularly used `esm.pretrained.esm1_t12_85M_UR50S`)
    * `python data_preprocess/compute_seq_rep_using_esm1b.py`


## Model development
* `python models/train_val.py`
* `python models/eval_on_val_and_test.py`


## Downloadables
* Trained weights can be downloaded from [here](https://gmuedu-my.sharepoint.com/:f:/g/personal/akabir4_gmu_edu/Er-VOUPr8M5NsTR58Ykq_2UBrmfBws8VQJrC5-5pYobNDQ?e=pNK3nM). 
* A copy of dataset can be obtained from [here](https://gmuedu-my.sharepoint.com/:u:/g/personal/akabir4_gmu_edu/EQSAWpMgQc5Kmi-lqzBkgRkBca4kWuQMWfgfi6lIj_wuRQ?e=zzJZwH).


## Citation
If the model is found useful, we request to cite the relevant paper:
```bibtex
@Article{biom12111709,
    AUTHOR = {Kabir, Anowarul and Shehu, Amarda},
    TITLE = {GOProFormer: A Multi-Modal Transformer Method for Gene Ontology Protein Function Prediction},
    JOURNAL = {Biomolecules},
    VOLUME = {12},
    YEAR = {2022},
    NUMBER = {11},
    ARTICLE-NUMBER = {1709},
    URL = {https://www.mdpi.com/2218-273X/12/11/1709},
    PubMedID = {36421723},
    ISSN = {2218-273X},
    DOI = {10.3390/biom12111709}
}
```