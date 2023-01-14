import sys
sys.path.append("../GOProFormer")
import esm
import utils as Utils
import torch

species = "yeast"
max_seq_len = 512



esm1b, alphabet = esm.pretrained.esm1_t12_85M_UR50S()
esm1b_batch_converter = alphabet.get_batch_converter()

seq_db_dict = Utils.load_pickle(f"data/uniprotkb/{species}.pkl")


for i, (uniprot_id, all_info) in enumerate(seq_db_dict.items()):
    seq = all_info["seq"]
    uniprotid_seq = [(uniprot_id, seq)]
    uniprotid, batch_strs, seq_tokens = esm1b_batch_converter(uniprotid_seq)


    with torch.no_grad():
        results = esm1b(seq_tokens, repr_layers=[12], return_contacts=False)
    seq_rep = results["representations"][12] #1, max_seq_len, esmb_embed_dim
    seq_rep.squeeze_(0)
    seq_rep = seq_rep[1 : len(seq) + 1].mean(0) # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    

    Utils.save_as_pickle(seq_rep, f"data/uniprotkb/{species}_sequences_rep_mean/{uniprot_id}.pkl")
    # seq_rep = Utils.load_pickle(f"data/uniprotkb/{species}_sequences_rep_mean/{uniprot_id}.pkl")

    print(i, uniprot_id, seq_rep.shape)
    # break