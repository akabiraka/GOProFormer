import sys
sys.path.append("../GOProFormer")

from Bio import SeqIO
import utils as Utils

species = "yeast" # yeast

inp_file_path = f"data/downloads/{species}.fasta"
out_file_path = f"data/uniprotkb/{species}.pkl"

species_uniprot_dict = {}
for record in SeqIO.parse(inp_file_path, "fasta"):
    print(record.id.split("|")[1])
    species_uniprot_dict[record.id.split("|")[1]] = {"id": record.id, "name": record.name, "description": record.description, "seq": str(record.seq)}
    # break
    


Utils.save_as_pickle(species_uniprot_dict, out_file_path)
species_uniprot_dict = Utils.load_pickle(out_file_path)

print(f"num of seq: {len(species_uniprot_dict)}")


# sample usage
# if "A0A024RBG1" in species_uniprot_dict:
#     print(species_uniprot_dict.get("A0A024RBG1"))

# example format to use the dictionary
# {
#     "uniprotkb_id1": {"id": "--", "name": "--", "description": "--", "seq": "--"},
#     "uniprotkb_id2": {"id": "--", "name": "--", "description": "--", "seq": "--"}
#     ...
#     ...
#     "uniprotkb_id3": {"id": "--", "name": "--", "description": "--", "seq": "--"}
# }