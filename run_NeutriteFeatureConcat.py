from src.data.load_graph import MorphologyDatasetManager
from morphoclass.data import MorphologyDataset
from morphoclass.data.morphology_data import MorphologyData


import argparse

parser = argparse.ArgumentParser(description="Concatenate features from multiple neurite types.")
parser.add_argument("-axon", type=str, help="Directory containing axon features.", required=False)
parser.add_argument("-apical", type=str, help="Directory containing apical dendrite features.", required=False)
parser.add_argument("-basal", type=str, help="Directory containing basal dendrite features.", required=False)
parser.add_argument("-out", type=str, help="Output directory for concatenated features.", required=True)

args = parser.parse_args()

AXON_DIR = args.axon
APICAL_DIR = args.apical
BASAL_DIR = args.basal
NEUTRITE_CONCAT_DIR = args.out

AXON_DIR = "ouput/features/axon/"
APICAL_DIR = "ouput/features/apical_dendrite/"
BASAL_DIR = "ouput/features/apical_dendrite/"

NEUTRITE_CONCAT_DIR = "ouput/features/concat_neurites/"

class MultiNeutriteAttribute:
    def __init__(self, attr_axon=None, attr_apical=None, attr_basal=None):
        self.axon = attr_axon
        self.apical = attr_apical
        self.basal = attr_basal

def create_neurite_concat_dataset(path_axon, path_apical, path_basal,path_out):
        # Load the datasets
    
    dataset_per_neurite = {}
    if path_axon:
        dataset_per_neurite["axon"] = MorphologyDatasetManager.from_features_dir(path_axon).dataset
    if path_apical:
        dataset_per_neurite["apical"] = MorphologyDatasetManager.from_features_dir(path_apical).dataset
    if path_basal:
        dataset_per_neurite["basal"] = MorphologyDatasetManager.from_features_dir(path_basal).dataset
    if len(datalistofdict_per_neurite.keys()) < 2:
        raise ValueError("At least two neurite types must be provided for concatenation.")


    datalistofdict_per_neurite = {}
    for key, dataset in dataset_per_neurite.items():
        datalistofdict_per_neurite[key] = [data.__dict__ for data in dataset]
    

    OUT_data_list_of_dict = []
    for tuple in zip(datalistofdict_per_neurite.values()):
        if len(tuple) == 2:
            A, B = tuple
            assert A["path"] == B["path"]
            OUT_dict = {}
            for key in A.keys():
                if A[key] == B[key]:
                    OUT_dict[key] = A[key]
                else:
                    OUT_dict[key] = MultiNeutriteAttribute(A[key], B[key])
        elif len(tuple) == 3:
            A, B, C = tuple
            for key in C.keys():
                if A[key] == B[key] == C[key]:
                    OUT_dict[key] = A[key]
                else:
                    OUT_dict[key] = MultiNeutriteAttribute(A[key], B[key], C[key])
        OUT_data_list_of_dict.append(OUT_dict)
    data_list_of_morphologyData = [MorphologyData.from_dict(OUT_data_list_of_dict[i]) for i  in range(len(OUT_data_list_of_dict))]
    dataset_concat = MorphologyDataset(data_list_of_morphologyData)
    MorphologyDatasetManager.from_dataset(dataset_concat).write_features(path_out)

create_neurite_concat_dataset(AXON_DIR, APICAL_DIR, BASAL_DIR, NEUTRITE_CONCAT_DIR)