""" 
INFO:

VARIANT A: via custom class: 
    This script implements multiple neurite feature concatenation.

    it stores the feature that differ between the neurite types in a MultiNeutriteAttribute object.

    the object that previouly was accessible at 
        x 
    is tow accessible at:
        x.axon, x.apical, x.basal


    TODO: - Add support for reading the class with torch.load 
    when using morphoclass.data.MorphologyDatasetManager.from_features_dir

VARIANT B: just adding new attributes to the existing MorphologyData object: 
    x_axon
    x_apical
    x_basal
    -
    u_axon
    u_apical
    u_basal
    ...

    which is less elegant but more compatible with the existing codebase.
"""

from src.data.load_graph import MorphologyDatasetManager
from src.utils.multi_neutrite_attribute import MultiNeutriteAttribute
from morphoclass.data import MorphologyDataset
from morphoclass.data.morphology_data import MorphologyData
import numpy as np
import time
import torch

import argparse

MULTI_ATTRIBUTES = ['x','u', 'edge_index', 'edge_attr',  'bottleneck', 'landscape', 'morphometrics', 'persistence_image', 'sliced_wasserstein', 'tmd_neurites', 'wasserstein']
SINGLE_ATTRIBUTES = ['y', 'pos', 'normal', 'face', 'label', 'path', 'y_str']

def parser():
    print(f"NEX: ({time.strftime('%H:%M:%S')}) Parse arguments")
    parser = argparse.ArgumentParser(description="Concatenate features from multiple neurite types.")
    parser.add_argument("-axon", type=str, help="Directory containing axon features.", required=False)
    parser.add_argument("-apical", type=str, help="Directory containing apical dendrite features.", required=False)
    parser.add_argument("-basal", type=str, help="Directory containing basal dendrite features.", required=False)
    parser.add_argument("-out", type=str, help="Output directory for concatenated features.", required=True)

    args = parser.parse_args()
    if args.axon: AXON_DIR = "output/features/" + args.axon
    else: AXON_DIR = None
    if args.apical: APICAL_DIR = "output/features/" + args.apical
    else: APICAL_DIR = None
    if args.basal: BASAL_DIR = "output/features/" + args.basal
    else: BASAL_DIR = None
    NEUTRITE_CONCAT_DIR = "output/features/" + args.out
    print(f"NEX: ({time.strftime('%H:%M:%S')}) Done.")
    return AXON_DIR, APICAL_DIR, BASAL_DIR, NEUTRITE_CONCAT_DIR

def create_neurite_concat_dataset_A(path_axon, path_apical, path_basal,path_out):
        # Load the datasets
    print(f"NEX: ({time.strftime('%H:%M:%S')}) start loading datasets of different neurite types")
    dataset_per_neurite = {}
    if path_axon:
        dataset_per_neurite["axon"] = MorphologyDatasetManager.from_features_dir(path_axon).dataset
    if path_apical:
        dataset_per_neurite["apical"] = MorphologyDatasetManager.from_features_dir(path_apical).dataset
    if path_basal:
        dataset_per_neurite["basal"] = MorphologyDatasetManager.from_features_dir(path_basal).dataset
    if len(dataset_per_neurite.keys()) < 2:
        raise ValueError("At least two neurite types must be provided for concatenation.")
    datalistofdict_per_neurite = {}
    for key, dataset in dataset_per_neurite.items():
        datalistofdict_per_neurite[key] = [data.__dict__ for data in dataset]
    print(f"NEX: ({time.strftime('%H:%M:%S')}) done.")

    print(f"NEX: ({time.strftime('%H:%M:%S')}) Start concatenating features ")
    neutrites = list(datalistofdict_per_neurite.keys())
    data =list(datalistofdict_per_neurite.values())
    OUT_data_list_of_dict = []
    #case 1: 
    if len(neutrites) == 2:
        print("2 neurites")
        for A,B in zip(data[0],data[1]):
            OUT_dict = {}
            for key in A.keys():
                    if key in SINGLE_ATTRIBUTES:
                        assert A[key] == B[key]
                        OUT_dict[key] = A[key]
                    else:
                        multiatr= MultiNeutriteAttribute()
                        setattr(multiatr,neutrites[0], A[key])
                        setattr(multiatr,neutrites[1], B[key])
                        OUT_dict[key] = multiatr
            OUT_data_list_of_dict.append(OUT_dict)
    #
    if len(neutrites) == 3:
        i = 0
        for A,B,C in zip(data[0],data[1],data[2]):
            i += 1
            OUT_dict = {}
            for key in A.keys():
                if key in SINGLE_ATTRIBUTES:
                    assert A[key] == B[key] == C[key]
                    OUT_dict[key] = A[key]
                else:
                    multiatr= MultiNeutriteAttribute()
                    setattr(multiatr,neutrites[0], A[key])
                    setattr(multiatr,neutrites[1], B[key])
                    setattr(multiatr,neutrites[2], C[key])
                    OUT_dict[key] = multiatr
            OUT_data_list_of_dict.append(OUT_dict)
    else:
        raise ValueError("Only 2 or 3 neurite types are supported for concatenation.")
    print(f"NEX: ({time.strftime('%H:%M:%S')}) Done. ")
    print(f"NEX: ({time.strftime('%H:%M:%S')}) Save concatenated features")
    data_list_of_morphologyData = [MorphologyData.from_dict(OUT_data_list_of_dict[i]) for i  in range(len(OUT_data_list_of_dict))]
    dataset_concat = MorphologyDataset(data_list_of_morphologyData)
    MorphologyDatasetManager.from_dataset(dataset_concat).write_features(path_out,True)
    print(f"NEX: ({time.strftime('%H:%M:%S')}) Done. ")

def create_neurite_concat_dataset_B(path_axon, path_apical, path_basal,path_out):
        # Load the datasets
    print(f"NEX: ({time.strftime('%H:%M:%S')}) start loading datasets of different neurite types")
    dataset_per_neurite = {}
    if path_axon:
        dataset_per_neurite["axon"] = MorphologyDatasetManager.from_features_dir(path_axon).dataset
    if path_apical:
        dataset_per_neurite["apical"] = MorphologyDatasetManager.from_features_dir(path_apical).dataset
    if path_basal:
        dataset_per_neurite["basal"] = MorphologyDatasetManager.from_features_dir(path_basal).dataset
    if len(dataset_per_neurite.keys()) < 2:
        raise ValueError("At least two neurite types must be provided for concatenation.")
    datalistofdict_per_neurite = {}
    for key, dataset in dataset_per_neurite.items():
        datalistofdict_per_neurite[key] = [data.__dict__ for data in dataset]
    print(f"NEX: ({time.strftime('%H:%M:%S')}) done.")

    print(f"NEX: ({time.strftime('%H:%M:%S')}) Start concatenating features ")
    neutrites = list(datalistofdict_per_neurite.keys())
    data =list(datalistofdict_per_neurite.values())
    OUT_data_list_of_dict = []
    #case 1: 
    if len(neutrites) == 2:
        print("2 neurites")
        for A,B in zip(data[0],data[1]):
            OUT_dict = {}
            for key in A.keys():
                    if key in SINGLE_ATTRIBUTES:
                        assert A[key] == B[key]
                        OUT_dict[key] = A[key]
                    else:
                        ## CHANGE WRT VARIANT A
                        OUT_dict[key + "_" + neutrites[0]] = A[key]
                        OUT_dict[key + "_" + neutrites[1]] = B[key]
            OUT_data_list_of_dict.append(OUT_dict)
    #
    elif len(neutrites) == 3:
        i = 0
        for A,B,C in zip(data[0],data[1],data[2]):
            i += 1
            OUT_dict = {}
            for key in A.keys():
                if key in SINGLE_ATTRIBUTES:
                    assert A[key] == B[key] == C[key]
                    OUT_dict[key] = A[key]
                else:
                    ## CHANGE WRT VARIANT A
                    OUT_dict[key + "_" + neutrites[0]] = A[key]
                    OUT_dict[key + "_" + neutrites[1]] = B[key]
                    OUT_dict[key + "_" + neutrites[2]] = C[key]
            OUT_data_list_of_dict.append(OUT_dict)
    else:
        raise ValueError("Only 2 or 3 neurite types are supported for concatenation.")
    print(f"NEX: ({time.strftime('%H:%M:%S')}) Done. ")
    print(f"NEX: ({time.strftime('%H:%M:%S')}) Save concatenated features")
    data_list_of_morphologyData = [MorphologyData.from_dict(OUT_data_list_of_dict[i]) for i  in range(len(OUT_data_list_of_dict))]
    dataset_concat = MorphologyDataset(data_list_of_morphologyData)
    MorphologyDatasetManager.from_dataset(dataset_concat).write_features(path_out,True)
    print(f"NEX: ({time.strftime('%H:%M:%S')}) Done. ")

AXON_DIR, APICAL_DIR, BASAL_DIR, NEUTRITE_CONCAT_DIR = parser()
create_neurite_concat_dataset_B(AXON_DIR, APICAL_DIR, BASAL_DIR, NEUTRITE_CONCAT_DIR)

# # # Test Variant A
# # # Load the concatenated dataset
# dataset_concat = MorphologyDatasetManager.from_features_dir(NEUTRITE_CONCAT_DIR).dataset
# print(dataset_concat[0].__dict__.keys())
# print(type(dataset_concat[0].x))
# print(dataset_concat[0].x.axon)
# print(dataset_concat[0].x.apical)


# # Test Variant A
# # Load the concatenated dataset
dataset_concat = MorphologyDatasetManager.from_features_dir(NEUTRITE_CONCAT_DIR).dataset
print(dataset_concat[0].__dict__.keys())
print(type(dataset_concat[0].x_axon))
print(dataset_concat[0].x_axon)
print(dataset_concat[0].x)