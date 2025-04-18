from  morphoclass.data import MorphologyDataset
from morphoclass import transforms, vis , training
import os
from typing import List, Union
import matplotlib.pyplot as plt
import numpy as np
from morphoclass.data.filters import inclusion_filter, attribute_check
import pathlib



## inclusiton filters according to morpho class filter implementation to handle the same data loading as in load tdm
  
## handling in tdm_loading: 
    # # create list of all cell types that we want to load:
    # if type(types) == str:
    #     types = [x  for x in os.listdir(datapath) if types + "_" in x]


TMD_TYPES = {
    "apical": "apical_dendrite",
    "axon": "axon",
    "basal": "basal_dendrite",
    "neurites": "neurites",
    "all": "neurites",
}

TMD_TYPES_INV = {v: k for k, v in TMD_TYPES.items()}

@inclusion_filter.register(str)
def mtype_inclusion_filter(mtype_substring):
    """Inclusion filter based on the m-type of sample.

    It is assumed that the morphology files are organised
    in folders, each folder representing an m-type. If
    the `mtype_substring` is part of the m-type folder
    name of the given `data` instance, then this instance
    is included.
    """
    print(f"CALLED: mtype_inclusion_filter(mtype_substring)  ||| mtype_substring: {mtype_substring}")

    
    @attribute_check("path", default_return=False)
    def filter_implementation(data):
        # Check if the directory name (m-type) contains the substring
        print(f"CALLED: filter_implementation(data) ||| data.path: {data.path}")
        if mtype_substring + "_" in pathlib.Path(data.path).parent.name.upper():
            return True
        return False

    return filter_implementation

@inclusion_filter.register(list)
def mtype_inclusion_filter(mtype_strings: List[str]):
    """Inclusion filter based on the m-type of sample.

    It is assumed that the morphology files are organised
    in folders, each folder representing an m-type. If
    the m-type folder name of the given `data` instance
    is in `mtype_strings` ,then this instance
    is included.
    """
    @attribute_check("path", default_return=False)
    def filter_implementation(data):
        # Check if the directory name (m-type) contains the substring
        if pathlib.Path(data.path).parent.name.upper() in mtype_strings:
            return True
        return False

    return filter_implementation


def load_graph(
    datapath , # "assets/"
    layer, # "L3"
    types= list or str, 
    # list of neuron types to load or string with Layer name or layer type  
    # e.g ”L2" or "L3" or "L4" or "L5"  //  OR:
    # e.g. ["L2_IPC", "L2_TPC:A", "L5_UTPC", "L5_STPC", "L5_TTPC1", "L5_TTPC2"]
    neurite_type= "apical_dendrite",
    verbose = True, # print the loading process
    ):

    if verbose:
        print("setting filters")

    if verbose:
        print("setting pretrasforms")
    pre_transform = transforms.Compose([
        transforms.ExtractTMDNeurites(neurite_type=TMD_TYPES_INV[neurite_type]),
        transforms.ExtractEdgeIndex(),      
    ])

    if verbose:
        print("Loading dataset")
    dataset = MorphologyDataset.from_structured_dir(
        data_path=datapath,
        #layer could be used alternatively to overriding filters , however filters is more flexible (e.g. for pairwise extration in same layer)
        layer = layer,
        pre_transform = pre_transform,
        pre_filter= inclusion_filter(types) if types else None,
    )


    print("Neuron types found:")
    for sample in dataset:
        print(f"Path: {sample.path}")
    return dataset

"""
Unlike in docs_ read form structured dir expects path -> Layer -> Label -> Files structure
"""
# Path to morphology file (make sure this exists)
PATH = "assets" # or "assets/L2_TPC:A" or "assets/L5_UTPC"
LAYER = "L2" # or "L3" or "L4" or "L5"
TYPES = "" #["L2_IPC", "L2_TPC:A"] or "L2" or other.
NEUTRITE_TYPE = "apical_dendrite" # "basal_dendrite", "axon", "neurites", "all"


dataset = load_graph(
    datapath= PATH,
    layer= LAYER,
    types= TYPES,
    neurite_type= NEUTRITE_TYPE,
    )

if len(dataset) == 0:
    raise ValueError("Dataset is empty. Please check the data path and neuron types.")



# Visualize the dataset
print("Dataset Information:")
print(f"Number of samples: {len(dataset)}")
print(f"Features available: {dataset.features if hasattr(dataset, 'features') else 'No features loaded'}")



# Check the dimensions of the first sample
if len(dataset) > 0:
    ##example plot
    fig, axs = plt.subplots(1, 3, figsize=(12, 7))
    for i, ax in enumerate(axs):
        ax.set_aspect("equal")
        ax.set_title(f"#{i}")
        vis.plot_tree(dataset[0].tmd_neurites[0], ax, node_size=1)
    fig.show()

    #print stats proposed from GPT
    sample = dataset[0]
    print("\nFirst Sample Information:")
    print(f"Path: {sample.path}")
    print(f"Number of neurites: {len(sample.tmd_neurites)}")
    print(f"Graph edge index shape: {sample.edge_index.shape if hasattr(sample, 'edge_index') else 'No edge index'}")
    print(f"Graph features shape: {sample.x if hasattr(sample, 'x') else 'No features'}")

# Visualize the first neurite of the first sample
if len(sample.tmd_neurites) > 0:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect("equal")
    ax.set_title("Visualization of the First Neurite")
    vis.plot_tree(sample.tmd_neurites[0], ax, node_size=1)
    plt.show()
else:
    print("No neurites available for visualization.")


# graph = MorphologyDataset.from_paths(data_path + "L2_TPC:A", "L5_TPC_A")
# fig, axs = plt.subplots(1, 3, figsize=(12, 7))
# for i, ax in enumerate(axs):
#     print(f"Neuron {i}")
#     ax.set_aspect("equal")
#     ax.set_title(f"#{i}")
#     vis.plot_tree(dataset[0].tmd_neurites[0], ax, node_size=1)
# fig.show()


# def load_data(
#     datapath= "../../assets/",
#     types= list or str, # list of neuron types to load or string with Layer name or layer type  
#     # e.g. "L2_IPC", "L2_TPC:A", "L5_UTPC", "L5_STPC", "L5_TTPC1", "L5_TTPC2"
#     neurite_type= "apical_dendrite", # basal_dendrite, axons, dendrites = combo of basal an apical
#     verbose = True, # print the loading process 
# ):
    
#     """
#         ...
#     Loads and processes neuron morphology data for machine learning tasks.
#     Parameters:
#     ----------
#     datapath : str, optional
#         Path to the directory containing the reconstructed neuron data. 
#         Default is "../../Reconstructed/".
#     types : list or str
#         List of neuron types to load or a string specifying a layer name or type 
#         (e.g., "L2_IPC", "L5_TTPC1").
#     neurite_type : str, optional
#         Type of neurite to analyze. Options include "apical_dendrite", 
#         "basal_dendrite", "axons", or "dendrites" (combination of basal and apical). 
#         Default is "apical_dendrite".

#     Returns:
    
#     tuple:
#             (labels, vectorized_pds)
#             - labels: List of integer labels corresponding to neuron groups.
#             - vectorized_pds: Vectorized persistence diagrams based on the specified 
#               vectorization method(s).
#               if multiple vectorizations are given, return a dict with keys as vectorization methods and values as the vectorized data.
#     """
#     if type(types) == str:
#         types = [x  for x in os.listdir(datapath) if types + "_" in x]

#     if verbose:
#         print("Loading data from: ", types)
#         for neuron_type in types:
#             print(f"Processing neuron type: {neuron_type}")
    
#     def get_graph(dataset: MorphologyDataset): 
#         dataset.get_neurons()
#         return dataset.neurons

#     groups = [MorphologyDataset.from_structured_dir(data_path=data_path,layer= type + "_") for type in types]
#     labels = [i + 1 for i, k in enumerate(groups) for j in k.len()]
#     graphs = [
#         get_graph(j, neurite_type=neurite_type)
#         for i, k in enumerate(groups)
#         for j in k.neurons
#         if (lambda x: True if x else False)(tmd.methods.get_ph_neuron(j,feature=pers_hom_function, neurite_type=neurite_type))
#     ]

#     graphs = False
#     return  (labels, graphs)


def feature_extractor(dataset: MorphologyDataset, feature: str):

    feature_extractor = transforms.Compose(
        transforms.ExtractRadialDistances(), 
        transforms.ExtractBranchingAngles(),
    )
    
    transform, fitted_scaler = training.make_transform(
        dataset=dataset,
        feature_extractor=feature_extractor,
        n_features=1,
        fitted_scaler=fitted_scaler,
    )

    dataset.transform = transform
    return dataset, fitted_scaler
