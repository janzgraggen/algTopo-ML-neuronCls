from  morphoclass.data import MorphologyDataset
from morphoclass import transforms
from morphoclass.data.filters import inclusion_filter, attribute_check, combined_filter , has_apicals_filter

from typing import List
import matplotlib.pyplot as plt
import pathlib
from torch.utils.data import ConcatDataset


TMD_TYPES = {
    "apical": "apical_dendrite",
    "axon": "axon",
    "basal": "basal_dendrite",
    "neurites": "neurites",
    "all": "neurites",
}

TMD_TYPES_INV = {
    v: k for k, v in TMD_TYPES.items()
    }

def ntype_convert(typespec: str,inverse:bool = False, twoway: bool = False): 
    """
    Converts a neurite type specification to its corresponding value based on predefined mappings.
    by default it converts from MORPHIO to TMD_TYPES

    if inverse == True:
        Converts form TMD_TYPES to MORPHIO
    if toway == True:
        Converts from MORPHIO to TMD_TYPES and vice versa based on the input.
        This function checks if the provided `typespec` exists in either the `TMD_TYPES` or 
        `TMD_TYPES_INV` dictionaries. If found, it returns the corresponding value. If not, 
        it raises a `ValueError`.
    
    Args:
        typespec (str): The neurite type specification to be converted.
        inverse (bool): If True, converts from TMD_TYPES to MORPHIO.
        twoway (bool): If True, converts from MORPHIO to TMD_TYPES and vice versa.
    Returns:
        The corresponding value from `TMD_TYPES` or `TMD_TYPES_INV` based on the input.
    Raises:
        ValueError: If `typespec` is not found in either `TMD_TYPES` or `TMD_TYPES_INV`.
    Notes:
        - `TMD_TYPES` and `TMD_TYPES_INV` are expected to be predefined dictionaries 
          containing valid neurite type mappings.
        - The error message includes the valid keys from both dictionaries for reference.
    """
    # DEFAULT CONVERSION  TMD_TYPES -> MORPHIO 
    if (not inverse) and (not twoway):
        if typespec in TMD_TYPES.keys(): 
            return typespec
        else:
            return TMD_TYPES_INV[typespec]
        
    #  CONVERSION   MORPHIO  -> TMD_TYPES 
    if inverse and not twoway:
        if typespec in TMD_TYPES.values():
            return typespec
        else:
            return TMD_TYPES[typespec]

    #  CONVERSION   MORPHIO  <-> TMD_TYPES
    if twoway:
        if typespec in TMD_TYPES.keys():
            return TMD_TYPES[typespec]
        elif typespec in TMD_TYPES_INV.keys():
            return TMD_TYPES_INV[typespec]

@inclusion_filter.register(str)
def mtype_inclusion_filter(mtype_substring):
    """Inclusion filter based on the m-type of sample (FOLDER LEVEL).

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
    """Inclusion filter based on the m-type of sample (FOLDER LEVEL).

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
    layer= List[str] or str, # "L3" or "L2" or "L4" or "L5" or ["L2", "L3"]
    types= list or str, # INCLUSIN FILTERS   # e.g. ["L2_IPC", "L2_TPC:A", "L5_UTPC", "L5_STPC", "L5_TTPC1", "L5_TTPC2"]
    neurite_type= "apical_dendrite",
    preprocess_transform: transforms.Compose = 
        transforms.Compose([
            # augmenting: 
            transforms.BranchingOnlyNeurites(),
        ]),
    feature_extractor: transforms.Compose =
        transforms.Compose([
            # feature extraction. 
            transforms.ExtractPathDistances(),
            transforms.ExtractDiameters()
        ]),
    verbose = True, # print the loading process
    ):

    ## Defining filters according to the m_type_inclusion filter based on the types argument
    ## And the neurite type, fiter only apicals if neurite_type is "apical_dendrite"/"apocal"
    if verbose:
        print("setting filters")
    if types and ntype_convert(neurite_type) == "apical":
        filters = combined_filter(
            has_apicals_filter,
            inclusion_filter(types) 
            ) 
    elif types:
        filters= inclusion_filter(types) 
    elif ntype_convert(neurite_type) == "apical":
        filters = has_apicals_filter
    else:
        filters = None


    ## Defining the pre_transform according to the neurite type
    if verbose:
        print("setting pretrasforms")
    morphology_loader = transforms.Compose([
        transforms.ExtractTMDNeurites(neurite_type=ntype_convert(neurite_type)),
        preprocess_transform,
        feature_extractor,
        transforms.ExtractEdgeIndex(),      
    ])


    if verbose:
        print("Loading dataset")
    layer_datasets = [
        MorphologyDataset.from_structured_dir(
            data_path=datapath,
            #layer could be used alternatively to overriding filters , however filters is more flexible (e.g. for pairwise extration in same layer)
            layer = layer_entry,
            pre_transform = morphology_loader,
            pre_filter= filters,
            ) for layer_entry in (layer if isinstance(layer, list) else [layer])
        ] 
    dataset = ConcatDataset(layer_datasets)

    if verbose:
        print("Neuron types found:")
        for sample in dataset:
            print(f"Path: {sample.path}")

    return dataset
