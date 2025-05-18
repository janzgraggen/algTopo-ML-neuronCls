import numpy as np
import tmd
import os
import time
from typing import Union, List, Literal

from src.tda_toolbox.diagram import bottleneck_distance, wasserstein_distance, sliced_wasserstein_distance  # M for sliced, k, m for landscape
from src.tda_toolbox.diagram import landscape
from src.tda_toolbox.diagram import gaussian_image

VECTORIZATION_TYPES = Literal['persistence_image', 'wasserstein', 'bottleneck', 'sliced_wasserstein', 'landscape']

def pd_distance_wrapper(
    distance: Literal['wasserstein', 'bottleneck', 'sliced_wasserstein'] = "wasserstein",
    M: int = None, # for sliced wasserstein
): 
    """
    Create a labda function for pariwise distance calculation according to the given distance metric.
    """
    if distance == "wasserstein":
        return lambda x, y: wasserstein_distance(x, y)
    elif distance == "bottleneck":
        return lambda x, y: bottleneck_distance(x, y)
    elif distance == "sliced_wasserstein":
        return lambda x, y: sliced_wasserstein_distance(x, y, M=M)

def compute_pairwise_distance_matrix(
        pers_diagrams,
        distance: Literal['wasserstein', 'bottleneck', 'sliced_wasserstein'] = "wasserstein", 
        M: int = None, # for sliced wasserstein
):
    """"
    calculate pairwise distance matrix for a given persistence diagrams (for 1 diag)
    :param pers_diagram: persistence diagram
    :param distance: distance function to use (e.g. sw, wasserstein, etc.)
    :return: pairwise distance matrix
    """
    ## PAIRWISE DISTANCE MATRIX
    
    print(f'    Computing PW-dist matrix with d: {distance}')

    matrix = np.zeros((len(pers_diagrams), len(pers_diagrams)))
    for i in range(len(pers_diagrams)):
        for j in range(i, len(pers_diagrams)):  # Start from i to leverage symmetry
            matrix[i, j] = pd_distance_wrapper(distance=distance, M=M)(pers_diagrams[i], pers_diagrams[j])
            if i != j:  # Use symmetry to fill the other half
                matrix[j, i] = matrix[i, j]
    return matrix

def vectorize_persistence_diagrams(
    pers_diagrams, 
    vectorization: Union[Literal[VECTORIZATION_TYPES], List[Literal[VECTORIZATION_TYPES]]] = "persistence_image",
    M: int = None, # for sliced wasserstein
    k: int = None, # for landscape
    m: int = None,  # for landscape
    flatten: bool = True # for gaussian image
    ): 
    """
    Vectorizes a list of persistence diagrams using various methods.
    The methods are kategorized into groups:
        1. Based on kernel density estimation (Gaussian) : persistence_image
        2. Based on function approximation : landscape
        3. Based on pairwise distance matrix : wasserstein, bottlenek, slice_wasserstein
    
    Parameters:
    -----------
    pers_diagrams : list
        A list of persistence diagrams [a pd is a list of tuples: (birth, death) ] to be vectorized.
    vectorization : Literal['persistence_image', 'wasserstein', 'bottlenek', 'slice_wasserstein', 'landscape'], optional
        The method used for vectorization. Default is "persistence_image".
        - "persistence_image": Uses kernel density estimation (Gaussian) to create persistence images.
        - "landscape": Computes persistence landscapes.
        - "wasserstein", "bottlenek", "slice_wasserstein": Computes pairwise distance matrices based on the specified distance metric.
    M : int, optional
        Number of slices for "slice_wasserstein". Default is None.
    k : int, optional
        Number of landscapes for "landscape". Default is None.
    m : int, optional
        Resolution for "landscape". Default is None.
    
    Returns:
    --------
    vectorized : dict
        ->reurn a dict with keys as vectorization methods and values as the vectorized data.
         keys: 
            "vectorization" (e.g. "persistence_image", "wasserstein", etc.)
         values: 
             A list of vectorized representations of the input persistence diagrams.
    """
    #make iterable
    if type(vectorization) ==str:
        vectorization = [vectorization]

    #the landscape vectorization needs k and m and has no error handling
    if "landscape" in vectorization:
        if (k is None) or (m is None):
            raise ValueError("k and m : int >0, must be given for landscape vectorization")
        
    if "sliced_wasserstein" in vectorization:
        if (M is None):
            raise ValueError("M : int >0 ,must be given for sliced_wasserstein vectorization")
    

    vectorization_dict = {}
    for v_type in vectorization:
        # based on kernel density estimation (gaussian) ––––––––––––––––
        print(f" VEC: ({time.strftime('%H:%M:%S')}) start {v_type}-PD vectorization")
        if v_type == "persistence_image":
            xlim, ylim = tmd.analysis.get_limits(pers_diagrams)
            #-> other tdm version: xlim, ylim = tmd.vectorizations.get_limits(pers_diagrams)
            vectorized = [
                tmd.analysis.get_persistence_image_data(pd, xlim=xlim, ylim=ylim) for pd in pers_diagrams ## alt use gaussian_image
                #-> other tdm version: tmd.vectorizations.persistence_image_data(pd, xlim=xlim, ylim=ylim) for pd in pers_diagrams
            ]
            if flatten:
                vectorized = [i.flatten() for i in vectorized]
        ## based on function approximation –––––––––––––––––––––––––––––––
        elif v_type == "landscape":
            vectorized = [
                landscape(pd, k=k, m=m) for pd in pers_diagrams # k and m for landscape 
            ]

        ## based on pairwise distance matrix ––––––––––––––––––
        else:
            pw_dist_mat = compute_pairwise_distance_matrix(
                pers_diagrams, distance=v_type, M=M # M for sliced wasserstein
            )
            vectorized = [
                pw_dist_mat[:, j] for j in range(len(pers_diagrams))
            ]
        vectorization_dict[v_type] = np.array(vectorized)
        print(f" VEC: ({time.strftime('%H:%M:%S')}) done.")
    return vectorization_dict


def load_tmd(
    datapath= "assets",
    layer= "L2", # layer name
    types= [], # list of neuron types to load or string with type  
    # e.g. "L2_IPC", "L2_TPC:A", "L5_UTPC", "L5_STPC", "L5_TTPC1", "L5_TTPC2"
    neurite_type= "apical_dendrite", # basal_dendrite, axons, dendrites = combo of basal an apical
    pers_hom_function = "radial_distances", # radial_distances or path or else. 
    vectorization:  Union[VECTORIZATION_TYPES, List[VECTORIZATION_TYPES]] = "persistence_image", # persistence_image or persistence_diagram
    M: int = 0, # for sliced wasserstein
    k: int = 0, # for landscape
    m: int = 0,  # for landscape
    flatten: bool = True, # for gaussian image
    verbose = True, # print the loading process 
    return_pds = False # return persistence diagrams
):
    
    """
        ...
    Loads and processes neuron morphology data for machine learning tasks.
    Parameters:
    ----------
    datapath : str, optional
        Path to the directory containing the reconstructed neuron data. 
        Default is "../../Reconstructed/".
    layer : list or str
        Layer name to load data from. Default is "L2". from "L2", "L3", "L4", "L5" , "L6", "L23"
    types : list or str
        List of neuron types to load or a string specifying a type/types (substring to match)
        (e.g., "L2_IPC", "L5_TTPC1").
    neurite_type : str, optional
        Type of neurite to analyze. Options include "apical_dendrite", 
        "basal_dendrite", "axons", or "dendrites" (combination of basal and apical). 
        Default is "apical_dendrite".
    pers_hom_function : str, optional
        Function to compute persistent homology. Options include "radial_distances", 
        "path", or other supported methods. Default is "radial_distances".
    vectorization : Literal['persistence_image', 'wasserstein', 'bottlenek', 
        'slice_wasserstein', 'landscape'] or list of them for multiple, optional
        Method(s) for vectorizing persistence diagrams. Default is "persistence_image".
    M : int, optional
        Parameter for sliced Wasserstein vectorization. Default is None.
    k : int, optional
        Parameter for landscape vectorization. Default is None.
    m : int, optional
        Parameter for landscape vectorization. Default is None.
    flatten : bool, optional
        Whether to flatten the Gaussian persistence image. Default is True.
    verbose : bool, optional
        Whether to print the loading process. Default is True.
    return_pds : bool, optional
        Whether to return the raw persistence diagrams. Default is False.
    Returns:
    
    tuple
        If `return_pds` is False:
            (labels, vectorized_pds)
            - labels: List of integer labels corresponding to neuron groups.
            - vectorized_pds: Vectorized persistence diagrams based on the specified 
              vectorization method(s).
              if multiple vectorizations are given, return a dict with keys as vectorization methods and values as the vectorized data.
        If `return_pds` is True:
            (labels, vectorized_pds, pers_diagrams)
            - labels: List of integer labels corresponding to neuron groups.
            - vectorized_pds: Vectorized persistence diagrams based on the specified 
              vectorization method(s).
              if multiple vectorizations are given, return a dict with keys as vectorization methods and values as the vectorized data.
            - pers_diagrams: Raw persistence diagrams for the neurons.
    """
    #take all types of a layer if no type is given
    if not  types:
        types = [x for x in os.listdir(datapath + "/" +layer + "/")]
    if type(types) == str: 
        types = [types]
    if types[0][0] == "L" :
        types = [ x.split("_")[0] + "/" + x  for x in types]
    else: 
        types = [layer + "/" + x for x in types]


    if verbose:
        print("Loading data from: ", types)
        for neuron_type in types:
            print(f"Processing neuron type: {neuron_type}")
    
    groups = [tmd.io.load_population(os.path.join(datapath, type ), use_morphio=True) for type in types]
    labels = [i + 1 for i, k in enumerate(groups) for j in k.neurons]
    pers_diagrams = [
        tmd.methods.get_ph_neuron(j,feature=pers_hom_function, neurite_type=neurite_type)
        for i, k in enumerate(groups)
        for j in k.neurons
        if (lambda x: True if x else False)(tmd.methods.get_ph_neuron(j,feature=pers_hom_function, neurite_type=neurite_type))
    ]

    vectorized_pds = vectorize_persistence_diagrams(
        pers_diagrams, 
        vectorization, 
        flatten=flatten,
        M=M, k=k, m=m
        )

    return (labels, vectorized_pds, pers_diagrams) if return_pds else (labels, vectorized_pds)


