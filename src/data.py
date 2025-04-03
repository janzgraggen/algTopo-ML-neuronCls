import numpy as np
import tmd
import os
from typing import Union, List, Literal

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../tda_toolbox')))
from diagram import bottleneck_distance, wasserstein_distance, sliced_wasserstein_distance  # M for sliced, k, m for landscape
from diagram import landscape
from diagram import gaussian_image

def pd_distance_wrapper(
    distance: Literal['wasserstein', 'bottleneck', 'sliced_wasserstein'] = "wasserstein",
    M: int = None, # for sliced wasserstein
): 
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
    if distance == "wasserstein":
        matrix = np.zeros((len(pers_diagrams), len(pers_diagrams)))
        for i in range(len(pers_diagrams)):
            for j in range(i, len(pers_diagrams)):  # Start from i to leverage symmetry
                matrix[i, j] = pd_distance_wrapper(distance=distance, M=M)(pers_diagrams[i], pers_diagrams[j])
                if i != j:  # Use symmetry to fill the other half
                    matrix[j, i] = matrix[i, j]
    return matrix

def vectorize_persistence_diagrams(
    pers_diagrams, 
    vectorization: Literal['persistence_image', 'wasserstein', 'bottlenek', 'slice_wasserstein','landscape'] =  "persistence_image",
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
    vectorized : list
        A list of vectorized representations of the input persistence diagrams.
        if multiple vectiorizatons are given: reurn a dict with keys as vectorization methods and values as the vectorized data.
    """
    # based on kernel density estimation (gaussian) ––––––––––––––––
    if vectorization == "persistence_image":
        xlim, ylim = tmd.vectorizations.get_limits(pers_diagrams)
        vectorized = [
            tmd.vectorizations.persistence_image_data(pd, xlim=xlim, ylim=ylim) for pd in pers_diagrams
        ]
        if flatten:
            vectorized = [i.flatten() for i in vectorized]

    ## based on function approximation –––––––––––––––––––––––––––––––
    elif vectorization == "landscape":
        vectorized = [
            landscape(pd, k=k, m=m) for pd in pers_diagrams # k and m for landscape
        ]

    ## based on pairwise distance matrix ––––––––––––––––––
    else:
        pw_dist_mat = compute_pairwise_distance_matrix(
            pers_diagrams, distance=vectorization, M=M # M for sliced wasserstein
        )
        vectorized = [
            pw_dist_mat[:, j] for j in range(len(pers_diagrams))
        ]

    return vectorized

VECTORIZATION_TYPES = Literal['persistence_image', 'wasserstein', 'bottleneck', 'slice_wasserstein', 'landscape']

def load_data(
    datapath= "../../Reconstructed/",
    types= list or str, # list of neuron types to load or string with Layer name or layer type  
    # e.g. "L2_IPC", "L2_TPC:A", "L5_UTPC", "L5_STPC", "L5_TTPC1", "L5_TTPC2"
    neurite_type= "apical_dendrite", # basal_dendrite, axons, dendrites = combo of basal an apical
    pers_hom_function = "radial_distances", # radial_distances or path or else. 
    vectorization:  Union[VECTORIZATION_TYPES, List[VECTORIZATION_TYPES]] = "persistence_image", # persistence_image or persistence_diagram
    M: int = None, # for sliced wasserstein
    k: int = None, # for landscape
    m: int = None,  # for landscape
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
    types : list or str
        List of neuron types to load or a string specifying a layer name or type 
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
    -------
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
    if type(types) == str:
        types = [x  for x in os.listdir(datapath) if types + "_" in x]

    if verbose:
        print("Loading data from: ", types)
        
    
    groups = [tmd.io.load_population(os.path.join(datapath, type ), use_morphio=True) for type in types]
    labels = [i + 1 for i, k in enumerate(groups) for j in k.neurons]
    pers_diagrams = [
        tmd.methods.get_ph_neuron(j,feature=pers_hom_function, neurite_type=neurite_type)
        for i, k in enumerate(groups)
        for j in k.neurons
        if (lambda x: True if x else False)(tmd.methods.get_ph_neuron(j,feature=pers_hom_function, neurite_type=neurite_type))
    ]

    ## handle multi vectorization
    if type(vectorization) ==list: 
        vectorized_pds = {}
        for i in vectorization:
            vectorized_pds[i] = vectorize_persistence_diagrams(
                pers_diagrams, 
                vectorization=i, 
                flatten=flatten,
                M=M, k=k, m=m
            )
    ## handle single vectorization
    else:
        vectorized_pds = vectorize_persistence_diagrams(
            pers_diagrams, 
            vectorization=vectorization, 
            flatten=flatten,
            M=M, k=k, m=m
        )

    
    return (labels, vectorized_pds, pers_diagrams) if return_pds else (labels, vectorized_pds)


