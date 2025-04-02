import numpy as np
import tmd
import os


def load_data(datapath= "../../Reconstructed/",
              types= list or str, # list of neuron types to load or string with Layer name or layer type  
                # e.g. "L2_IPC", "L2_TPC:A", "L5_UTPC", "L5_STPC", "L5_TTPC1", "L5_TTPC2"
              neurite_type= "apical_dendrite", # basal_dendrite, axons, dendrites = combo of basal an apical
              flatten = True, # flatten the images
              verbose = True # print the loading process

):
    """
    Load data for the given neuron types and neurite type.
    """
    if type(types) == str:
        types = [x  for x in os.listdir(datapath) if types + "_" in x]

    if verbose:
        print("Loading data from: ", types)
        
    
    groups = [tmd.io.load_population(os.path.join(datapath, type ), use_morphio=True) for type in types]
    labels = [i + 1 for i, k in enumerate(groups) for j in k.neurons]
    pers_diagrams = [
        tmd.methods.get_ph_neuron(j, neurite_type=neurite_type)
        for i, k in enumerate(groups)
        for j in k.neurons
        if (lambda x: True if x else False)(tmd.methods.get_ph_neuron(j, neurite_type=neurite_type))
    ]
    xlim, ylim = tmd.vectorizations.get_limits(pers_diagrams)
    pers_images = [
        tmd.vectorizations.persistence_image_data(p, xlim=xlim, ylim=ylim) for p in pers_diagrams
    ]

    if flatten:
        pers_images = [i.flatten() for i in pers_images]
    return labels, pers_images , pers_diagrams


def calc_pw_dist_mat(pers_images,distance): 
    """"
    calculate pairwise distance matrix for the given persistence images
    :param pers_images: persistence images
    :param distance: distance function to use (e.g. sw, wasserstein, etc.)
    :return: pairwise distance matrix
    """
    raise NotImplementedError
