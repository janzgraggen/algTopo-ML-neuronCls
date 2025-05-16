from morphoclass.data import MorphologyDataset
from morphoclass import transforms
from morphoclass.data.filters import inclusion_filter, attribute_check, combined_filter , has_apicals_filter
from morphoclass.data.morphology_data import MorphologyData
from morphoclass.console.cmd_extract_features import _find_bad_diagrams, _find_bad_neurites

from typing import List, Union, Literal
from torch.utils.data import ConcatDataset

from src.data.load_tdm  import VECTORIZATION_TYPES ,vectorize_persistence_diagrams
from tmd.Topology.methods import get_persistence_diagram
from neurom.apps import morph_stats

import torch
import pathlib
import numpy as np
from sklearn.preprocessing import RobustScaler
import yaml
import warnings

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

class MorphologyDatasetManager: 
    """ 
    Class for loading dataset and extending it with additional features. 
    or scaling the features.
    """
    def __init__(self, 
        datapath , # "assets/"
        layer= List[str] or str, # "L3" or "L2" or "L4" or "L5" or ["L2", "L3"]
        types= list or str, # INCLUSIN FILTERS   # e.g. ["L2_IPC", "L2_TPC:A", "L5_UTPC", "L5_STPC", "L5_TTPC1", "L5_TTPC2"]
        neurite_type= "apical_dendrite",
        simplify: bool = True,
        orient: bool = True,
        feature_extractor: transforms.Compose =
            transforms.Compose([
                transforms.ExtractRadialDistances()
            ]),
        normalize: bool = True, 
        ):
        self.types = types
        self.neurite_type = neurite_type
        self.orient = orient
        self.simplify = simplify

        filters = self._get_filters()
        ZERO_transform = self._get_ZERO_transform()
        
        morphology_loader = transforms.Compose([
            ZERO_transform,
            feature_extractor,
        ])

        layer_datasets = [
            MorphologyDataset.from_structured_dir(
                data_path=datapath,
                layer = layer_entry,
                pre_transform = morphology_loader,
                pre_filter= filters,
                ) for layer_entry in (layer if isinstance(layer, list) else [layer])
            ] 
        morpho_dataset = MorphologyDataset(ConcatDataset(layer_datasets))

        ## remove diags with less thatn 3 nodes
        idx_keep = [idx for idx in range(len(morpho_dataset)) if idx not in _find_bad_neurites(morpho_dataset)]
        morpho_dataset = morpho_dataset.index_select(idx_keep)
        
        self.dataset_ = morpho_dataset

        if normalize: 
            self.scale_graph()
        
    def _get_filters(self):
        """
        Defining filters according to the m_type_inclusion filter based on the types argument
        And the neurite type, fiter only apicals if neurite_type is "apical_dendrite"/"apical"
        """
        if self.types and ntype_convert(self.neurite_type) == "apical":
            filters = combined_filter(
                has_apicals_filter,
                inclusion_filter(self.types) 
                ) 
        elif self.types:
            filters= inclusion_filter(self.types) 
        elif ntype_convert(self.neurite_type) == "apical":
            filters = has_apicals_filter
        else:
            filters = None
        return filters
        
    def _get_ZERO_transform(self):
        """
        Defining the pre_transform according to the neurite type
        """
        ZERO_transform: list[object] = [
            transforms.ExtractTMDNeurites(neurite_type=ntype_convert(self.neurite_type))
        ]
        if self.orient:
            ZERO_transform.append(transforms.OrientApicals())
        if self.simplify:
            ZERO_transform.append(transforms.BranchingOnlyNeurites())
        ZERO_transform.append(transforms.ExtractEdgeIndex())
        ZERO_transform = transforms.Compose(ZERO_transform)
        return ZERO_transform
        
    def add_vecotized_pd(
        self, 
        pers_hom_function: Literal["radial_distances", "projection"] = "radial_distances", # radial_distances or path or else. 
        vectorization:  Union[VECTORIZATION_TYPES, List[VECTORIZATION_TYPES]] = "persistence_image", # persistence_image or persistence_diagram
        M: int = None, # for sliced wasserstein
        k: int = None, # for landscape
        m: int = None,  # for landscape
        flatten: bool = True, # for gaussian image
        normalize: bool = True, # for all vectorizations
        ):
        """
        Add the vectorized persistence diagrams to the dataset.

        Args:

            dataset (MorphologyDataset): The dataset to add the vectorized persistence diagrams to.
            pers_hom_function (str): The persistence homology function to use. 
                Options: "radial_distances", "projection", "path".
            vectorization (str or list): The vectorization method to use.
                Options: "persistence_image", "persistence_diagram", "landscape", "bottleneck", "wasserstein", "sliced_wasserstein".
            M (int): The number of slices for the sliced wasserstein vectorization.
            k (int): The number of landscapes for the landscape vectorization.
            m (int): The resolution for the landscape vectorization.
            flatten (bool): Whether to flatten the persistence image.
            normalize (bool): Whether to normalize the persistence diagrams.
        Returns:
            MorphologyDataset: The dataset with the vectorized persistence diagrams added.      
        """
        neurite_collection = [data.tmd_neurites for data in self.dataset_]
        ## Use get_persistence_diagram from tmd and not get_ph_neuron (as used in load_tmd) as we suppose that we only have 1 neutrite type

        diagrams = [
            np.array([
                persistence
                for tree in neurites
                for persistence in get_persistence_diagram(tree, pers_hom_function)
            ])
            for neurites in neurite_collection
        ]
        # Normalize diagrams: 
        xmin, ymin = np.stack([d.min(axis=0) for d in diagrams]).min(axis=0)
        xmax, ymax = np.stack([d.max(axis=0) for d in diagrams]).max(axis=0)
        xscale = max(abs(xmax), abs(xmin))
        yscale = max(abs(ymax), abs(ymin))
        ## Cannot use torch.tensor(scale) as need type for dinosius on which pw distance is computed
        ## torch version: 
        ##      scale = np.array([[xscale, yscale]])
        ##      normalized_diagrams = [torch.tensor(diagram / scale).float() for diagram in diagrams]
        normalized_diagrams = [
            [(float(point[0] / xscale), float(point[1] / yscale)) for point in diagram]
            for diagram in diagrams
        ]
        ## Remove bad diagrams
        idx_keep = [idx for idx in range(len(self.dataset_)) if idx not in _find_bad_diagrams(self.dataset_)]
        self.dataset_ = self.dataset_.index_select(idx_keep)
        diagrams = [diagrams[idx] for idx in idx_keep]

        vectorized_diagrams = vectorize_persistence_diagrams(
            normalized_diagrams,
            vectorization,
            M=M,
            k=k,
            m=m,
            flatten=flatten
        )

        if normalize:
            for v in vectorization:
                features = vectorized_diagrams[v]
                
                if v != "persistence_image":
                    # Case 1: Non-image vectorization (list of 1D arrays, each of shape (n_samples,))
                    X = np.stack(features, axis=1)  # shape: (n_samples, n_features)
                    scaler = RobustScaler(with_centering=False)
                    X_scaled = scaler.fit_transform(X)
                    vectorized_diagrams[v] = [X_scaled[:, i] for i in range(X_scaled.shape[1])]
                
                else:
                    # Case 2: Persistence images (list of 2D arrays)
                    # Flatten each image
                    flat_images = [img.flatten() for img in features]
                    X = np.stack(flat_images, axis=0)  # shape: (n_samples, n_pixels)

                    # Scale
                    scaler = RobustScaler(with_centering=False)
                    X_scaled = scaler.fit_transform(X)

                    # Reshape back to original image shape
                    original_shape = features[0].shape
                    scaled_images = [x.reshape(original_shape) for x in X_scaled]

                    vectorized_diagrams[v] = scaled_images

        # Add fields:
        data_list_of_dict = [data.__dict__ for data in self.dataset_]
        for i,dict in enumerate(data_list_of_dict):
            for v in vectorization:
                vec_formated = vectorized_diagrams[v][i]
                vec_formated = vec_formated[np.newaxis,np.newaxis] #newaxis for: shape = (batch, channels, orignial_shape) -> CNN expected
                dict[v] = torch.tensor(vec_formated.copy()).float()
        data_list_of_morphologyData = [MorphologyData.from_dict(data_list_of_dict[i]) for i  in range(len(data_list_of_dict))]
        dataset_out = MorphologyDataset(data_list_of_morphologyData)
        self.dataset_ = dataset_out

    def add_morphometrics(
        self, 
        morphometrics_config_file: str,
        normalize: bool = True,
    ):
        """
        Add morphometrics to the dataset.

        Args:
            dataset (MorphologyDataset): The dataset to add morphometrics to.
            morphometrics_config_file (str): The path to the morphometrics config file.
            remove_morphology (bool): Whether to remove the morphology field from the dataset.
        Returns:
            MorphologyDataset: The dataset with morphometrics added.
        """
        morphometrics_config_file = pathlib.Path(morphometrics_config_file)
        if not morphometrics_config_file.is_file():
            raise ValueError(f"Morphometrics config file not found: {morphometrics_config_file}")
        with morphometrics_config_file.open() as fh:
            morphometrics_config = yaml.safe_load(fh)
        
        # Add fields:
        data_list_of_dict = [data.__dict__ for data in self.dataset_]
        features = []
        for dict in data_list_of_dict:
            morphology = dict["morphology"]
            metrics = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                metrics_dict = morph_stats.extract_stats(morphology, morphometrics_config)
                for k, v in metrics_dict.items():
                    metrics.extend(v.values())
            features.append(np.array(metrics))
        
        if normalize:
            X = np.stack(features, axis=1)  # shape: (n_samples, n_features)
            scaler = RobustScaler(with_centering=False)
            X_scaled = scaler.fit_transform(X)
            features= [X_scaled[:, i] for i in range(X_scaled.shape[1])]
        
        for dict, metrics in zip(data_list_of_dict, features): 
            metrics_formated = metrics[np.newaxis,np.newaxis] #newaxis for: shape = (batch, channels, orignial_shape) -> CNN expected
            dict["morphometrics"] = torch.tensor(metrics_formated.copy()).float()
        data_list_of_morphologyData = [MorphologyData.from_dict(data_list_of_dict[i]) for i  in range(len(data_list_of_dict))]
        dataset_out = MorphologyDataset(data_list_of_morphologyData)
        self.dataset_ = dataset_out
    
    def scale_graph(
        self, 
        ):
        """
        Scale the GRAPH features (ONLY !) of the dataset using RobustScaler.
        Args:
            dataset (MorphologyDataset): The dataset to scale.
            feature_indices (list): The indices of the features to scale.
        Returns:
            MorphologyDataset: The scaled dataset.
            RobustScaler: The fitted scaler.
        """

        ## scale node features (x)
        idx = list(range(self.dataset_[0].x.shape[1]))
        scaler = transforms.FeatureRobustScaler(
                feature_indices=idx, 
                with_centering=False,
                )
        scaler.fit(self.dataset_)
        self.dataset_.transform = transforms.Compose([transforms.MakeCopy(), scaler])

        # scale global features (u)
        if hasattr(self.dataset_[0], "u") and self.dataset_[0].u is not None:
            idx = list(range(self.dataset_[0].u.shape[1]))
            scaler_glob = transforms.FeatureRobustScaler(
                    feature_indices=idx, 
                    is_global_feature=True,
                    )
            scaler_glob.fit(self.dataset_)
            self.dataset_.transform = transforms.Compose([transforms.MakeCopy(),scaler,scaler_glob])

    @staticmethod
    def _load_features(features_dir):
        """
        Load features from a directory.
        Args:
            features_dir (str): The directory to load features from.
        Returns:
            MorphologyDataset: The loaded dataset.
        """
        data = []
        features_dir = pathlib.Path(features_dir)
        for path in sorted(features_dir.glob("*.features")):
            data.append(MorphologyData.load(path))
        return MorphologyDataset(data)

    @classmethod
    def from_features_dir(cls, features_dir):
        instance = cls.__new__(cls)
        instance.dataset_ = cls._load_features(features_dir)
        return instance
    
    @property
    def dataset(self):
        return self.dataset_
    
    def remove_morphology_field(self):
        """
        Remove the morphology field from the dataset.

        This is required for when we want to save the dataset with pickle

        Args:
            dataset (MorphologyDataset): The dataset to remove the morphology field from.
        Returns:
            MorphologyDataset: The dataset with the morphology field removed.
        """
        keep_fields = [
            "edge_index", "edge_attr","u",
            "x", "y","y_str", "label", "z",
            "diagram","image","landscape" ,"bottleneck","wasserstein","sliced_wasserstein",
            "morphometrics",
            "path","tmd_neurites","__num_nodes__" # set by data.num_nodes = <value>]
            ]
        self.dataset_.transform = transforms.Compose(
            [transforms.MakeCopy(keep_fields), self.dataset_.transform]
        )


    def write_features(self, output_dir, force: bool = False):
        """
        Write the features of the dataset to a file.
        Args:
            dataset (MorphologyDataset): The dataset to write.
            output_dir (str): The directory to write the features to.
            force (bool): Whether to overwrite existing files.

        Returns:
            None
        """
        if hasattr(self.dataset_[0], "morphology"):
            self.remove_morphology_field()
            if hasattr(self.dataset_[0], "morphology"):
                raise ValueError("Morphology field not removed")
        if not force and output_dir.is_dir():
            raise ValueError(f"Output directory already exists: {output_dir}")
        output_dir = pathlib.Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, sample in enumerate(self.dataset_):
            # if i ==0: 
            #     for field, value in vars(sample).items():
            #         print(f"  {field}: {value}")
            file_name = pathlib.Path(sample.path).with_suffix(".features").name
            sample.save(output_dir / file_name)
