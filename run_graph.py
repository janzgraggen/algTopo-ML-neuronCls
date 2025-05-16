from src.data.load_graph import  MorphologyDatasetManager
from morphoclass import transforms 
# ─────────────────────────────────────────────────────
# 1) Data loading parameters
# ─────────────────────────────────────────────────────
PATH = "assets/datasets_structured_layer/kanari18_laylab"
OUT_PATH = "output/multiconcat_features_full"
LAYER = "L5"
TYPES = ""  # or ["L5_TPC:A", ...]
NEURITE_TYPE = "apical_dendrite"

FEATURE_EXTRACTOR = transforms.Compose([
    # feature extraction. 
    
    # GLOBAL:
    #transforms.ExtractNumberLeaves(),
    #transforms.ExtractNumberBranchPoints(),
    # transforms.ExtractMaximalApicalPathLength(),
    # transforms.TotalPathLength(),
    # transforms.AverageBranchOrder(),
    # transforms.AverageRadius(),

    # EDGE:
    transforms.ExtractDistanceWeights(),

    # NODE:
    # transforms.ExtractBranchingAngles(),
    # transforms.ExtractConstFeature(),
    # transforms.ExtractCoordinates(),
    # transforms.ExtractDiameters(),
    # transforms.ExtractIsBranching(),
    # transforms.ExtractIsIntermediate(),
    # transforms.ExtractIsLeaf(),
    # transforms.ExtractIsRoot(),
    # transforms.ExtractPathDistances(),
    transforms.ExtractRadialDistances(),
    # transforms.ExtractVerticalDistances()
])
CONFIG_MORPHOMETRICS = "CLI/configs/feature-morphometrics.yaml"
NORMALIZE = True # normalize the features

PH_F = "radial_distances"   
VECTORIZATION = ["persistence_image", "wasserstein"  ]#  ,"bottleneck","sliced_wasserstein", "landscape"] # or "landscape" or "bottleneck" or "wasserstein" or "slice_wasserstein"
FLATTEN = False # flatten the image
M_SW = 20 #  sliced_wasserstein: number of slices
K_LS = 10 #  landscape: number of landscapes
M_LS = 5 #  landscape:resolution

# ─────────────────────────────────────────────────────
# 2) Load data
# ─────────────────────────────────────────────────────
mdm = MorphologyDatasetManager(
    datapath=PATH,
    layer=LAYER,
    types=TYPES,
    neurite_type=NEURITE_TYPE,
    feature_extractor=FEATURE_EXTRACTOR,
    normalize= NORMALIZE
    )
mdm.add_vecotized_pd(
    pers_hom_function=PH_F,
    vectorization=VECTORIZATION,
    flatten=FLATTEN,
    M=M_SW,
    k=K_LS,
    m=M_LS,
    normalize=NORMALIZE
    )
mdm.add_morphometrics(
    morphometrics_config_file=CONFIG_MORPHOMETRICS,
    normalize=NORMALIZE,
    remove_morphology=True
    )
mdm.write_features(
    output_dir=OUT_PATH,
    force=True,
    )

# ─────────────────────────────────────────────────────
print(mdm.dataset[0].__dict__.keys())
for key, value in mdm.dataset[0].__dict__.items():
    print(f"{key}: {value}")

# ─────────────────────────────────────────────────────
mdm_read = MorphologyDatasetManager.from_features_dir(OUT_PATH)
print(mdm_read.dataset[0].__dict__.keys())
for key, value in mdm_read.dataset[0].__dict__.items():
    print(f"{key}: {value}") 