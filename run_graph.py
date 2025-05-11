from src.data.load_graph import load_graph , write_features, scale_graph, add_vecotized_pd
from src.train.train_graph import ScaledTrainer
from morphoclass import transforms ,models,data
import torch
import morphoclass


# ─────────────────────────────────────────────────────
# 1) Data loading parameters
# ─────────────────────────────────────────────────────
PATH = "assets/datasets_structured_layer/kanari18_laylab"
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
    # transforms.ExtractEdgeIndex(),
    # transforms.ExtractDistanceWeights(),

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

PH_F = "radial_distances"   
VECTORIZATION = ["persistence_image", "wasserstein","bottleneck","sliced_wasserstein", "landscape"] # or "landscape" or "bottleneck" or "wasserstein" or "slice_wasserstein"
FLATTEN = True # flatten the image
M_SW = 20 #  sliced_wasserstein: number of slices
K_LS = 1 #  landscape: number of landscapes
M_LS = 1 #  landscape:resolution

# ─────────────────────────────────────────────────────
# 2) Load data
# ─────────────────────────────────────────────────────
dataset = load_graph(
    datapath=PATH,
    layer=LAYER,
    types=TYPES,
    neurite_type=NEURITE_TYPE,
    feature_extractor=FEATURE_EXTRACTOR,
    verbose=False
)

dataset, fitted_scaler = scale_graph(dataset)

dataset_extended = add_vecotized_pd(
    dataset,
    pers_hom_function="radial_distances",
    vectorization=['persistence_image', 'wasserstein', 'bottleneck', 'sliced_wasserstein', 'landscape'],
    flatten=False,
    )

print("**********")
print("after passing through add_vecotized_pd")
for key , val in vars(dataset_extended[0]).items():
    print(key, val)

write_features(
    dataset_extended,
    output_dir="output/myown_features",
    force=True
    )

# ─────────────────────────────────────────────────────
# 3) Instantiate model
# ─────────────────────────────────────────────────────
LR = 1e-2
DEVICE = "cuda"  # or "cpu"
TRAIN_STRATEGY = "crossval"  # "single_split" or "crossval"
in_feat = dataset[0].x.shape[1] if dataset[0].x is not None else 0
out_classes = len(set(dataset.labels))
MODEL = models.ManNet(in_feat, 0, out_classes)
OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=LR, weight_decay=0.01)

# ─────────────────────────────────────────────────────
# 5) Build trainer & run
# ─────────────────────────────────────────────────────