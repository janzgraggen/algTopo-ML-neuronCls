from src.data.load_graph import load_graph , write_features, scale_graph
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
    # transforms.ExtractNumberLeaves(),
    # transforms.ExtractNumberBranchPoints(),
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


# ─────────────────────────────────────────────────────
# 2) Load data
# ─────────────────────────────────────────────────────
dataset = load_graph(
    datapath=PATH,
    layer=LAYER,
    types=TYPES,
    neurite_type=NEURITE_TYPE,
    feature_extractor=FEATURE_EXTRACTOR,
    verbose=True
)

dataset, fitted_scaler = scale_graph(dataset)

print("\n\n\n")
print("**********")
write_features(
    dataset,
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
trainer = ScaledTrainer(MODEL, dataset, optimizer=OPTIMIZER, loader_class=data.MorphologyDataLoader)

# history = trainer.train_single_split(
#     split_ratio=0.8,
#     n_epochs=100,
#     batch_size=16,
#     load_best=True,
#     )
# history = trainer.train_crossval(
#     n_splits=5,
#     n_epochs=100,
#     batch_size=16,
#     load_best=False,
#     )

# #print("Training complete. History keys:", history[0].keys() if isinstance(history, list) else history.keys())
# for k,v in history.items():
#     print("************")
#     print(f"{k}:\n    {v}")