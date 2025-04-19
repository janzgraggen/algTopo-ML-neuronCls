from src.data.load_graph import load_graph
from morphoclass import transforms
"""
Unlike in docs_ read form structured dir expects path -> Layer -> Label -> Files structure
"""
# Path to morphology file (make sure this exists)
PATH = "assets" # or "assets/L2_TPC:A" or "assets/L5_UTPC"
LAYER = "L2" # "L2" # or "L3" or "L4" or "L5" pr list : ["L2", "L3"]
TYPES = "" #["L2_IPC", "L2_TPC:A"] or "L2" or other.
NEUTRITE_TYPE = "apical_dendrite" # "basal_dendrite", "axon", "neurites", "all"
PREPROCESS = transforms.Compose([
    # augmenting: 
    transforms.BranchingOnlyNeurites(),
])
FEATURE_EXTRACTOR = transforms.Compose([
    # feature extraction. 
    transforms.ExtractPathDistances(),
    transforms.ExtractDiameters()
])

dataset = load_graph(
    datapath= PATH,
    layer= LAYER,
    types= TYPES,
    neurite_type= NEUTRITE_TYPE,
    preprocess_transform= PREPROCESS,
    feature_extractor= FEATURE_EXTRACTOR,
    verbose= True,
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
