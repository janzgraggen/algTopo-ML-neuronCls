from src.data.load_graph import  MorphologyDatasetManager
from src.configs.confighandler import Config
import sys

if len(sys.argv) != 2:
    print("!!! Usage: python run_MultiFeatureExtract.py <config_path>")
    print("-> proceed with default config: src/configs/extract-simple.yaml")
    CFG_PATH = "src/configs/extract-simple.yaml"
else:
    CFG_PATH = sys.argv[1]
CFG = Config(CFG_PATH)

# ─────────────────────────────────────────────────────
# 2) Load data
# ─────────────────────────────────────────────────────
mdm = MorphologyDatasetManager(
    datapath=CFG.PATH,
    layer=CFG.LAYER,
    types=CFG.TYPES,
    neurite_type=CFG.NEURITE_TYPE,
    feature_extractor=CFG.FEATURE_EXTRACTOR,
    normalize=CFG.NORMALIZE
    )
if CFG.ADD_VEC:
    mdm.add_vecotized_pd(
        pers_hom_function=CFG.PH_F,
        vectorization=CFG.VECTORIZATION,
        flatten=CFG.FLATTEN,
        M=CFG.M_SW,
        k=CFG.K_LS,
        m=CFG.M_LS,
        normalize=CFG.NORMALIZE
        )
if CFG.ADD_MORPH:
    mdm.add_morphometrics(
        morphometrics_config_file=CFG.CONFIG_MORPHOMETRICS,
        normalize=CFG.NORMALIZE
        )
mdm.write_features(
    output_dir=CFG.OUT_PATH,
    force=True,
    )

# ─────────────────────────────────────────────────────
print("TEST: (end)  printing the keys of the first dataset")
print(mdm.dataset[0].__dict__.keys())
for key, value in mdm.dataset[0].__dict__.items():
    print(f"{key}: {value}")

# ─────────────────────────────────────────────────────
mdm_read = MorphologyDatasetManager.from_features_dir(CFG.OUT_PATH)
print("TEST: (end)  printing the keys of the reloaded dataset")
print(mdm_read.dataset[0].__dict__.keys())
for key, value in mdm_read.dataset[0].__dict__.items():
    print(f"{key}: {value}") 