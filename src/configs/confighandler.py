import yaml
from morphoclass import transforms

class Config:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)
        self._parse()

    def _parse(self):
        self.PATH = self.cfg['PATH']
        self.OUT_PATH = self.cfg['OUT_PATH']
        self.LAYER = self.cfg['LAYER']
        self.TYPES = self.cfg['TYPES']
        self.NEURITE_TYPE = self.cfg['NEURITE_TYPE']
        self.CONFIG_MORPHOMETRICS = self.cfg['CONFIG_MORPHOMETRICS']
        self.NORMALIZE = self.cfg['NORMALIZE']
        self.PH_F = self.cfg['PH_F']
        self.VECTORIZATION = self.cfg['VECTORIZATION']
        self.FLATTEN = self.cfg['FLATTEN']
        self.M_SW = self.cfg['M_SW']
        self.K_LS = self.cfg['K_LS']
        self.M_LS = self.cfg['M_LS']
        self.ADD_VEC = self.cfg['ADD_VEC']
        self.ADD_MORPH = self.cfg['ADD_MORPH']
        self.FEATURE_EXTRACTOR = self._build_feature_extractor(self.cfg['FEATURE_EXTRACTOR'])

    def _build_feature_extractor(self, extractor_names):
        transform_list = []
        for name in extractor_names:
            try:
                transform_cls = getattr(transforms, name)
                transform_list.append(transform_cls())
            except AttributeError:
                raise ValueError(f"Unknown transform: {name}")
        return transforms.Compose(transform_list)

# Usage:
# config = Config("config.yaml")
# features = config.FEATURE_EXTRACTOR
