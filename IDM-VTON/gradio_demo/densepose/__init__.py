# Copyright (c) Facebook, Inc. and its affiliates.
# Data/evaluation imports are optional at load-time (not needed for single-image inference)
try:
    from .data.datasets import builtin  # just to register data
except (ImportError, Exception):
    pass
try:
    from .converters import builtin as builtin_converters  # register converters
except (ImportError, Exception):
    pass
from .config import (
    add_densepose_config,
    add_densepose_head_config,
    add_hrnet_config,
    add_dataset_category_config,
    add_bootstrap_config,
    load_bootstrap_config,
)
from .structures import DensePoseDataRelative, DensePoseList, DensePoseTransformData
try:
    from .evaluation import DensePoseCOCOEvaluator
except (ImportError, Exception):
    pass
from .modeling.roi_heads import DensePoseROIHeads
try:
    from .modeling.test_time_augmentation import (
        DensePoseGeneralizedRCNNWithTTA,
        DensePoseDatasetMapperTTA,
    )
except (ImportError, Exception):
    pass
try:
    from .utils.transform import load_from_cfg
except (ImportError, Exception):
    pass
from .modeling.hrfpn import build_hrfpn_backbone
