from .backbone import CLIPBackbone
from .clip_model import CLIPRetrieval
from .adapter import ResidualAdapter
from .local_head import LocalPatchHead
from .local_region import (
    sample_region_boxes,
    crop_regions,
    pool_region_patches,
)