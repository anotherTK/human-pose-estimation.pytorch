from . import MSPN
from . import PRESN
from . import PHRN
from . import PEFFN

_META_ARCH = {
    "MSPN": MSPN,
    "RESNET": PRESN,
    "HRNET": PHRN,
    "EFFNET": PEFFN,
}

def build_model(cfg):
    meta_arch = _META_ARCH[cfg.MODEL.ARCH]
    return meta_arch(cfg)