from . import MSPN
from . import PResNet

_META_ARCH = {
    "MSPN": MSPN,
    "RESNET": PResNet,
}

def build_model(cfg):
    meta_arch = _META_ARCH[cfg.MODEL.ARCH]
    return meta_arch(cfg)