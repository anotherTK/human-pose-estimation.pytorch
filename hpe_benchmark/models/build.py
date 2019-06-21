from . import MSPN

_META_ARCH = {
    "MSPN": MSPN,
}

def build_model(cfg):
    meta_arch = _META_ARCH[cfg.MODEL.ARCH]
    return meta_arch(cfg)