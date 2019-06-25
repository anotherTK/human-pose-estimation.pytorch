from .coco import COCODataset
from .mpii import MPIIDataset

__all__ = [
    "COCODataset",
    "MPIIDataset",
]

def get(name):
    if name == "COCO":
        return COCODataset
    elif name == "MPII":
        return MPIIDataset
    else:
        raise NotImplementedError("Dataset {} is not supported yet!".format(name))
