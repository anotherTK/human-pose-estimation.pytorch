from .coco import COCODataset

__all__ = [
    "COCODataset",
]

def get(name):
    if name == "COCO":
        return COCODataset
    else:
        raise NotImplementedError("Dataset {} is not supported yet!".format(name))