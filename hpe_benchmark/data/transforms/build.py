import random
import torchvision.transforms as transforms

def build_transforms(cfg, is_train=True):
    if is_train:
        brightness = cfg.DATA.BRIGHTNESS
        contrast = cfg.DATA.CONTRAST
        saturation = cfg.DATA.SATURATION
        hue = cfg.DATA.HUE
    else:
        brightness = 0.0
        contrast = 0.0
        saturation = 0.0
        hue = 0.0

    normalize = transforms.Normalize(
        mean=cfg.DATA.MEANS,
        std=cfg.DATA.STDS
    )
    color_jitter = transforms.ColorJitter(
        brightness=brightness,
        contrast=contrast,
        saturation=saturation,
        hue=hue,
    )

    transform = Compose(
        [
            transforms.ToPILImage(),
            color_jitter,
        ],
        [
            transforms.ToTensor(),
            normalize,
        ],
    )
    return transform


class Compose(object):
    def __init__(self, random_transforms, transforms):
        self.random_transforms = random_transforms
        self.transforms = transforms

    def __call__(self, image):
        for t in self.random_transforms:
            if random.random() > 0.5:
                image = t(image)
        for t in self.transforms:
            image = t(image)
        return image

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        for t in self.random_transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string
