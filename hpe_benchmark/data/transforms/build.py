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

    transform = transforms.Compose(
        [
            color_jitter,
            transforms.ToTensor(),
            normalize,
        ]
    )
    return transform
