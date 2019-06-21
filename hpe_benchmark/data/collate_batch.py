

class BatchCollator(object):
    def __init__(self, size_divisible, stage):
        self.stage = stage
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        if self.stage == 'train':
            images = torch.stack(transposed_batch[0], dim=0)
            valids = torch.stack(transposed_batch[1], dim=0)
            labels = torch.stack(transposed_batch[2], dim=0)
            
            return images, valids, labels
        else:
            images = torch.stack(transposed_batch[0], dim=0)
            scores = list(transposed_batch[1])
            centers = list(transposed_batch[2])
            scales = list(transposed_batch[3])
            image_ids = list(transposed_batch[4])

            return images, scores, centers, scales, image_ids
