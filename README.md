# human-pose-estimation.pytorch
This repo aims to integral many fantastic works in human pose estimation with pytorch. This codes are heavily borrowed from HRNet and MSPNet, the code structure imitates the maskrcnn-benchmark, aims to fast, modular and distributed training with pytorch! 

| model   | pretrained | dataset | optimizer   | iteration | eval AP |
| ------- | ---------- | ------- | ----------- | --------- | ------- |
| MSPN    | N          | COCO    | SGD         | 96k       | 72.5    |
| MSPN    | N          | COCO    | Adam        | 96k       | 73.5    |
| MSPN    | Y          | COCO    | Adam        | 96k       | 74.7    |
| MSPN    | Y          | COCO    | Adam(paper) | 96k       | 74.6    |
| RES-50  | Y          | COCO    | Adam        | 96k       | 70.9    |
| RES-101 | Y          | COCO    | Adam        | 96k       | *       |
| HR-W48  | Y          | COCO    | Adam        | 96k       | 74.2    |
| EFFN-b4 | Y          | COCO    | Adam        | 96k       | 71.4    |
| MSPN    | Y          | MPII    | Adam(paper) | 28.8k     | 90.17   |
| RES-50  | Y          | MPII    | Adam        | 28.8k     | 88.94   |
| HR-W48  | Y          | MPII    | Adam        | 28.8k     | *       |
| EFFN-b4 | Y          | MPII    | Adam        | 28.8k     | 88.88   |


## Miletone

- [x] coco dataset support
- [x] add MSPN
- [x] add resnet
- [x] add hrnet
- [x] add efficientnet
- [x] mpii dataset support

## Usage

Train the model by yourself is easily with this repo.

1. Clone this repo

```sh
git clone https://github.com/tkianai/human-pose-estimation.pytorch
```

2. Prepare the data

```sh
# overall
datasets
├── coco
└── mpii
# coco
├── annotations
    ├── captions_train2017.json
    ├── captions_val2017.json
    ├── instances_train2017.json
    ├── instances_val2017.json
    ├── person_keypoints_train2017.json
    ├── person_keypoints_val2017_det.json.json
    └── person_keypoints_val2017.json
├── train2017
├── val2017
# mpii
├── annotations
    ├── test.json
    ├── train.json
    ├── valid.json
    └── valid.mat
└── images
```

3. Train

```sh
# single GPU training
python train.py --config-file configs/coco_mspn.yaml
# distributed training
python -m torch.distributed.launch --nproc_per_node=8 train.py --config-file configs/coco_mspn.yaml
```

4. Test

```sh
# single GPU testing
python test.py --config-file configs/coco_mspn.yaml --ckpt <model location>
# distributed testing
python -m torch.distributed.launch --nproc_per_node=8 test.py --config-file configs/coco_mspn.yaml --ckpt <model location>
```

## Attention

Try my best to keep update the new model structure, datasets and training strategies. Please star this repo~~~

New pull requests are strongly welcomed!