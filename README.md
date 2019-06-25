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
| EFFN-b4 | Y          | COCO    | Adam        | 96k       | *       |
| MSPN    | Y          | MPII    | Adam(paper) | 28.8k     | *       |
| RES-50  | Y          | MPII    | Adam        | 28.8k     | *       |
| HR-W48  | Y          | MPII    | Adam        | 28.8k     | *       |
| EFFN-b4 | Y          | MPII    | Adam        | 28.8k     | *       |


## Miletone

- [x] coco dataset support
- [x] add MSPN
- [x] add resnet
- [x] add hrnet
- [x] add efficientnet
- [x] mpii dataset support