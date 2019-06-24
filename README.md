# human-pose-estimation.pytorch
This repo aims to integral many fantastic works in human pose estimation with pytorch. This codes are heavily borrowed from HRNet and MSPNet, the code structure imitates the maskrcnn-benchmark, aims to fast, modular and distributed training with pytorch! 

| model  | pretrained | optimizer   | iteration | eval AP |
| ------ | ---------- | ----------- | --------- | ------- |
| MSPN   | N          | SGD         | 96k       | 72.5    |
| MSPN   | N          | Adam        | 96k       | 73.5    |
| MSPN   | Y          | Adam        | 96k       | 74.7    |
| MSPN   | Y          | Adam(paper) | 96k       | *       |
| RES-50 | Y          | Adam        | 96k       | *       |
| HR-W48 | Y          | Adam        | 96k       | *       |


## Miletone

- [x] add MSPN
- [x] add resnet
- [x] add hrnet
- [ ] add efficientnet