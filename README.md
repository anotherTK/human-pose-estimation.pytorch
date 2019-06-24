# human-pose-estimation.pytorch
This repo aims to integral many fantastic works in human pose estimation with pytorch.

| model | pretrained | optimizer   | iteration | eval AP |
| ----- | ---------- | ----------- | --------- | ------- |
| MSPN  | N          | SGD         | 96k       | 72.5    |
| MSPN  | N          | Adam        | 96k       | 73.5    |
| MSPN  | Y          | Adam        | 96k       | 74.7    |
| MSPN  | Y          | Adam(paper) | 96k       | *       |