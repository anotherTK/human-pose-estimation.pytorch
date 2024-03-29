
import cv2
import json
import os
import numpy as np

from hpe_benchmark.data.datasets.joints import JointsDataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class COCODataset(JointsDataset):
    def __init__(self, cfg, stage, transform=None):
        super().__init__(cfg, stage, transform)
        self.cur_dir = cfg.DATA.DATASET_ROOT

        self.train_gt_file = 'person_keypoints_train2017.json'
        self.train_gt_path = os.path.join(self.cur_dir, 'annotations',
                                          self.train_gt_file)

        self.val_gt_file = 'person_keypoints_val2017.json'
        self.val_gt_path = os.path.join(self.cur_dir, 'annotations',
                                        self.val_gt_file)
        self.val_det_file = 'person_keypoints_val2017_det.json'
        self.val_det_path = os.path.join(self.cur_dir, 'annotations',
                                         self.val_det_file)

        self.test_det_file = ''
        self.test_det_path = os.path.join(self.cur_dir, 'annotations',
                                          self.test_det_file)

        self._exception_ids = ['366379']

        self.data = self._get_data()
        self.data_num = len(self.data)

    def _get_data(self):
        data = list()
        if self.stage == 'train':
            coco = COCO(self.train_gt_path)
        elif self.stage == 'val':
            coco = COCO(self.val_gt_path)
            self.val_gt = coco
        else:
            pass

        if self.stage == 'train':
            for aid, ann in coco.anns.items():
                img_id = ann['image_id']
                if img_id not in coco.imgs or img_id in self._exception_ids:
                    continue

                if ann['iscrowd']:
                    continue

                img_name = coco.imgs[img_id]['file_name']
                prefix = self.stage + '2017'
                img_path = os.path.join(self.cur_dir, prefix, img_name)

                bbox = np.array(ann['bbox'])
                area = ann['area']
                joints = np.array(ann['keypoints']).reshape((-1, 3))
                headRect = np.array([0, 0, 1, 1], np.int32)

                center, scale = self._bbox_to_center_and_scale(bbox)

                if np.sum(joints[:, -1] > 0) < self.kp_load_min_num or \
                        ann['num_keypoints'] == 0:
                    continue

                d = dict(
                    aid=aid,
                    area=area,
                    bbox=bbox,
                    center=center,
                    headRect=headRect,
                    img_id=img_id,
                    img_name=img_name,
                    img_path=img_path,
                    joints=joints,
                    scale=scale
                )
                data.append(d)

        else:
            if self.stage == 'val':
                det_path = self.val_det_path
            else:
                det_path = self.test_det_path
            dets = json.load(open(det_path))

            for det in dets:
                if det['image_id'] not in coco.imgs or det['category_id'] != 1:
                    continue

                img_id = det['image_id']
                img_name = '{:0>12}.jpg'.format(img_id)
                img_path = os.path.join(
                    self.cur_dir, self.stage + '2017', img_name)

                bbox = np.array(det['bbox'])
                center, scale = self._bbox_to_center_and_scale(bbox)
                joints = np.zeros((self.keypoint_num, 3))
                score = det['score']
                headRect = np.array([0, 0, 1, 1], np.int32)

                d = dict(
                    bbox=bbox,
                    center=center,
                    headRect=headRect,
                    img_id=img_id,
                    img_name=img_name,
                    img_path=img_path,
                    joints=joints,
                    scale=scale,
                    score=score
                )
                data.append(d)
        return data

    def _bbox_to_center_and_scale(self, bbox):
        x, y, w, h = bbox

        center = np.zeros(2, dtype=np.float32)
        center[0] = x + w / 2.0
        center[1] = y + h / 2.0

        # TODO: how pixel_std affect the performace -> link to transform[note]
        scale = np.array([w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
                         dtype=np.float32)

        return center, scale

    def evaluate(self, pred_path):
        pred = self.val_gt.loadRes(pred_path)
        coco_eval = COCOeval(self.val_gt, pred, iouType='keypoints')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    def visualize(self, img, joints, score=None):
        pairs = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                 [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                 [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        color = np.random.randint(0, 256, (self.keypoint_num, 3)).tolist()

        for i in range(self.keypoint_num):
            if joints[i, 0] > 0 and joints[i, 1] > 0:
                cv2.circle(img, tuple(joints[i, :2]), 2, tuple(color[i]), 2)

        if score:
            cv2.putText(img, score, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (128, 255, 0), 2)

        def draw_line(img, p1, p2):
            c = (0, 0, 255)
            if p1[0] > 0 and p1[1] > 0 and p2[0] > 0 and p2[1] > 0:
                cv2.line(img, tuple(p1), tuple(p2), c, 2)

        for pair in pairs:
            draw_line(img, joints[pair[0] - 1], joints[pair[1] - 1])

        return img


if __name__ == "__main__":
    from hpe_benchmark.config import cfg
    dataset = COCODataset(cfg, 'train')
    print(dataset.data_num)
    print(dataset.data[0])
    print(dataset.__getitem__(0))
