

import logging
import time
import os
import cv2
import numpy as np
import json
import torch
from tqdm import tqdm

from hpe_benchmark.config import cfg
from hpe_benchmark.utils.comm import is_main_process, get_world_size
from hpe_benchmark.utils.comm import all_gather
from hpe_benchmark.utils.comm import synchronize
from hpe_benchmark.utils.timer import Timer, get_time_str
from hpe_benchmark.data.transforms import flip_back
import hpe_benchmark.data.datasets as D

def transform_back(outputs, centers, scales, kernel=11, shifts=[0.25]):
    scales *= 200
    nr_img = outputs.shape[0]
    preds = np.zeros((nr_img, cfg.KEYPOINT.NUM, 2))
    maxvals = np.zeros((nr_img, cfg.KEYPOINT.NUM, 1))
    for i in range(nr_img):
        score_map = outputs[i].copy()
        score_map = score_map / 255 + 0.5
        kps = np.zeros((cfg.KEYPOINT.NUM, 2))
        scores = np.zeros((cfg.KEYPOINT.NUM, 1))
        border = 10
        dr = np.zeros((cfg.KEYPOINT.NUM,
                       cfg.DATA.OUTPUT_SHAPE[0] + 2 * border, cfg.DATA.OUTPUT_SHAPE[1] + 2 * border))
        dr[:, border: -border, border: -border] = outputs[i].copy()
        for w in range(cfg.KEYPOINT.NUM):
            dr[w] = cv2.GaussianBlur(dr[w], (kernel, kernel), 0)
        for w in range(cfg.KEYPOINT.NUM):
            for j in range(len(shifts)):
                if j == 0:
                    lb = dr[w].argmax()
                    y, x = np.unravel_index(lb, dr[w].shape)
                    dr[w, y, x] = 0
                    x -= border
                    y -= border
                lb = dr[w].argmax()
                py, px = np.unravel_index(lb, dr[w].shape)
                dr[w, py, px] = 0
                px -= border + x
                py -= border + y
                ln = (px ** 2 + py ** 2) ** 0.5
                if ln > 1e-3:
                    x += shifts[j] * px / ln
                    y += shifts[j] * py / ln
            x = max(0, min(x, cfg.DATA.OUTPUT_SHAPE[1] - 1))
            y = max(0, min(y, cfg.DATA.OUTPUT_SHAPE[0] - 1))
            kps[w] = np.array([x * 4 + 2, y * 4 + 2])
            scores[w, 0] = score_map[w, int(round(y) + 1e-9),
                                     int(round(x) + 1e-9)]
        # aligned or not ...
        kps[:, 0] = kps[:, 0] / cfg.DATA.INPUT_SHAPE[1] * scales[i][0] + \
            centers[i][0] - scales[i][0] * 0.5
        kps[:, 1] = kps[:, 1] / cfg.DATA.INPUT_SHAPE[0] * scales[i][1] + \
            centers[i][1] - scales[i][1] * 0.5
        preds[i] = kps
        maxvals[i] = scores

    return preds, maxvals

def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    results = []
    cpu_device = torch.device("cpu")
    data = tqdm(data_loader) if is_main_process() else data_loader
    for _, batch in enumerate(data):
        imgs, scores, centers, scales, img_ids = batch
        imgs = imgs.to(device)
        with torch.no_grad():
            if timer:
                timer.tic()
            outputs = model(imgs)
            outputs = outputs.to(cpu_device).numpy()

            if cfg.TEST.FLIP:
                imgs_flipped = np.flip(imgs.to(cpu_device).numpy(), 3).copy()
                imgs_flipped = torch.from_numpy(imgs_flipped).to(device)
                outputs_flipped = model(imgs_flipped)
                outputs_flipped = outputs_flipped.to(cpu_device).numpy()
                outputs_flipped = flip_back(
                    outputs_flipped, cfg.KEYPOINT.FLIP_PAIRS)

                outputs = (outputs + outputs_flipped) * 0.5
            if timer:
                if not cfg.MODEL.DEVICE == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
        
        centers = np.array(centers)
        scales = np.array(scales)
        preds, maxvals = transform_back(outputs, centers, scales, cfg.TEST.GAUSSIAN_KERNEL, cfg.TEST.SHIFT_RATIOS)
        kp_scores = maxvals.squeeze().mean(axis=1)
        preds = np.concatenate((preds, maxvals), axis=2)

        if isinstance(data_loader.dataset, D.MPIIDataset):
            results.append(preds)
        else:
            for i in range(preds.shape[0]):
                keypoints = preds[i].reshape(-1).tolist()
                score = scores[i] * kp_scores[i]
                image_id = img_ids[i]

                results.append(dict(image_id=image_id,
                                    category_id=1,
                                    keypoints=keypoints,
                                    score=score))
    return results


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)

    if not is_main_process():
        return

    predictions = list()
    for p in all_predictions:
        predictions.extend(p)

    return predictions

def inference(
    model,
    data_loader,
    dataset_name,
    iou_types=("bbox",),
    device="cuda",
    output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("HPE.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(
        dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, device, inference_timer)
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time *
            num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        #torch.save(predictions, os.path.join(output_folder, "predictions.pth"))
        if isinstance(data_loader.dataset, D.MPIIDataset):
            predictions = np.vstack(predictions)
            data_loader.dataset.evaluate(predictions)
        else:
            with open(os.path.join(output_folder, "predictions.json"), 'w') as w_obj:
                json.dump(predictions, w_obj)
            data_loader.dataset.evaluate(
                os.path.join(output_folder, "predictions.json"))
