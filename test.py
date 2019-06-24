
import argparse
import os

import torch
from hpe_benchmark.config import cfg
from hpe_benchmark.data import make_data_loader
from hpe_benchmark.engine.inference import inference
from hpe_benchmark.models import build_model
from hpe_benchmark.utils.checkpoint import Checkpointer
from hpe_benchmark.utils.comm import synchronize, get_rank
from hpe_benchmark.utils.logger import setup_logger
from hpe_benchmark.utils.miscellaneous import mkdir

# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Human Pose Estimation Inference")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--ckpt",
        help="The path to the checkpoint for test, default is the latest checkpoint.",
        default=None,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = os.path.join(cfg.OUTPUT_DIR, 'test')
    mkdir(save_dir)
    logger = setup_logger("HPE", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    model = build_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    # Initialize mixed-precision if necessary
    use_mixed_precision = cfg.DTYPE == 'float16'
    amp_handle = amp.init(enabled=use_mixed_precision, verbose=cfg.AMP_VERBOSE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = Checkpointer(model, save_dir=output_dir)
    ckpt = cfg.MODEL.WEIGHT if args.ckpt is None else args.ckpt
    _ = checkpointer.load(ckpt, use_latest=args.ckpt is None)

    iou_types = ("keypoints",)
    dataset_name = cfg.DATA.DATASET_NAME
    output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
    mkdir(output_folder)
    data_loader_test = make_data_loader(cfg, stage=("test" if cfg.TESTSET_ENABLE else "val"), is_distributed=distributed)
    inference(
        model,
        data_loader_test,
        dataset_name=dataset_name,
        iou_types=iou_types,
        device=cfg.MODEL.DEVICE,
        output_folder=output_folder,
    )
    synchronize()


if __name__ == "__main__":
    main()
