# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Modified by Youngwan Lee (ETRI), 2020. All Rights Reserved.
import logging
import os
from collections import OrderedDict
import torch

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    # CityscapesInstanceEvaluator,
    # CityscapesSemSegEvaluator,
    # COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from centermask.evaluation import (
    COCOEvaluator,
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.checkpoint import DetectionCheckpointer
from centermask.config import get_cfg

from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator

from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime
import logging
from detectron2.data import build_detection_train_loader
from detectron2.structures import BoxMode, PolygonMasks, polygons_to_bitmask, BitMasks
import numpy as np
from detectron2.data import detection_utils as utils
import detectron2.data.transforms as T
import albumentations as A
import cv2, random

import numpy as np
from fvcore.transforms.transform import Transform, NoOpTransform
from detectron2.data.transforms.augmentation import Augmentation

from pathlib import Path
import json
from detectron2.data.datasets.register_coco import register_coco_instances

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class Segment2COCO:
    """
    This class to handle data extracted from the extractor and then store it in COCO json Format
    """

    def __init__(self, categories=None):
        """
        Args:
            categories (dict): Dictionary represent the labels
        """
        # Create empty data first
        # self.reference_json = reference_json
        self.coco_format = {
            "info": {
                "year": 2021,
                "version": "1.0",
                "description": "Annotated of Segmented images",
                "contributor": "KAID",
                "date_created": str(datetime.datetime.now())
            },
            "licences": [
                {
                    "id": 1,
                    "name": "",
                    "url": ""
                }
            ],
            "categories": [],  # [{}]
            "images": [],  # [{}]
            "annotations": []  # [{}]
        }
        # Then create the category part
        if categories is None:
            categories = {1: "Illustration", 2: "Text", 3: "ScienceText"}
        self._set_category_annotation(categories)

    def _get_img_id_by_name(self, name):
        pass

    def _set_category_annotation(self, category_dict):
        """
        Expect category_dict as {1: "Illustration", 4: "Text", 12: "ScienceText"}
        """
        category_list = []
        for key, value in category_dict.items():
            category = {
                "id": key,
                "name": value,
                "supercategory": "",
            }
            category_list.append(category)
        self.coco_format["categories"] = category_list

    def add_an_image_annotation(self, file_name, width, height, image_id):
        """
        Add image information inside the coco format
        """
        image = {
            "file_name": file_name,
            "height": height,
            "width": width,
            "id": image_id
        }
        self.coco_format["images"].append(image)

    # This can be add during the extraction phase.
    def add_an_annotation_format(self, bbox, segmentation, image_id, category_id, annotation_id, polyon_area=None,
                                 score=1):
        area = polyon_area
        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": segmentation,
            "area": area,
            "bbox": bbox,
            "score": score,
            "iscrowd": 0
        }
        self.coco_format["annotations"].append(annotation)

    def save(self, path_to_file_name, only_ann=False):
        data = self.coco_format["annotations"] if only_ann else self.coco_format
        with open(path_to_file_name, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, sort_keys=False, indent=4, ensure_ascii=False, cls=NpEncoder)


def coerce_to_path_and_create_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

class MyMapper(DatasetMapper):
    """Mapper which uses `detectron2.data.transforms` augmentations"""

    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        augs = utils.build_augmentation(cfg, is_train)
        if is_train:
            # augs.append(T.RandomCrop_CategoryAreaConstraint(crop_type="relative_range", crop_size=(0.2, 0.7)))
            # augs.append(T.RandomRotation(angle=[-2, 2]))
            # augs.append(T.RandomSaturation(intensity_min=0.5, intensity_max=1.5))
            # augs.append(T.RandomBrightness(intensity_min=0.5, intensity_max=1.5))
            # augs.append(T.RandomContrast(intensity_min=0.5, intensity_max=1.5))
            pass
        if cfg.INPUT.CROP.ENABLED and is_train:
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = True

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret


class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()

        return losses

    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)


class MyTrainer(DefaultTrainer):

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg, mapper=MyMapper(cfg, True)
        )

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(
            cfg, dataset_name, mapper=MyMapper(cfg, True)
        )

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator(dataset_name, cfg)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                MyMapper(self.cfg, True)
            )
        ))
        return hooks


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

import yaml
import wandb
def main(args):
    cfg = setup(args)
    cfg_wandb = yaml.safe_load(cfg.dump())
    run = wandb.init(project="doc_layout", name="centermask", config=cfg, sync_tensorboard=True)
    _PREDEFINED_SPLITS_DCU = {
        "DCU_test": (args.dataset_images_path, f"{args.dataset_annotation_path}/testing.json"),
        "DCU_train": (args.dataset_images_path, f"{args.dataset_annotation_path}/training.json"),
    }

    metadata_dcu = {
        "thing_classes": ['Illustration', 'Text', 'ScienceText']
    }

    # register datasets
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_DCU.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            metadata_dcu,
            json_file,
            image_root,
        )
    if args.eval_only:
        model = MyTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = MyTrainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(MyTrainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--dataset_annotation_path", type=str, default=None, help="path to the annotation file")
    parser.add_argument("--dataset_images_path", type=str, default=None, help="path to the images file")
    args = parser.parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
