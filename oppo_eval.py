# -*- coding:utf-8 -*-
###
# File: oppo_eval.py
# Created Date: Saturday, March 28th 2020, 9:26:03 pm
# Author: yusnows
# -----
# Last Modified:
# Modified By:
# -----
# Copyright (c) 2020 yusnows
#
# All shall be well and all shall be well and all manner of things shall be well.
# Nope...we're doomed!
# -----
# HISTORY:
# Date      	By	Comments
# ----------	---	----------------------------------------------------------
###

import os
import json
import glob
import argparse
import numpy as np
from collections import OrderedDict, defaultdict
import torch
import PIL.Image as Image
import matplotlib.pyplot as plt
import detectron2
import detectron2.utils.comm as comm
from detectron2.evaluation import verify_results
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2 import model_zoo
from detectron2.structures import BoxMode
from detectron2.utils.logger import setup_logger
setup_logger()


def get_person_dict(data_dir):
    cdir = os.getcwd()
    os.chdir(data_dir)
    jnames = glob.glob("*.json")
    os.chdir(cdir)
    dataset_dicts = []
    for jname in jnames:
        jfile = os.path.join(data_dir, jname)
        with open(jfile) as f:
            im_anns = json.load(f)
        record = {}
        # imname = im_anns["imagePath"]
        imname = jname[:-4] + "jpg"
        imfile = os.path.join(data_dir, imname)
        record["file_name"] = imfile
        record["image_id"] = imname
        record["height"] = im_anns["imageHeight"]
        record["width"] = im_anns["imageWidth"]
        objs = []
        shapes = im_anns["shapes"]
        for shape in shapes:
            obj = {
                "bbox": [shape["points"][0][0], shape["points"][0][1], shape["points"][1][0], shape["points"][1][1]],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": int(shape["label"]),
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop. You can use
    "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        # logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        # logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def main():
    parser = argparse.ArgumentParser("person falldown trainer")
    parser.add_argument("--datapath", type=str, default="./data")
    parser.add_argument("--res_base_dir", type=str, default="oppo_eval/")
    parser.add_argument("--cfg_dir", type=str, default="Misc/")
    parser.add_argument("--cfg", type=str, default="cascade_mask_rcnn_R_50_FPN_3x.yaml")
    parser.add_argument(
        "--min_sizes_test", nargs='+', type=int,
        default=[300, 340, 380, 420, 480, 512, 560, 600, 640, 700, 760, 800,
                 840, 900, 960, 1000, 1080, 1140, 1200])
    parser.add_argument("--model_url", type=str, default="")
    parser.add_argument("--tta", action='store_true', default=False)
    args = parser.parse_args()
    DatasetCatalog.register("pfallcnt_"+args.datapath, lambda d=args.datapath: get_person_dict(d))
    MetadataCatalog.get("pfallcnt_"+args.datapath).set(thing_classes=["0", "1"])
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.cfg_dir+args.cfg))
    cfg.DATASETS.TRAIN = ()
    cfg.DATASETS.TEST = ("pfallcnt_"+args.datapath,)
    cfg.CUDNN_BENCHMARK = True
    cfg.DATALOADER.NUM_WORKERS = 2
    # Let training initialize from model zoo
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.cfg_dir+args.cfg)
    if args.model_url is not None and args.model_url != "":
        cfg.MODEL.WEIGHTS = args.model_url
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has two class
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
    cfg.INPUT.MAX_SIZE_TEST = 10000
    cfg.TEST.AUG.ENABLED = args.tta
    dataset_dict = get_person_dict(args.datapath)
    fall_cnt_dict_gt = defaultdict(int)
    for data_dict in dataset_dict:
        annos = data_dict["annotations"]
        for anno in annos:
            fall_cnt_dict_gt[data_dict["image_id"]] += anno["category_id"]

    res_dict = {}
    for min_size in args.min_sizes_test:
        cfg.INPUT.MIN_SIZE_TEST = min_size
        cfg.OUTPUT_DIR = os.path.join(args.res_base_dir, str(min_size))
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=True
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        del model
        with open(os.path.join(args.res_base_dir, str(min_size)+"/inference/coco_instances_results.json")) as f:
            coco_obj_dicts = json.load(f)
        fall_cnt_dict = defaultdict(int)
        for obj_dict in coco_obj_dicts:
            fall_cnt_dict[obj_dict["image_id"]] += obj_dict["category_id"]

        mse = 0
        for key in fall_cnt_dict.keys():
            mse += (fall_cnt_dict[key] - fall_cnt_dict_gt[key])**2
        mse = mse/len(fall_cnt_dict)
        res["mse"] = mse
        res_dict[min_size] = res
    with open(os.path.join(args.res_base_dir, "res_dict.json"), 'w') as f:
        json.dump(res_dict, f, indent=1)


if __name__ == "__main__":
    main()
