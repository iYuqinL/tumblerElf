# -*- coding:utf-8 -*-
###
# File: pfallcnt_train.py
# Created Date: Sunday, February 2nd 2020, 8:30:37 pm
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
import torch
import PIL.Image as Image
import matplotlib.pyplot as plt
import detectron2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, DefaultTrainer
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


def main():
    parser = argparse.ArgumentParser("person falldown trainer")
    parser.add_argument("--datapath", type=str, default="./data")
    parser.add_argument("--outdir", type=str, default="./output/")
    parser.add_argument("--cfg_dir", type=str, default="Misc/")
    parser.add_argument("--cfg", type=str, default="cascade_mask_rcnn_R_50_FPN_3x.yaml")
    parser.add_argument("--max_iters", type=int, default="10000")
    parser.add_argument("--model_url", type=str, default="")
    parser.add_argument("--resume_from", action='store_true', default=False)
    args = parser.parse_args()
    DatasetCatalog.register("pfallcnt_"+args.datapath, lambda d=args.datapath: get_person_dict(d))
    MetadataCatalog.get("pfallcnt_"+args.datapath).set(thing_classes=["notfalldown", "falldown"])
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.cfg_dir+args.cfg))
    cfg.DATASETS.TRAIN = ("pfallcnt_"+args.datapath,)
    cfg.DATASETS.TEST = ()
    cfg.CUDNN_BENCHMARK = True
    cfg.DATALOADER.NUM_WORKERS = 2
    # Let training initialize from model zoo
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.cfg_dir+args.cfg)
    if args.model_url is not None and args.model_url != "":
        cfg.MODEL.WEIGHTS = args.model_url
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.000333  # pick a good LR
    cfg.MODEL.MASK_ON = False
    # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
    cfg.SOLVER.MAX_ITER = args.max_iters
    cfg.SOLVER.CHECKPOINT_PERIOD = 10000
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has two class
    #cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.MAX_SIZE_TRAIN = 1560
    cfg.INPUT.MIN_SIZE_TRAIN = (
        300, 320, 360, 400, 480, 512, 564, 580, 600, 640, 672, 704, 736, 768, 800, 842, 874,
        904, 960,)  # 1000, ) #1080, )  # 1140, 1200, 1240, 1300, 1360)
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = 'choice'
    if args.resume_from:
        cfg.MODEL.WEIGHTS = ""

    cfg.OUTPUT_DIR = args.outdir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    if (args.model_url is None or args.model_url == "") or args.resume_from:
        trainer.resume_or_load(resume=True)
    else:
        print("load the model you give")
        trainer.checkpointer._load_model(torch.load(args.model_url))
    trainer.train()


if __name__ == "__main__":
    main()
