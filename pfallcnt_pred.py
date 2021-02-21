# -*- coding:utf-8 -*-
###
# File: pfallcnt_pred.py
# Created Date: Monday, February 3rd 2020, 9:30:43 am
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
import cv2
import numpy as np
import pandas as pd
import PIL.Image as Image
import matplotlib.pyplot as plt
import torch
import detectron2
import detectron2.data.transforms as T
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.modeling import build_model
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.checkpoint import DetectionCheckpointer
from detectron2 import model_zoo
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.logger import setup_logger
setup_logger()


def get_person_dict(data_dir):
    cdir = os.getcwd()
    os.chdir(data_dir)
    imnames = glob.glob("*.jpg")
    imnames += glob.glob("*.jpeg")
    imnames = sorted(imnames)
    os.chdir(cdir)
    dataset_dicts = []
    for imname in imnames:
        record = {}
        imfile = os.path.join(data_dir, imname)
        record["file_name"] = imfile
        record["image_id"] = imname
        width, height = Image.open(imfile).size
        record["height"] = height
        record["width"] = width
        dataset_dicts.append(record)
    return dataset_dicts


class PredictorWithTTA(object):
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)
        self.model = GeneralizedRCNNWithTTA(cfg, self.model)
        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, image):
        with torch.no_grad():
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                image = image[:, :, ::-1]
            height, width = image.shape[:2]
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs], merge_method='vote', same_obj_iou_threshold=0.65, vote_threshold=0.33)[0]
        return predictions


def main():
    parser = argparse.ArgumentParser("person falldown trainer")
    parser.add_argument("--datapath", type=str, default="./data")
    parser.add_argument("--res_dir", type=str, default="./res-2")
    parser.add_argument("--cfg_dir", type=str, default="COCO-Detection/")
    parser.add_argument("--cfg", type=str, default="faster_rcnn_R_50_FPN_3x.yaml")
    parser.add_argument("--model_url", type=str, default="./output")
    parser.add_argument("--tta", action='store_true', default=False)
    parser.add_argument("--min_size", type=int, default=800)
    parser.add_argument("--save_bbox", action="store_true", default=False)
    parser.add_argument("--bbox_dir", type=str, default="./bbox_out/")
    parser.add_argument("--model2_url", type=str, default=None)
    parser.add_argument("--cfg2_dir", type=str, default="COCO-Detection/")
    parser.add_argument("--cfg2", type=str, default="faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    args = parser.parse_args()
    if args.save_bbox:
        os.makedirs(args.bbox_dir, exist_ok=True)
    if args.cfg2 is not None and args.cfg2_dir is not None and args.model2_url is not None:
        has_model2 = True
    else:
        has_model2 = False
    DatasetCatalog.register("pfallcnt_" + args.datapath, lambda d=args.datapath: get_person_dict(d))
    MetadataCatalog.get("pfallcnt_" + args.datapath).set(
        thing_classes=["0", "1"], thing_colors=[(0, 255, 0), (255, 0, 0)])
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.cfg_dir+args.cfg))
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has two class
    cfg.MODEL.WEIGHTS = args.model_url
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.895   # set the testing threshold for this model
    cfg.DATASETS.TEST = ("pfallcnt_" + args.datapath)
    cfg.INPUT.MIN_SIZE_TEST = args.min_size
    cfg.TEST.AUG.ENABLED = args.tta

    if args.tta:
        #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.993
        predictor = PredictorWithTTA(cfg)
    else:
        predictor = DefaultPredictor(cfg)

    data_dicts = get_person_dict(args.datapath)
    person_metadata = MetadataCatalog.get("pfallcnt_" + args.datapath)
    os.makedirs(os.path.join(args.res_dir, "vis"), exist_ok=True)
    name_list = []
    fallcnts = []
    for d in data_dicts:
        im = cv2.imread(d["file_name"])
        height, width = im.shape[:2]
        outputs = predictor(im)
        field_dict = outputs["instances"].to("cpu").get_fields()
        pred_cls = field_dict["pred_classes"].numpy()
        name_list.append(d["image_id"])
        fallcnts.append(pred_cls.sum())
        print(name_list[-1], fallcnts[-1])
        if args.save_bbox:
            anno_dict = {"version": "4.2.7", "flag": {}, "shapes": [], "imagePath": "../A/"+d["image_id"],
                         "imageData": None, "imageHeight": height, "imageWidth": width}
            bboxes = field_dict["pred_boxes"].tensor.numpy().astype(np.float64)
            for i in range(bboxes.shape[0]):
                box_dict = {"label": str(int(pred_cls[i])), "gound_id": None, "shape_type": "rectangle", "flags": {}}
                box_dict["points"] = [[bboxes[i][0], bboxes[i][1]], [bboxes[i][2], bboxes[i][3]]]
                anno_dict["shapes"].append(box_dict)
            with open(os.path.join(args.bbox_dir, d["image_id"][:-3]+"json"), 'w') as f:
                json.dump(anno_dict, f, indent=1)
        v = Visualizer(im[:, :, ::-1], metadata=person_metadata, scale=1.0, instance_mode=ColorMode.SEGMENTATION)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        save_file = os.path.join(os.path.join(args.res_dir, "vis"), d["image_id"])
        cv2.imwrite(save_file, v.get_image()[:, :, ::-1])

    del predictor

    if has_model2:
        print("has model2, building model2 config")
        cfg2 = get_cfg()
        cfg2.merge_from_file(model_zoo.get_config_file(args.cfg2_dir+args.cfg2))
        cfg2.DATALOADER.NUM_WORKERS = 2
        cfg2.SOLVER.IMS_PER_BATCH = 2
        cfg2.MODEL.MASK_ON = False
        cfg2.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has two class
        cfg2.MODEL.WEIGHTS = args.model2_url
        cfg2.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.92   # set the testing threshold for this model
        cfg2.DATASETS.TEST = ("pfallcnt_" + args.datapath)
        cfg2.INPUT.MIN_SIZE_TEST = args.min_size
        cfg2.TEST.AUG.ENABLED = args.tta

        if args.tta:
            predictor2 = PredictorWithTTA(cfg2)
        else:
            predictor2 = DefaultPredictor(cfg2)
        idx = -1
        modifies = []
        for d in data_dicts:
            idx += 1
            im = cv2.imread(d["file_name"])
            outputs2 = predictor2(im)
            field_dict2 = outputs2["instances"].to("cpu").get_fields()
            pred_cls2 = field_dict2["pred_classes"].numpy()
            fallcnt2 = pred_cls2.sum()
            print("model 2: ", d["file_name"], fallcnt2)
            if fallcnt2 > 5 and fallcnts[idx] < fallcnt2 and name_list[idx] == d["image_id"]:
                fallcnts[idx] = fallcnt2
                modifies.append(name_list[idx])
            elif name_list[idx] != d["image_id"]:
                print("file name is not the same")

    csv_dict = {"file": name_list, "fall_count": fallcnts}
    csv_df = pd.DataFrame(csv_dict)
    csv_df.to_csv(os.path.join(args.res_dir, "fallcnt_submit.csv"), sep=",", index=False)
    print(len(name_list), len(fallcnts))
    if has_model2:
        print(modifies)
        print("modify %d results" % len(modifies))


if __name__ == "__main__":
    main()
