# -*- coding:utf-8 -*-
###
# File: video_pred.py
# Created Date: Monday, April 27th 2020, 6:23:26 pm
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
from detectron2.utils.video_visualizer import VideoVisualizer
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


def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.cfg_dir+args.cfg))
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has two class
    cfg.MODEL.WEIGHTS = args.model_url
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence   # set the testing threshold for this model
    cfg.INPUT.MIN_SIZE_TEST = args.min_size
    # cfg.TEST.AUG.ENABLED = args.tta
    return cfg


def parse_arg():
    parser = argparse.ArgumentParser("person falldown trainer")
    parser.add_argument("--video_file", type=str, default="./testdata/test_0.mp4")
    parser.add_argument("--res_dir", type=str, default=None)
    parser.add_argument("--cfg_dir", type=str, default="Misc/")
    parser.add_argument("--cfg", type=str, default="cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml")
    parser.add_argument("--model_url", type=str, default="./results")
    parser.add_argument("--tta", action='store_true', default=False)
    parser.add_argument("--min_size", type=int, default=800)
    parser.add_argument("--confidence", type=float, default=0.85)
    # parser.add_argument("--save_bbox", action="store_true", default=False)
    # parser.add_argument("--bbox_dir", type=str, default="./bbox_out/")
    # parser.add_argument("--model2_url", type=str, default=None)
    # parser.add_argument("--cfg2_dir", type=str, default="COCO-Detection/")
    # parser.add_argument("--cfg2", type=str, default="faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    args = parser.parse_args()
    print(args)
    return args


def process_predictions(frame, predictions, person_metadata):
    predictions['instances'] = predictions['instances'].to('cpu')
    indices = predictions['instances'].pred_classes == 1
    predictions['instances'] = predictions['instances'][indices]
    if(len(predictions['instances']) == 0):
        return frame
    v = Visualizer(frame[:, :, ::-1], metadata=person_metadata, instance_mode=ColorMode.SEGMENTATION)
    v = v.draw_instance_predictions(predictions["instances"])
    # Converts Matplotlib RGB format to OpenCV BGR format
    vis_frame = cv2.cvtColor(v.get_image(), cv2.COLOR_RGB2BGR)
    return vis_frame


def main_pred(cfg, args):
    DatasetCatalog.register("pfallcnt_"+args.video_file, lambda d: [])
    MetadataCatalog.get("pfallcnt_" + args.video_file).set(
        thing_classes=["0", "1"], thing_colors=[(0, 255, 0), (255, 0, 0)])
    predictor = DefaultPredictor(cfg)
    video = cv2.VideoCapture(args.video_file)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    basename = os.path.basename(args.video_file)
    if args.res_dir is not None and args.res_dir != "":
        os.makedirs(args.res_dir, exist_ok=True)
        output_fname = os.path.join(args.res_dir, basename)
        output_fname = os.path.splitext(output_fname)[0] + ".mp4"
        assert not os.path.isfile(output_fname), output_fname
        assert not os.path.isfile(output_fname), output_fname
        output_file = cv2.VideoWriter(
            filename=output_fname,
            # some installation of opencv may not support x264 (due to its license),
            # you can try other format (e.g. MPEG)
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=float(frames_per_second),
            frameSize=(width, height),
            isColor=True,
        )
    person_metadata = MetadataCatalog.get("pfallcnt_" + args.video_file)
    # v = VideoVisualizer(metadata=person_metadata, instance_mode=ColorMode.IMAGE)
    cnt = 0
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        if cnt > 0 and (cnt % 20 == 0):
            print("frame: %d" % cnt)
        cnt += 1
        vis_frame = process_predictions(frame, predictor(frame), person_metadata)
        if args.res_dir is not None and args.res_dir != "":
            output_file.write(vis_frame)

    print("prediction on video end")
    video.release()
    if args.res_dir is not None and args.res_dir != "":
        output_file.release()


def main():
    args = parse_arg()
    cfg = setup_cfg(args)
    main_pred(cfg, args)


if __name__ == "__main__":
    main()
