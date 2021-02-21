# -*- coding:utf-8 -*-
###
# File: demo.py
# Created Date: Wednesday, May 13th 2020, 9:57:40 pm
# Author: yusnows
# -----
# Last Modified: Thu May 14 2020
# Modified By: yusnows
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
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from predictor import Predictor
# setup_logger()


def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.cfg_dir + args.cfg))
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has two class
    cfg.MODEL.WEIGHTS = args.model_url
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence   # set the testing threshold for this model
    cfg.INPUT.MIN_SIZE_TEST = args.min_size
    # cfg.TEST.AUG.ENABLED = args.tta
    cfg.freeze()
    return cfg


def parse_arg():
    parser = argparse.ArgumentParser("person falldown trainer")
    parser.add_argument("--input_type", type=int, default=0,
                        help="the input data type. 0: video, 1: camera, 2: images directory, 3: image file")
    parser.add_argument("--input", type=str, default="./testdata/test_0.mp4")
    parser.add_argument(
        "--res_dir", type=str, default=None,
        help="detection result output directory. if it is None or \"\",\
             it will not save the detection result, but visulize it by cv2")
    parser.add_argument("--cfg_dir", type=str, default="Misc/", help="the cfg directory, COCO_Detection or Misc")
    parser.add_argument("--cfg", type=str,
                        default="cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml", help="cfg file name")
    parser.add_argument("--model_url", type=str, default="./results", help="the trained model file")
    # parser.add_argument("--tta", action='store_true', default=False)
    parser.add_argument("--min_size", type=int, default=800, help="the size for image input")
    parser.add_argument("--confidence", type=float, default=0.82, help="detection confidence threshold")
    parser.add_argument("--parallel", action='store_true', default=False, help="if use multi gpus to predict")
    args = parser.parse_args()
    print(args)
    return args


def main_pred(args):
    mp.set_start_method("spawn", force=True)
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)
    demo = Predictor(cfg, parallel=args.parallel)

    if args.input_type == 0:  # video file
        video = cv2.VideoCapture(args.input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.input)
        if args.res_dir is not None and args.res_dir != "":
            os.makedirs(args.res_dir, exist_ok=True)
            output_fname = os.path.join(args.res_dir, basename)
            output_fname = os.path.splitext(output_fname)[0] + ".mp4"
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
        assert os.path.isfile(args.input)
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.res_dir is not None and args.res_dir != "":
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        video.release()
        if args.res_dir is not None and args.res_dir != "":
            output_file.release()
        else:
            cv2.destroyAllWindows()
    elif args.input_type == 1:  # webcam
        cam = cv2.VideoCapture(0)
        for vis in tqdm.tqdm(demo.run_on_video(cam)):
            cv2.namedWindow("person falldown detection", cv2.WINDOW_NORMAL)
            cv2.imshow("person falldown detection", vis)
            if cv2.waitKey(1) == 27:
                break  # esc to quit
        cv2.destroyAllWindows()
    elif args.input_type == 2 or args.input_type == 3:  # image directory
        if args.input_type == 2:
            datasets = glob.glob(os.path.join(args.input, "*.jpg"))
            datasets += glob.glob(os.path.join(args.input, "*.png"))
        else:
            datasets = [args.input]
        for path in tqdm.tqdm(datasets, disable=not (args.res_dir is not None and args.res_dir != "")):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            # logger.info(
            #     "{}: {} in {:.2f}s".format(
            #         path,
            #         "detected {} instances".format(len(predictions["instances"]))
            #         if "instances" in predictions
            #         else "finished",
            #         time.time() - start_time,
            #     )
            # )
            if args.res_dir is not None and args.res_dir != "":
                assert os.path.isdir(args.res_dir), args.res_dir
                out_filename = os.path.join(args.res_dir, os.path.basename(path))
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow("person falldown detection", cv2.WINDOW_NORMAL)
                cv2.imshow("person falldown detection", visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit


if __name__ == "__main__":
    args = parse_arg()
    os.makedirs(args.res_dir, exist_ok=True)
    main_pred(args)
