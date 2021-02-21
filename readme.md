![](readme.assets/dianti_1.gif)

## 代码说明

### 1. 代码总体情况

本项目代码基于[detectron2](https://github.com/facebookresearch/detectron2) 进行的二次开发. 因此运行该项目的前提是先安装[detectron2](https://github.com/facebookresearch/detectron2), 但是由于我们对[detectron2](https://github.com/facebookresearch/detectron2) 的代码功能做了一些小的改动, 所以需要按照下面的步骤安装.

### 2. 安装项目依赖

#### 2.1 detectron2 的依赖安装

- Linux with Python ≥ 3.6
- PyTorch ≥ 1.3
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation. You can install them together at [pytorch.org](https://pytorch.org/) to make sure of this.
- OpenCV, optional, needed by demo and visualization
- pycocotools: `pip install cython; pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'`

#### 2.2 本项目修改版 detectron2 安装

```bash
# 进入项目主文件夹 person-fallcnt
# 然后执行以下命令
$ cd detectron2
$ pip install -e .
# 如果上面命令遇到权限问题, 加上--user
```

如果安装遇到错, 可查看[install.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) ，或者联系我们团队进行解决。

### 3. 代码运行

#### 训练

训练程序接口文件为`pfallcnt_train.py` 。训练方式和参数如下示例。

```bash
# 训练 faster_rcnn_R_50_FPN
$ python3 pfallcnt_train.py --datapth path_to_the_training_set --outdir path_to_training_output --max_iters 200000
# 训练 faster_rcnn_X_101_32x8d_FPN 模型
$ python3 pfallcnt_train.py --cfg faster_rcnn_X_101_32x8d_FPN_3x.yaml --datapath path_to_the_training_set --outdir path_to_training_output --max_iters 200000
# 训练 cascade_mask_rcnn_x_152_FPN
$ python3 pfallcnt_train.py --cfg_dir Misc/ --cfg cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml --datapath path_to_the_training_set --outdir path_to_training_output --max_iters 200000
```
#### demo运行

测试程序接口文件为 `demo.py` ，运行方式如下示例。

```bash
# 用单个模型测试
$ python3 demo.py --input_type 0(or 1 or 2 or 3) --cfg_dir COCO-Detection/(or Misc/) --cfg model_config_file --res_dir path_to_testing_result(can be None or "", if None or "". it requires opencv and desktop enviroment) --model_url model_file --min_size image_size --input input_file_or_directory
#########################################################################################
# 比如: 用cascade_mask_rcnn_X_152_FPN 对视频进行预测
$ python3 demo.py --cfg_dir Misc --cfg cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml --res_dir results --model_url mmodels/cascade_x152_model.pth --input test_data/test_0.mp4
# 比如: 用 faster_rcnn_X_101_32x8d_FPN 进行预测
$ python3 demo.py --cfg faster_rcnn_X_101_32x8d_FPN_3x.yaml --res_dir results --model_url models/faster_x101_model.pth --input test_data/test_0.mp4
# 比如: 用cascade_mask_rcnn_X_152_FPN 对某个文件夹下面的图片进行预测
$ python3 demo.py --cfg_dir Misc --cfg cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml --res_dir results --model_url mmodels/cascade_x152_model.pth --input_type 2 --input test_data/images/
# 比如: 用cascade_mask_rcnn_X_152_FPN 对摄像头视频流进行预测，此时只能在桌面环境进行可视化，不支持保存到文件。
$ python3 demo.py --cfg_dir Misc --cfg cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml --model_url mmodels/cascade_x152_model.pth --input_type 1
```

更多参数的细节，请使用`--help`参数获得。

注1: 由于对demo程序的测试没有桌面环境，因此无法测试直接可视化的过程（即只测试了把检测结果保存到文件的部分），因此无法保证可视化部分一定可以顺利运行，如果遇到问题，请联系解决。

注2: 由于模型参数文件过大, 因此提交的文件并不包含训练好的模型.  需要可以联系发送。

#### 3.1 训练数据格式

训练集文件夹里面应该包含图片数据和bbox标注数据, bbox标注数据问json文件, 图片为jpg格式.

bbox标注数据文件格式如下:

```json
{
  "version": "4.2.7",
  "flags": {},
  "shapes": [
    {
      "gound_id": null,
      "label": "1",
      "points": [
        [
          340.9351501464844,
          147.46168518066406
        ],
        [
          538.90576171875,
          214.04153442382812
        ]
      ],
      "group_id": null,
      "shape_type": "rectangle",
      "flags": {}
    },
    {
      "gound_id": null,
      "label": "0",
      "points": [
        [
          186.7875518798828,
          36.57759094238281
        ],
        [
          216.6598358154297,
          110.41905212402344
        ]
      ],
      "group_id": null,
      "shape_type": "rectangle",
      "flags": {}
    }
  ],
  "imagePath": "./tr-1.jpg",
  "imageData": null,
  "imageHeight": 302,
  "imageWidth": 550,
  "flag": {}
}
```

