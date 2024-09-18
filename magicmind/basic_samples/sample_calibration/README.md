# Calibration API usage example based on Caffe parser

Table Of Contents
- [内容描述](#description)
- [工作原理](#how-does-this-sample-work)
- [编译运行](#compiling-and-running-the-sample)
- [端侧推理](#remote-running-on-edge-sample)

## 内容描述

sample_calibration是基于Caffe模型，向用户展示如何使用MagicMind的API进行模型解析、量化校准、编译的例子。

## 工作原理

本示例基于MagicMind C++ API，展示模型解析到运行的整体流程，包括：
1. 基于Caffe模型解析并生成INetwork。
2. 基于校准数据集进行量化校准。
3. 用户选择量化类型，包括对称量化和非对此量化，默认为对称量化。
4. 配置并构建网络。
5. 序列化模型，生成`model_quant`。

## 编译运行

运行本示例前，用户需自行配置对应的人工智能框架模型和数据集，寒武纪提供了部分可用于benchmark的网络模型/数据集列表，见《寒武纪MagicMind Benchmark指南》`模型和数据集`章节。
以ResNet50举例：
1. 使用的数据集为：ILSVRC2012_img_val.tar，可在imagenet官网下载或者联系寒武纪技术支持人员获取。
  图片列表如下：
  ILSVRC2012_val_00000001.JPEG
  ILSVRC2012_val_00000002.JPEG
  ...
  ILSVRC2012_val_00001000.JPEG
2. 从官网下载ResNet50网络权重和模型。
3. 示例提供了简单的数据预处理模块在目录`samples/tools/preprocess`下，入口文件为 `samples/tools/preprocess/preprocess.py`，用户可以直接调用当前文件来对`imagenet`数据集进行处理，具体使用方式见`samples/tools/preprocess/README.md`。
假定标定数据集处理后存储目录为`result`, 存储目录中的`file_list`包含所有标定数据集文件名，预处理后的数据可以用于量化校准。
sample_calibration文件夹内预先提供了1000张图片的sample_labels.txt，请用户自行根据实际运行数据集调整。

4. 执行calibration：
  ```bash
  samples/basic_samples/build.sh -f calibration
  cd samples/basic_samples/build
  ./sample_calibration --data_path ./path/to/output/ \
                       --label_path ./path/to/sample_labels.txt \
                       --prototxt_path ./path/to/model/resnet50.prototxt  \
                       --caffemodel_path ./path/to/model/resnet50.caffemodel
  或远端执行
  ./remote_server &
  ./sample_calibration --data_path ./path/to/output/ \
                       --label_path ./path/to/sample_labels.txt \
                       --prototxt_path ./path/to/model/resnet50.prototxt  \
                       --caffemodel_path ./path/to/model/resnet50.caffemodel \
                       --rpc_server address:port
  ```
脚本会在build文件夹中运行示例并输出文件`model_quant`，如需运行本示例生成的网络部署文件，可以使用示例中的run.sh（需要同时编译sample_runtime与mm_run），或使用运行时示例sample_runtime调用MagicMind运行时，或使用进阶示例MagicMind模型部署工具mm_run，其命令参照run.sh。
