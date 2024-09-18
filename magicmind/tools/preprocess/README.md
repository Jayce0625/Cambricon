# MagicMind Preprocess Tool

## 模型数据预处理

如果需要对模型进行量化处理，可以使用当前脚本模块对依赖的数据集进行预处理操作来制作标定数据集。

> <a>Note:</a>
> * `cv`类模型量化之前需要对图像数据的进行预处理操作, 本示例以`imagenet`数据集为例展示操作流程
> * 用户可以参考此脚本，对其他种类模型数据进行预处理<sup>e.g. NLP类模型, 语音类模型等</sup>

## 安装依赖

```bash
pip install -r requirements.txt
```

## 模型配置

所有四类框架模型数据预处理参数配置通过config下面的`yaml`文件完成，比如`caffe`的模型配置 `config/caffe.yaml`:

```yaml
# model configuration for caffe.

# resnet50 预处理参数
resnet50:
  resize_size: [256, 256]
  center_crop_size: [224, 224]
  need_transpose: False
  need_normalize: True
  mean: [103.94,116.78,123.68]
  std: [1.0,1.0,1.0]

vgg16:
  resize_size: [256, 256]
  center_crop_size: [224, 224]
  need_transpose: False
  need_normalize: True
  mean: [103.939, 116.779, 123.68]
  std: [1.0,1.0,1.0]

...
```
<a>Note:</a>
* 具体参数配置含义参考`utils/image_process.py`
* 其它的模型预处理可以参考`./config/`目录的配置文件, 用户可以自行添加


## 执行预处理

`preprocess.py`是数据预处理入口python文件:
```bash
usage: preprocess.py [-h] -f STR -i DIR [-n INT] [-l DIR] -s DIR -m DIR

ImageNet Preprocess

optional arguments:
  -h, --help            show this help message and exit
  -f STR, --framework STR
                        framework to use (caffe/onnx/tensorflow/pytorch).
  -i DIR, --image_path DIR
                        path to dataset
  -n INT, --max_image_number INT
                        Max image number to preprocess (default: 100)
  -l DIR, --labels DIR  Path to labels.txt
  -s DIR, --save_path DIR
                        path to save dataset
  -m DIR, --model_name DIR
                        network e.g. resnet50
```


**Exmaple:**

```bash
mkdir -p /path/to/result
python ../tools/preprocess/preprocess.py --framework caffe --image_path /data/datasets/imagenet --save_path ./result --model_name resnet50 --max_image_number 50
```

输出:
```bash
Preprocess image(0): /data/datasets/imagenet/ILSVRC2012_val_00024108.JPEG
Preprocess image(1): /data/datasets/imagenet/ILSVRC2012_val_00024109.JPEG
Preprocess image(2): /data/datasets/imagenet/ILSVRC2012_val_00024110.JPEG
Preprocess image(3): /data/datasets/imagenet/ILSVRC2012_val_00024111.JPEG
Preprocess image(4): /data/datasets/imagenet/ILSVRC2012_val_00024112.JPEG
Preprocess image(5): /data/datasets/imagenet/ILSVRC2012_val_00024113.JPEG
Preprocess image(6): /data/datasets/imagenet/ILSVRC2012_val_00024114.JPEG
Preprocess image(7): /data/datasets/imagenet/ILSVRC2012_val_00024115.JPEG
Preprocess image(8): /data/datasets/imagenet/ILSVRC2012_val_00024116.JPEG
Preprocess image(9): /data/datasets/imagenet/ILSVRC2012_val_00024117.JPEG
Preprocess image(10): /data/datasets/imagenet/ILSVRC2012_val_00024118.JPEG
Preprocess image(11): /data/datasets/imagenet/ILSVRC2012_val_00024119.JPEG
....
Preprocess image(40): /data/datasets/imagenet/ILSVRC2012_val_00019416.JPEG
Preprocess image(41): /data/datasets/imagenet/ILSVRC2012_val_00019417.JPEG
Preprocess image(42): /data/datasets/imagenet/ILSVRC2012_val_00024388.JPEG
Preprocess image(43): /data/datasets/imagenet/ILSVRC2012_val_00024389.JPEG
Preprocess image(44): /data/datasets/imagenet/ILSVRC2012_val_00024390.JPEG
Preprocess image(45): /data/datasets/imagenet/ILSVRC2012_val_00024391.JPEG
Preprocess image(46): /data/datasets/imagenet/ILSVRC2012_val_00024392.JPEG
Preprocess image(47): /data/datasets/imagenet/ILSVRC2012_val_00024393.JPEG
Preprocess image(48): /data/datasets/imagenet/ILSVRC2012_val_00024394.JPEG
Preprocess image(49): /data/datasets/imagenet/ILSVRC2012_val_00024395.JPEG
Calibration data:
   \__ file_list:  ./result/file_list
   \__ calibration_data_path:  ./result
```

其中:
* `file_list`文件包含所有已经处理过的标定数据集文件列表，其格式为filename shape[a,b,c,d]用来做量化标定使用
* `calibration_data_path` 表示输出标定数据集目录
