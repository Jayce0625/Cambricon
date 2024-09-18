# MagicMind C++ Samples

## Contents
| 目录结构                | 内容描述                                                                                |
|---|---|
| basic_samples           | MagicMind C++ API基本使用方式的代码示例，具体示例内容见其内部的README.md                |
| common                  | 调用MagicMind接口过程中常用的简单工具封装，具体见其内部的README.md及代码注释            |
| mm_build                | 以框架模型为输入，以MagicMind编译模型为输出的大型示例/工具，具体内容见其内部的README.md |
| mm_run                  | 以MagicMind离线文件为输入，高效执行/分析性能的大型示例/工具，具体内容见其内部的README.md|
| tools                   | 可以直接编译使用的简单工具封装，具体内容见其内部的README.md                             |
| third_party             | 示例使用的开源第三方代码，包括json11和half_float，内容及licence见文件头                 |
| CMakeSampleTemplate.txt | 编译示例代码所需的基本CMake配置模板                                                     |
| crosscompile.cmake      | 编译边缘侧示例代码所需的基本CMake配置模板                                               |
| build_run_all.sh        | 编译并运行当前目录下所有测例的脚本文件，具体配置见脚本注释或--help                      |
| build_template.sh       | 编译mm_build，mm_run，tools，third_party的模板脚本，命令为build_template.sh dir         |
| README.md               | 对目录结构和功能作用的描述文件                                                          |

## Notes

  - Sample的运行依赖于当前环境下寒武纪工具链的安装，具体见每个sample对应的描述文件。
  - 对框架模型的解析需要针对不同框架提供支持版本的模型和必需参数，具体模型支持版本见《寒武纪MagicMind用户手册》导入框架模型章节。
  - 执行一键运行脚本build_run_all需要将对应的`model`和`data`路径拷贝到当前目录下，并根据数据集调整sample_calibration内的sample_labels.txt：
  ```bash
  model/
  |--ResNet50_train_val_merge_bn.prototxt
  |--ResNet50_train_val_merge_bn.caffemodel
  |--resnet50-v1-7.onnx"
  |--py3.6.12_resnet50_v1.pb
  `--py3.7.9_torch1.6.0_resnet50.pt
  data/
  |-- ILSVRC2012_val_00000001.JPEG
  |-- ILSVRC2012_val_00000002.JPEG
  |-- ILSVRC2012_val_00000003.JPEG
  |-- ILSVRC2012_val_00000004.JPEG
  |-- ILSVRC2012_val_00000005.JPEG
  ...
  ```































































