# MagicMind Plugin Crop usage example

Table Of Contents
- [内容描述](#description)
- [工作原理](#how-does-this-sample-work)
- [编译运行](#compiling-and-running-the-sample)

## 内容描述

sample_plugin_crop展示了MagicMind如何根据网络中使用到的Plugin算子对用户提供的Plugin静态库裁剪, 序列化到模型并运行的过程。目前MagicMind搭建的模型中有部分算子是通过用户自定义实现的，自定义实现的算子需要用户提供对应的Plugin算子库，然而当自定义算子较多且模型仅用到了部分自定义算子时，将全部的Plugin算子库加载进来会造成不少主机侧和设备侧资源的浪费。为了优化这一现象，MagicMind提供根据模型，定制化裁剪Plugin算子库的功能，它通过编译期筛选出模型中用到的Plugin算子，进而确定用到的PLUGIN API，对静态库文件libplugin.a进行裁剪，并序列化到模型中。
当前目录结构包括Plugin网络搭建及运行。（sample_plugin_crop）

## 工作原理

本示例展示了使用MagicMind Plugin裁减功能对用户提供的静态库文件进行裁剪的具体步骤：
1. 用户需要保证编译出的Plugin静态库中目标文件(.o文件)的命名是以PluginName_xxx.o的形式。如果存在common文件存储跨算子的公共接口，需要以common_xxx.o的形式摆放。本sample的实施如下:
- 首先，对Plugin算子库中的源文件按照如下的目录组织，组织形式如下：
```bash
tree plugin_kernels/
├── PluginCropAndResize
│   ├── README.md
│   ├── kernel.mlu
│   ├── plugin_crop_and_resize.cc
│   ├── plugin_crop_and_resize.h
│   └── plugin_crop_and_resize_kernel.h
├── PluginResizeYuvToRgba
│   ├── README.md
│   ├── kernel.mlu
│   ├── plugin_resize_yuv_to_rgba.cc
│   ├── plugin_resize_yuv_to_rgba.h
│   ├── plugin_resize_yuv_to_rgba_helper.cc
│   ├── plugin_resize_yuv_to_rgba_helper.h
│   ├── plugin_resize_yuv_to_rgba_kernel.h
│   ├── plugin_resize_yuv_to_rgba_macro.h
│   ├── plugin_resize_yuv_to_rgba_mlu200_kernel.h
│   └── plugin_resize_yuv_to_rgba_mlu300_kernel.h
└── PluginSpatialTransform
    ├── README.md
    ├── kernel_spatial_transform.h
    ├── plugin_spatial_transform_kernel.mlu
    ├── plugin_spatial_transform_op.cc
    └── plugin_spatial_transform_op.h
```

- 其次，在使用cmake编译时，需要将目标文件编译为PluginName_xxx.o的形式。其中，需要用到Bang cmake中的bang_wrap_srcs target，该target的使用格式为bang_wrap_srcs(目标文件名前缀 目标文件格式 存储目标文件的cmake变量 源文件), 以上述CropAndResize算子为例，目标文件名前缀为算子名称即PluginCropAndResize，目标文件格式为OBJ，源文件为PluginCropAndResize目录下的.cc或者.mlu文件。具体CMakeList命令详见samples/cc/basic_samples/CMakeLists.txt中Plugin算子编译部分。

- 最后，将目标文件编译为静态库文件。

2. 用户在编译网络前传入Plugin算子库的路径用于裁剪，支持绝对路径或者相对路径。 
  ```c++
  INetwork* network = CreateINetwork();
  IBuilder *builder = CreateIBuilder();
  IBuilderConfig* config = CreateIBuilderConfig();
  config->ParseFromString(R"(
{
  "crop_config": {
    "mtp_372": {
      "plugin": {
        "lib_path": "./libmagicmind_plugin_static.a"
      }
    }
  }
})");
  IModel *model = builder->BuildModel("test_model", network, config);
  ```
3. 在编译期，筛选网络中用到的Plugin算子，通过运行BuildModel生成模型的同时并将网络中用到的Plugin算子序列化到模型中。

## 编译运行

执行下列命令或使用一键运行脚本以编译并执行本测例：
```bash
samples/cc/basic_samples/build.sh -f plugin_crop #编译plugin_crop示例
cd samples/cc/basic_samples/build
bash run_plugin_crop.sh
```
