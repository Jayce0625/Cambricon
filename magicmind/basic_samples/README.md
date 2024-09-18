# MagicMind Samples

## Contents
| 目录结构   | 内容描述 |
|---|---|
| sample_calibration    | 基于Caffe模型，向用户展示如何使用MagicMind的API进行模型解析、量化校准、编译的例子。|
| sample_quantization   | 展示如何通过调用MagicMind API接口构建网络、设置ranges、混合精度、编译模型的例子。|
| sample_network_modify | 展示如何通过调用相关API变更原始网络模型。|
| sample_ops            | 展示如何通过调用MagicMind Builder API接口构建一个单算子模型。|
| sample_pluginop       | 展示如何通过调用MagicMind IPluginNode系列API，使用自定义算子构建网络片段。具体实现指导见《寒武纪MagicMind用户手册》自定义算子章节。|
| sample_refit          | 展示如何在运行期进行refit，动态修改模型weight且不影响模型正常推理。|
| sample_runtime        | 展示如何调用MagidMind Runtime API交叉使用多卡/自定义内存分配器/Profiler/远端执行/Dump中间结果特性完成模型推理。|
| sample_plugin_crop    | 展示如何通过在config中设置plugin的lib_path，实现对plugin算子的裁剪功能。|

## Note

- 具体介绍详见各sample目录下的README.md，执行脚本见各sample目录下的run.sh

- 编译脚本见：
```bash
samples/basic_samples/build.sh
```
编译脚本在basic_samples/build生成编译产物，并且将各个目录下的运行脚本run.sh重命名为run_[sample_name].sh拷入build文件夹，其参数通过
```bash
samples/basic_samples/build.sh -h/--help
```
查看，可以具体选择编译的sample对象，目标arch，是否为debug模式等。
若需编译边缘侧的pluginop及runtime示例，使用脚本命令
```bash
samples/basic_samples/build.sh --cpu_arch aarch64
```
编译时应保证NEUWARE_HOME/edge文件夹下的依赖完好，包含MagicMind边缘侧运行时及其所需依赖。

- 一键运行脚本见：
```bash
samples/build_run_all.sh
```
