# Builder API usage example based on add

Table Of Contents
- [内容描述](#description)
- [工作原理](#how-does-this-sample-work)
- [编译运行](#compiling-and-running-the-sample)

## 内容描述

sample_network_modify展示了用户如何通过调用MagicMind Builder API接口变更已有网络模型的例子。

## 工作原理

本示例基于MagicMind C++ API，展示如何更改网络模型，并进行运行部署前的编译工作，包括：
1. 构建原始网络。
2. 更改原始网络。
3. 创建模型。
4. 序列化模型，生成`model_modify`。

## 编译运行

执行下列命令或使用一键运行脚本以编译并执行本测例：
```bash
samples/basic_samples/build.sh -f network_modify
cd samples/basic_samples/build
./sample_network_modify
```
脚本会在build文件夹中运行示例并输出文件`model_modify`，如需运行本示例生成的网络部署文件，可以使用示例中的run.sh（需要同时编译sample_runtime与mm_run），或使用运行时示例sample_runtime调用MagicMind运行时，或使用进阶示例MagicMind模型部署工具mm_run，其命令参照run.sh。
