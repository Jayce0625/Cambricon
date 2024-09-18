# Builder API usage example

Table Of Contents
- [内容描述](#description)
- [工作原理](#how-does-this-sample-work)
- [编译运行](#compiling-and-running-the-sample)

## 内容描述

sample_quantization展示了用户如何通过调用MagicMind API接口构建网络、设置ranges、混合精度、编译模型的例子。

## 工作原理

本示例基于MagicMind C++ API，展示由卷积算子、深度可分离卷积算子、激活算子构造的网络片段，设置ranges、混合精度，并进行运行部署前的编译工作，包括：
1. 构建网络。
2. 设置ranges。
3. 设置全局配置以及混合精度。
4. 创建并序列化名为`model_quantization`的二进制模型。

## 编译运行

执行下列命令或使用一键运行脚本以编译并执行本测例：
```bash
samples/basic_samples/build.sh -f quantization
cd samples/basic_samples/build
./sample_quantization
```
脚本会在build文件夹中运行示例并输出文件`model_quantization`（该模型不可变，输入的shape固定），如需运行本示例生成的网络部署文件，可以使用示例中的run.sh（需要同时编译sample_runtime与mm_run），或使用运行时示例sample_runtime调用MagicMind运行时，或使用进阶示例MagicMind模型部署工具mm_run，其命令参照run.sh。
