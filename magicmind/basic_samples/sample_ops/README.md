# Builder API usage example

Table Of Contents
- [内容描述](#description)
- [工作原理](#how-does-this-sample-work)
- [编译运行](#compiling-and-running-the-sample)
- [注意](#attention)

## 内容描述

sample_ops展示了用户如何通过调用MagicMind Builder API接口构建一个单算子模型的例子。当前目录结构为sample_ops/sample_[op_name]。

## 工作原理

本示例基于MagicMind C++ API，展示如何构造一个算子，并进行运行部署前的编译工作，包括：
1. 创建输入tensors。
2. 构建网络。
3. 创建模型。
4. 序列化模型，生成`[op_name]_model`。

## 编译运行

执行下列命令或使用一键运行脚本以编译并执行本测例：
```bash
samples/basic_samples/build.sh -f ops #编译全部算子示例
cd samples/basic_samples/build
./sample_[op_name]
```
该指令会在build文件夹中运行测例并输出文件`[op_name]_model`，如需运行本示例生成的网络部署文件，可以使用示例中的run.sh（需要同时编译sample_runtime与mm_run），或使用运行时示例sample_runtime调用MagicMind运行时，或使用进阶示例MagicMind模型部署工具mm_run，其命令参照run.sh。
实际使用需要根据算子实际使用场景更改示例中的输入规模，数据类型和参数设置等。
