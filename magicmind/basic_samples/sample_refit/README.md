# Refitter API usage example

Table Of Contents
- [内容描述](#description)
- [工作原理](#how-does-this-sample-work)
- [编译运行](#compiling-and-running-the-sample)

## 内容描述

sample_refit 展示了使用 MagicMind Refitter API 在运行期动态更新模型Weight的方法。

## 工作原理

Reffiter 可以在不中断线上服务的同时对模型的 Weight 进行动态更新，且更新过程具备原子性，
推理结果或者是由旧模型得到，或者是由新模型得到。

Refitter API 自身较为简洁明了，本示例主要通过模拟一个简单的 Server 来体现 Refitter 在并行环境中动态修改模型 Weight 的能力。

本示例基于 MagicMind C++ API，测例展示的内容包括：
1. 创建一个简单的SimpleConvRelu模型，并为需要动态更新的模型参数预先进行命名，以便后续进行Refit。
2. 开启若干RunEngine线程负责进行Engine的创建及动态更新，每个Engine将会对应至多个Worker工作线程。
3. 先使用初始模型进行推理，一段时间后，在不中断Worker线程的同时，对Engine并行动态Refit。

## 编译运行

执行下列命令或使用一键运行脚本以编译并执行本测例：
```bash
samples/basic_samples/build.sh -f refit
cd samples/basic_samples/build
./sample_refit
```
或者使用示例中的run.sh。
如果运行成功，那么终端会输出类似下面展示的内容:

```bash
Worker 2 started.
Worker 3 started.
Worker 0 started.
Worker 1 started.
Worker 2 get different output at 0
Worker 3 get different output at 0
Worker 0 get different output at 0
Worker 1 get different output at 0
Engine 1 weights updating:  prod_bias_tensor prod_b_tensor conv0_filter conv0_bias
Engine 0 weights updating:  prod_bias_tensor prod_b_tensor conv0_filter conv0_bias
Worker 3 get different output at 963
Worker 1 get different output at 935
Worker 2 get different output at 1010
Worker 0 get different output at 864
Worker 2 stopped.
Worker 0 stopped.
Worker 3 stopped.
Worker 1 stopped.
```
