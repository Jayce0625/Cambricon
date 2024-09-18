# Runtime API usage example

Table Of Contents
- [内容描述](#description)
- [工作原理](#how-does-this-sample-work)
- [编译运行](#compiling-and-running-the-sample)

## 内容描述

sample_runtime展示如何调用MagidMind Runtime API交叉使用多卡/自定义内存分配器/性能数据采集/远端执行/拷出中间结果特性完成模型推理。

## 工作原理

本示例基于MagicMind C++ API，展示运行时如下特性交叉使用的方法：

1. 多卡执行

   - 使用用户指定的设备号，初始化并构造多个IEngine实例。

2. 自定义内存分配器

   - 用户传入内存分配器IAllocator，MagicMind所申请的设备内存都通过该IAllocator对象。IAllocator需要在IEngine释放之后再释放。
   - 用户通过查询静态内存和运行时所需最大内存大小，调用寒武纪底层接口（比如cnrtMalloc）分配出对应大小内存一次性提供给IEngine
     和IContext使用。此时用户如果尝试另外配置IAllocator对象，IAllocator将不起作用。

3. 性能数据采集

   - 配置Profiler并在运行时输出性能数据。

   - 读入Profiler输出并进行可视化。

4. 远端执行

   - 进行远程部署，远程服务器发起Server服务。
   - 本地发送请求在远程服务器上完成部署及推理全过程。
   - 获取运行结果，结束Server。

5. 拷出中间结果

   - 运行的过程中指定一个数据dump目录。
   - 输出中间结果至目录。（如是远端执行会传输回本地）。

测例展示的阶段包括：

1. 读入模型文件，构造运行时模型IModel实例。
2. 初始化并构造IEngine实例。
3. 使用IEngine实例，和模型中存储的给定输入规模，实例化IContext并运行。

## 命令行参数

```bash
sample_runtime --help
```

显示命令行与具体备选输入。

| 参数名称    | 是否必需 | 输入格式                      | 参数描述               | 注意事项                                                     |
| ----------- | -------- | ----------------------------- | ---------------------- | ------------------------------------------------------------ |
| model_path  | 是       | --model_path path/to/model    | 模型路径               | -                                                            |
| data_path   | 否       | --data_path input1 input2...  | 输入路径组             | 输入为二进制数据，默认使用随机数据                           |
| dev_ids     | 否       | --dev_ids card1 card2...      | MLU设备id号            | 默认为0                                                      |
| input_dims  | 是       | --input_dims input1 input2... | 输入形状组             | -                                                            |
| rpc_server  | 否       | --rpc_server remoteaddr       | rpc地址                | 默认为空，边缘侧不能使用。                                   |
| mem_stg     | 否       | --mem_stg static/dynamic      | 内存分配策略           | 默认为空，静态为用户使用最大Workspace分配中间结果静态地址，动态为用户外挂内存分配器动态分配 |
| profile     | 否       | --profile 0/1/True/False      | 是否使用性能数据采集   | 默认为否，性能数据输出目录为运行目录下`./sample_profiler`    |
| dump        | 否       | --dump 0/1/True/False         | 是否拷出中间结果       | 默认为否，中间结果输出目录为运行目录下`./sample_dump`        |
| plugin_libs | 否       | --plugin_libs lib1 lib2...    | 用户自定义算子库路径组 | 默认为空，不支持与RPC功能混用。                              |
| output_caps | 否       | --output_caps output_cap1 output_cap2...  | 输出容量组，以字节为单位  | 默认为空，不支持与RPC功能混用。  |
| threads     | 否       | --threads num                 | 执行线程数         | 指定每个设备以多少线程并发进行推理，每个线程创建一个IContext实例。  |
| visible_cluster| 否    | --visible_cluster cluster1 cluster2... | IContext实例绑定的cluster_ids，从0开始 | 默认为空，不绑定cluster_id，开启后为每个IContext实例绑定特定的cluster_ids。绑定的cluster_ids数量需要与并发的线程数一致，不支持与RPC功能混用。  |


## 编译运行

运行本示例前需要一份编译好的网络部署文件`model`，用户可以自行生成，或者使用sample中的任何一个Builder API的测例来产生。

注意本测例的运行输入基于一个输入规模无未知量模型，用户可以自行调整代码结构来适应一个输入规模不确定的模型，详细情况请参照《寒武纪MagicMind用户手册》C++ API章节。

执行下列命令或使用一键运行脚本以编译并执行本测例：
```bash
samples/basic_samples/build.sh -f runtime #编译示例及远端server
cd samples/basic_samples/build
./sample_runtime
```
该指令会在build文件夹中读取模型并运行测例，也可以使用示例中的run.sh（需要同时编译sample_runtime与sample_calibration）。

如使用性能可视化功能，需要安装寒武纪CNLight提供的TensorBoard python3.x wheel安装包:

```bash
pip install tensorboard-<x.y.z>-py3-none-any.whl
```

其中<x.y.z>为CNLight提供的TensorBoard对应的TensorBoard官方版本号。详细信息请参考《寒武纪CNLight用户手册》中"TensorBoard安装"章节。

安装TensorBoard后，可视化命令如下：

```bash
port=6688  # for example
tensorboard --logdir path/to/profiling_output --port $port --bind_all # 开启服务
```
浏览器中打开`http://$ip:$port`显示profiling可视化页面（由于使用TensorBoard的动态插件加载模式，启动服务后第一次打开浏览器页面请在右上角页面选项卡处手动切换PROFILE选项卡以显示profiling可视化页面）。

如使用RPC远端执行功能，需在设备侧另一进程启动server。

```bash
samples/basic_samples/build.sh -f runtime #编译示例及远端server
cd samples/basic_samples/build
./remote_server
```

## 端侧推理

本示例可通过交叉编译在有MLU卡的端侧设备执行。

用户可以通过交叉编译的方式编译出可在端侧运行的模型推理应用程序,
交叉编译依赖交叉编译工具链和端侧对应的MagicMind边缘侧运行时及其所需依赖位于NEUWARE_HOME/edge目录下。

可按下述步骤执行生成程序：

```bash
export TOOLCHAIN_ROOT=/path/to/cross_compile_tools_dir #默认为`/usr/local/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu/`
samples/basic_samples/build.sh --cpu_arch aarch64
cd samples/basic_samples/build
```
build目录下即生成可在aarch64边缘侧设备运行的sample_runtime及remote_server。
