# MagicMind API usage example based on plugin

Table Of Contents
- [内容描述](#description)
- [工作原理](#how-does-this-sample-work)
- [编译运行](#compiling-and-running-the-sample)

## 内容描述

sample_pluginop展示了MagicMind如何通过调用IPluginNode系列API，使用自己编写的算子构建网络片段的例子。具体实现指导见《寒武纪MagicMind用户手册》自定义算子章节。当前目录结构包括kernel实现（sample_pluginop/plugin_kernels/plugin_[op_name]）与算子编译和cpu真值（sample_pluginop/samples/plugin_[op_name]）。

## 工作原理

本示例基于MagicMind C++ API，展示用户自定义Node插入网络运行的能力：
测例展示的阶段包括（以下代码仅供举例）：
1. 实现BANG C kernel （kernel_relu.h/kernel_relu.mlu）
  编写BANG C代码
  ```c++
  #define BANG_NRAM_SIZE 1024 * 128
  template <typename T>
  __mlu_entry__ void VectorReLUBlock(const T *input, T *output, const uint32_t count) {
    ...
  }
  void ReluEnqueue(cnrtDim3_t dim,
                   cnrtFunctionType_t ktype,
                   cnrtQueue_t queue,
                   float *input_addr,
                   float *output_addr,
                   uint32_t count) {
    VectorReLUBlock<<<dim, ktype, queue>>>(input_addr, output_addr, count);
  }
  ```
2. 生成链接
  ```bash
  cncc --bang-arch=compute_20 --bang-arch=compute_30 --bang-wram-align64 -c ${PLUGIN_DIR}/kernel_relu.mlu -o kernel_relu.o -fPIC
  ```
3. 添加并使用自定义算子
  - 注册自定义算子。
    ```c++
    //实现形状推导函数
    Status DoShapeInfer(IShapeInferResource *context) {
      Status ret;
      std::vector<int64_t> input_shape;
      ret = context->GetShape("input", &input_shape);
      if (ret != Status::OK()) {
        return ret;
      }
      ret = context->SetShape("output", input_shape);
      return ret;
    }
    //注册算子
    PLUGIN_REGISTER_OP("PluginReLU")
        .Input("input")
        .TypeConstraint("T")
        .Output("output")
        .TypeConstraint("T")
        .Param("T")
        .Type("type")
        .Allowed({magicmind::DataType::FLOAT32, magicmind::DataType::FLOAT16})
        .Param("Dim")
        .TypeList("int")
        .Default(std::vector<int64_t>{1, 1, 1})
        .Param("FuncType")
        .Type("int")
        .Default(1)
        .ShapeFn(DoShapeInfer);
    ```
  - 实现算子kernel。（plugin_relu.cc plugin_relu.h）
    ```c++
    class PluginReLUKernel : public magicmind::IPluginKernel {
     public:
      magicmind::Status SetLocalVar(magicmind::INodeResource *context) override;
      size_t GetWorkspaceSize(magicmind::INodeResource *context) override;
      magicmind::Status Enqueue(magicmind::INodeResource *context) override;
      ~PluginReLUKernel(){};
     private:
      uint64_t input_count_ = 1;
      void *input_addr_     = nullptr;
      void *output_addr_    = nullptr;
      //...
    };
    ```
  - 注册算子kernel（plugin_relu.h）
    ```c++
    class PluginReLUKernelFactory : public magicmind::IPluginKernelFactory {
     public:
      // rewrite create
      magicmind::IPluginKernel *Create() override { return new PluginReLUKernel(); }
      ~PluginReLUKernelFactory() {}
    };
    // register pluginop
    namespace magicmind {
    PLUGIN_REGISTER_KERNEL(CreatePluginKernelDefBuilder("PluginReLU").DeviceType("MLU"),
                           PluginReLUKernelFactory);
    }  // namespace magicmind
    ```
  - 使用自定义算子（sample_pluginop.cc）
    ```c++
    auto conv_output = conv->GetOutput(0);
    // add relu pluginop
    magicmind::TensorMap plugin_inputs;
    std::vector<magicmind::ITensor *> input{conv_output};
    plugin_inputs["input"] = input;
    magicmind::DataTypeMap plugin_outputs_dtype;
    plugin_outputs_dtype["output"] = {output_datatype};
    magicmind::IPluginNode *plugin_relu =
    network->AddIPluginNode("PluginReLU", plugin_inputs, plugin_outputs_dtype);
    plugin_relu->SetAttr("Dim", std::vector<int64_t>{1, 1, 1});
    plugin_relu->SetAttr("FuncType", (int64_t)1);  // BLOCK
    // set outputs nodes
    network->MarkOutput(plugin_relu->GetOutput(0));
    ```
- 使用dlopen调用自定义算子
  - 编译生成动态链接库
    ```bash
    add_library(plugin SHARED plugin_relu.cc kernel_relu.o)
    ```
  - 用户进行网络构建或者执行离线模型时，可以编译时链接libplugin.so并且引用plugin_relu.h，也可以调用dlopen。
    ```c++
      auto kernel_lib = dlopen("libplugin.so", RTLD_LAZY);
      //...
      ConstructConvReluNetwork(...);
      dlclose(kernel_lib);
    ```

## 编译运行

执行下列命令或使用一键运行脚本以编译并执行本测例：
```bash
samples/basic_samples/build.sh -f plugin #编译全部plugin算子及示例
samples/basic_samples/build.sh -f plugin --plugin_filter [op_name] #编译指定算子及示例
cd samples/basic_samples/build
./build/sample_[op_name]
```
该指令会在build/目录中生成libmagicmind_plugin.so，运行测例并输出文件`[op_name]_model`，如需运行本示例生成的网络部署文件，可以使用示例中的run.sh（需要同时编译sample_runtime与mm_run），或使用运行时示例sample_runtime调用MagicMind运行时，或使用进阶示例MagicMind模型部署工具mm_run，其命令参照run.sh。
实际使用需要根据算子实际使用场景更改示例中的输入规模，数据类型和参数设置等。

## 端侧推理

本示例可通过交叉编译在有MLU卡的端侧设备执行。

用户可以通过交叉编译的方式编译出可在端侧运行的模型推理应用程序,
交叉编译依赖交叉编译工具链和端侧对应的MagicMind边缘侧运行时及其所需依赖位于NEUWARE_HOME/edge目录下，cncc位于NEUWARE_HOME/bin目录下。

可按下述步骤执行生成程序：

```bash
export TOOLCHAIN_ROOT=/path/to/cross_compile_tools_dir #默认为`/usr/local/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu/`
samples/basic_samples/build.sh --cpu_arch aarch64
cd samples/basic_samples/build
```
build目录下即生成可在aarch64边缘侧设备使用的libmagicmind_plugin.so（不包含编译示例）。
