# MagicMind API usage of PluginSpatialTransform Operation

Table Of Contents
- [算子简介](#Brief-introduction-of-operation)
- [参数说明](#How-to-use-this-op)

## 算子简介

基于仿射矩阵，对输入的灰度图数据进行仿射变换。

## 参数说明

本节对算子接口中的参数进行详细说明。需要注意的是，尽管在接口能力上算子支持了FLOAT32类型的输入数据，但是由于完成算子运行的BangC Kenrel仅支持FLOAT16数据，FLOAT32的输入数据被装换为FLOAT16类型进行运算。因此算子的精度可能会和预期不同。可以参考demo目录下的示例代码进行算子的添加与运行。

### 输入说明

算子有三个输入张量，分别为：

- input: 代表输入灰度图的张量。维度数量必须为4，Layout可以是NCHW或者NHWC。张量的H维度必须是40，W维度必须是180，C维度必须是1。数据类型可以是FLOAT16或者FLOAT32。

- mat: 代表仿射矩阵的张量。维度至少为2，Layout必须是ARRAY。每个batch的元素数量必须是6。张量的N维度，也即第一个维度必须和另两个输入张量相等或者为1。当mat张量的N维度为1且和另两个输入的N维度不一致时，此时mat张量会在内部进行广播至N维度相等。数据类型可以是FLOAT16或者FLOAT32。

- multable_value：代表仿射矩阵中变量的张量。Layout必须是ARRAY。每个batch的元素数量必须是2。multable_value和mat共同组成完成的仿射矩阵。算子执行过程中，mat张量每个batch的6个数据中的第一个数据以及第五个数据会依次被multable_value张量相应batch的第一个和第二个元素替换。

### 输出说明

算子有一个输入张量，为：

- output: 代表输出灰度图的张量。张量形状和数据类型必须和input张量相同。

### 平台限制

该算子支持370-S4, 370-X4和3226
