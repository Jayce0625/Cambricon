# MagicMind API usage of PluginResizeYuvToRgba Operation

Table Of Contents
- [算子简介](#Brief-introduction-of-operation)
- [参数说明](#How-to-use-this-op)

## 算子原理

对输入的YUV420SP图片进行色彩空间变换以及基于双线性插值算法的缩放。

## 用法说明

本节对算子接口中的参数进行详细说明。

### 输入说明

算子有六个输入张量，分别为：

- y_tensors: 代表输入的YUV图片Y通道数据的tensorlist。其中的每一个元素均为维度数量为4，数据类型为UINT8，数据布局为NHWC的张量。基于YUV图片的定义，张量的H、W维度必须是偶数，C维度必须为1。举例说明，输入维度为：{{1，1080，1920，1}，{5，720，1280，1}}，代表输入数据分别是一张1080P图片和五张720P的图片。tensorlist中的元素个数至少为1。

- uv_tensors: 代表输入的YUV图片UV通道数据的tensorlist。其中的每一个元素均为维度数量为4，数据类型为UINT8，数据布局为NHWC的张量。[uv_tensors]和[y_tensors]中包含的张量个数必须相等，且相对应（或者说在两个tensorlist中序号相等的）的tensor分别代表同一张（组）YUV420SP格式图片的数据。基于YUV420SP格式的定义，UV通道的数据量为Y通道数据量的一半（4像素Y共用一像素UV）。因此UV通道的形状表示方式可有多种，仅需满足数据量为相应的Y通道张量的一半且N维度相等即可。举例说明，{{1, 540, 1920, 1}, {5, 360, 540, 2}}可为[y_tensors]示例中相应的[uv_tensors]形状。需要注意的是，由于Y通道张量的H、W维度和输入的YUV图片高、宽大小一致，因此本算子仅使用[y_tensors]中张量的形状作为输入形状。UV通道张量的表示方式不唯一，因此仅做数量的检查。

- input_rois: 代表输入图片的Region(R) Of(O) Interest(I)的张量。算子会基于[input_rois]中的数据对输入图片进行裁剪。张量的维度数量为2，数据类型为INT32，数据布局为NC。其中N维度的大小必须等于[y_tensors]以及[uv_tensors]中所有张量N维度之和，C维度必须为4。例如按照[y_tensors]中的例子计算，[input_rois]的形状为{6, 4}，也即存在6组ROI数据，依次对应第一张1080P图片以及五张720P图片。C维度的4代表ROI数据roi_x(左上角横坐标)，roi_y(左上角纵坐标)，roi_w(roi的宽度)，以及roi_h(roi的高度)。需要特别注意的是，[input_rois]的数据存在于主机侧。

- output_rois: 代表输出图片的Region(R) Of(O) Interest(I)的张量。算子会基于[output_rois]中的数据对输入图片进行裁剪。张量的维度数量为2，数据类型为INT32，数据布局为NC。其中N维度的大小必须等于[y_tensors]以及[uv_tensors]中所有张量N维度之和，C维度必须为4。例如按照[y_tensors]中的例子计算，[input_rois]的形状为{6, 4}，也即存在6组ROI数据，依次对应第一张1080P图片以及五张720P图片。C维度的4代表ROI数据roi_x(左上角横坐标)，roi_y(左上角纵坐标)，roi_w(roi的宽度)，以及roi_h(roi的高度)。需要注意目前output_rois的数据必须等于输出图片的形状，也即roi_x = roi_y = 0, roi_w = 输出图片的W维度，roi_h = 输出图片的H维度。需要特别注意的是，[output_rois]的数据存在于主机侧。

- output_shapes: 代表输出图片形状的张量。算子的形状推导函数会基于[output_shapes]张量中的数据在运行时设置算子输出张量的形状，因此[output_shapes]的数据存在于主机侧。张量的维度数据为2，数据类型为INT32，数据布局为NC。其中N维度的大小必须为1，C维度大小必须为3，也即代表所有的输入图片均会resize成相同大小的输出图片。C维度的3依次代表输出张量的N维度，H维度，以及W维度。

- fiil_color: 代表对输出图片进行填充的数据张量。当缩放过程需要符合保持长宽比时，输出图片的大小和实际有效图片的大小可以不相等。此时需要对无效数据部分进行填充。fill_color张量可以为任意维度，但是元素数量必须为3或者4，代表输出图片中有效通道的填充值。举例说明，当输入图片形状为(1920, 1080)，输出图片形状为(720, 540)，且保持输入长宽比时，实际有效输出图片形状为(720, 405)。此时实际输出图片中的部分行需要进行填充。当输出图片格式为rgb或者bgr时，[fill_color]的元素数量为3，当输出图片格式带有a通道时，例如rgba，[fill_color]的元素数量为4。

### 输出说明

算子有一个输出张量，为：

- rgba_tensors: 代表输出的RGB类型图片的tensorlist。其中每一个元素均为维度数量必须为4，数据类型必须为UINT8，数据布局必须为NHWC的张量。目前由于算子的限制，tensorlist中张量个数必须为1，也即代表所有输入的图片会resize为相同的形状。张量的形状会在运行时通过[output_format]以及[output_shapes]中的数据确定。关于参数[output_foramt]的介绍，可以参考下一节。

### 参数说明

算子有三个输入参数，分别为：

- input_format: 代表输入图片的格式。1代表YUV420SP_NV12，2代表YUV420SP_NV21。

- output_format: 代表输出图片的格式。1代表RGB，2代表BGR，3代表RGBA，4代表BGRA，5代表ARGB，6代表ABGR。

- pad_method: 代表输出图片是否保持长宽比以及填充冗余部分的方法。0代表不保持长宽比，1代表保持长宽比并且有效图片位于实际图片的中央（也即在上下或者左右进行填充），2代表保持长宽比并且有效图片位于实际图片的左上角（也即在右侧或者下侧进行填充）。

### 平台限制

该算子支持370-S4, 370-X4和3226
