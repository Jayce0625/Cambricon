# MagicMind C++ Diff Tool

## 数据精度对比

对给定的真值及对比数据，使用本工具对数据进行四类公式对比，并检测是否符合某阈值

## 编译运行

```bash
bash samples/build_template.sh samples/tools/diff
```
编译产物位于samples/tools/diff/build/diff_compare

## 命令行参数

```bash
diff_compare --help
```
显示命令行语法与具体备选输入。

| 参数名称   | 是否必需 | 输入格式            | 参数描述      | 注意事项           |
|---|---|---|---|---|
| data       | 是       | --data path/to/data | 对比数据地址  | 数据为二进制格式   |
| baseline   | 是       | --data path/to/data | 真值数据地址  | 数据为二进制格式   |
| datatype   | 是       | --datatype dtype    | 数据格式 | 无 |
| threshold1 | 否       | --threshold1 float  | 公式1对比阈值 | 不满足阈值会返回-1 |
| threshold2 | 否       | --threshold2 float  | 公式2对比阈值 | 不满足阈值会返回-1 |
| threshold3 | 否       | --threshold3 float1,float2 | 公式3对比阈值 | 不满足阈值会返回-1 |
| threshold4 | 否       | --threshold4 float  | 公式4对比阈值 | 不满足阈值会返回-1 |

## 运行示例

```bash
diff_compare --data path/to/data --baseline path/to/base --datatype float --threshold3  0.2,0.3
```
默认或输入阈值NaN为不进行此项对比。

## 公式及阈值说明

公式1：

$$
diff1 = {{\sum {|data_{eval}-data_{base}|}} \over {\sum {| data_{base} |}}}
$$

阈值代表绝对误差之和不超过原始数据的某一百分比。

公式2：

$$
diff2 =\sqrt {{\sum {{\left( data_{eval} - data_{base} \right)}^{2}}} \over {\sum {{data_{base}} ^ {2}}}}
$$

阈值代表误差平方和不超过原始数据平方和某一百分比。

公式3：

$$
diff3_1 = \max {{| {data_{eval}}_{i} - {data_{base}}_{i} |} \over {|  {data_{base}} _ {i} | }}\\
diff3_2 = \max {| {data_{eval}}_{i} - {data_{base}}_{i} |}
$$

阈值代表逐点最大相对误差与逐点最大绝对误差各自不超过某一值。

公式4：

$$
diff4_1 = {{count_1} \over {n}}\\
diff4_2 = {{count_2} \over {n}}
$$

count1代表逐点大于真值的个数，count2代表逐点小于真值的个数，n代表全部不相等总点数。

公式4返回在所有逐点不相等的值中，大于真值和小于真值的比例。阈值代表逐点不相等的数字不超过全部真值某一比例。

