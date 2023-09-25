
# 基于Mindspore构造Avg Pooling

# 导入numpy处理数据
import numpy as np
# 导入数据类型包
from mindspore import dtype as mstype
# 导入多维数组数据结构
from mindspore import Tensor
from mindspore import context, set_seed
set_seed(0)
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")


### 数据加载

np.random.seed(0)
x = np.random.randint(0, 10, [1, 2, 4, 4])
x = Tensor(x, mstype.float32)
print("输入数据x: \n", x)
print("\nx.shape:", x.shape)

## 构建模型
### 使用示例

# 导入功能模块nn
import mindspore.nn as nn

# 声明一个二维池化实体
pool = nn.AvgPool2d(kernel_size=3, stride=1)
# 执行池化功能并将结果返回
output = pool(x)
# 输出池化后的数据
print("池化后数据：\n", output)
# 输出池化后数组的形状
print("\n池化后数据的形状：\n",output.shape)

### 使用Mindspore官方定义的基类

# 导入网络层函数
from mindspore.ops import AvgPool
# 导入构造函数算子包
from mindspore.ops import constexpr
# 导入神经网络基本单元包
from mindspore.nn import Cell
# 导入MindSpore
import mindspore as ms
class _PoolNd(Cell):

    def __init__(self, kernel_size, stride, pad_mode, data_format="NCHW"):
        super(_PoolNd, self).__init__()         
        # 检查pad_mode是否为VALID或SAME
        if pad_mode != 'VALID' and pad_mode != 'SAME':
            raise ValueError('The pad_mode must be VALID or SAME')
        self.pad_mode = pad_mode
        # 检查data_format是否为NCHW或NHWC
        if data_format !='NCHW' and data_format != 'NHWC':
            raise ValueError('The format must be NCHW or NHWC')          
        self.format = data_format
        # NHWC数据格式仅支持GPU
        if ms.get_context("device_target") != "GPU" and self.format == "NHWC":
            raise ValueError(f"For '{self.cls_name}, the 'NHWC' format only support in GPU target, but got device "    
                             f"target {ms.get_context('device_target')}.")

        # 检查是否为int或tuple，且必须为正数
        def _check_int_or_tuple(arg_name, arg_value):                                                                  
            error_msg = f"For '{self.cls_name}', the '{arg_name}' must be an positive int number or " \
                        f"a tuple of two positive int numbers, but got {arg_value}"
            if isinstance(arg_value, int):
                if arg_value <= 0:
                    raise ValueError(error_msg)
            elif isinstance(arg_value, tuple):
                if len(arg_value) == 2:
                    for item in arg_value:
                        if isinstance(item, int) and item > 0:
                            continue
                        raise ValueError(error_msg)
                else:
                    raise ValueError(error_msg)
            else:
                raise ValueError(error_msg)
            return arg_value
        
        # kernel_size是一个正数或两个正数的元组
        self.kernel_size = _check_int_or_tuple('kernel_size', kernel_size)  
        # stride是一个正数或两个正数的元组
        self.stride = _check_int_or_tuple('stride', stride)                           

    def construct(self, *inputs):
        pass

    def extend_repr(self):
        return 'kernel_size={kernel_size}, stride={stride}, pad_mode={pad_mode}'.format(**self.__dict__)


# 继承基类实现二维平均池化
class AvgPool2d(_PoolNd):
    def __init__(self,
                 kernel_size=1,    # 卷积核大小为1
                 stride=1,         # 步长为1
                 pad_mode="VALID",
                 data_format="NCHW"):
        # 初始化二维平均池化.
        super(AvgPool2d, self).__init__(kernel_size, stride, pad_mode, data_format)
        self.avg_pool = AvgPool(kernel_size=self.kernel_size,
                                  strides=self.stride,
                                  pad_mode=self.pad_mode,
                                  data_format=self.format)
        
    # 构造函数
    def construct(self, x):
        return self.avg_pool(x)

## 模型测试

# 声明一个二维池化实体
pool = AvgPool2d(kernel_size=3, stride=1)
# 对输入数据进行平均池化并返回结果
output = pool(x)
print("输出数据为\n", output)
print("\n输出数据的形状\n", output.shape)
exit(0)