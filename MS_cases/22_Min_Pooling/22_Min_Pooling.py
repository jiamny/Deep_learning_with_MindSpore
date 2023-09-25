
## 基于MindSpore构建Min Pooling层


from mindspore import Tensor  
from mindspore import dtype as mstype    
import numpy as np                          
np.random.seed(1)
from mindspore import context, set_seed
set_seed(0)
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")


# 1x2x4x4的数据
x = Tensor(np.random.randint(0, 10, [1, 2, 4, 4]), mstype.float32)       

#### 数据加载
print(x)

### 模型构建

import mindspore.nn as nn
from mindspore import ops
import mindspore
from mindspore.nn import Cell


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
        if mindspore.get_context("device_target") != "GPU" and self.format == "NHWC":
            raise ValueError(f"For '{self.cls_name}, the 'NHWC' format only support in GPU target, but got device "    
                             f"target {mindspore.get_context('device_target')}.")

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


class MinPool2d(_PoolNd):
    def __init__(self, kernel_size=1, stride=1, pad_mode="VALID", data_format="NCHW"):
        # 检查输入是否规范
        super(MinPool2d, self).__init__(kernel_size, stride, pad_mode, data_format) 
        # 最大池化
        self.max_pool = ops.MaxPool(kernel_size=self.kernel_size,                          
                                  strides=self.stride,
                                  pad_mode=self.pad_mode,
                                  data_format=self.format)



    def construct(self, x):
        # minpool=- maxpool(-data)
        out = -self.max_pool(-x)                                                         
        return out

### 模型测试

# 直接调用定义好的最小池化类
pool = MinPool2d(kernel_size=3, stride=1)                                                 
pool_max=nn.MaxPool2d(kernel_size=3, stride=1)
print(x)
# 最小池化
output = pool(x)                                                                          
print(output)
# 最大池化
print(pool_max(x))                                                                        
exit(0)