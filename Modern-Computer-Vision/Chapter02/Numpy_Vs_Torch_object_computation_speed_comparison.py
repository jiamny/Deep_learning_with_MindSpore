

import mindspore as ms
import numpy as np
import mindspore.context as context

ms.set_seed(10)
context.set_context(device_target="GPU")
device = context.get_context('device_target')
print(device)

assert device == 'GPU', "This exercise assumes the notebook is on a GPU machine"

from timeit import timeit

def func():
    x = ms.ops.rand(1, 6400)
    y = ms.ops.rand(6400, 5000)
    z = (x@y)


print('Running on GPU:')
t = timeit('func()', 'from __main__ import func', number=1)
print(t)

context.set_context(device_target="CPU")
print('Running on CPU:')
t = timeit('func()', 'from __main__ import func', number=1)
print(t)


def func():
    x = np.random.random((1, 6400))
    y = np.random.random((6400, 5000))
    z = np.matmul(x,y)

    
print('Running on CPU and use numpy:')
t = timeit('func()', 'from __main__ import func', number=1)
print(t)
