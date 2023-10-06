import mindspore as ms
from mindspore import context, Tensor, nn
import numpy as np
ms.set_seed(42)
context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU", device_id=0)

class MeanSquareLoss:
    def __init__(self): pass

    def loss(self, y, y_pred):
        return ms.ops.sum(ms.ops.power((y - y_pred), 2),dim=1) / y.shape[0]

    def gradient(self, y, y_pred):
        return -(y - y_pred)

class CrossEntropy:
    def __init__(self): pass

    def loss(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * ms.ops.log(p) - (1 - y) * ms.ops.log(1 - p)

    def gradient(self, y, p):
        # Avoid division by zero
        p = ms.ops.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)

class MeanAbsoluteLoss:
    def __init__(self): pass

    def loss(self, y, y_pred):
        return ms.ops.sum(ms.ops.abs(y - y_pred), dim=1) / y.shape[0]

    def gradient(self, y, y_pred):
        return -(y - y_pred)

class HuberLoss:
    def __init__(self):pass

    def loss(self, y, y_pred, delta):
        if ms.ops.abs(y - y_pred) <=delta:
            return 0.5 * ms.ops.pow(y - y_pred, 2)
        else:
            return (delta * ms.ops.abs(y - y_pred)) - (0.5 * ms.ops.pow(delta, 2))

class HingeLoss:
    def __init__(self):
        pass

    def loss(self, y, y_pred):
        if float(((1-y) * y_pred).item()) < 1.0:
            return (1-y) * y_pred
        else:
            return Tensor(0.0, dtype=ms.float32)

class KLDivergence:
    def __init__(self):
        pass

    def loss(self, y, y_pred):
        return ms.ops.sum(y_pred * ms.ops.log((y_pred / y)))


y =  ms.ops.randint(1, 5, size=(8, 100)).astype(ms.float32)
y_pred = ms.ops.randint(1, 5, size=(8, 100)).astype(ms.float32)

kl = KLDivergence()
loss = kl.loss(y, y_pred)
print(loss)

y =  ms.ops.rand(1, dtype=ms.float32)
y_pred = ms.ops.rand(1, dtype=ms.float32)
hl = HingeLoss()
loss = hl.loss(y, y_pred)
print(loss)