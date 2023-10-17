from copy import deepcopy
import mindspore as ms
from mindspore import context, Tensor, nn
import numpy as np

x = Tensor([[1], [2], [3], [4]], dtype=ms.float32)
y = Tensor([[3], [6], [9], [12]], dtype=ms.float32)


def feed_forward(inputs, outputs, weights):
    out = ms.ops.dot(inputs, weights[0]) + weights[1]
    mean_squared_error = ms.ops.mean(ms.ops.square(out - outputs))
    return mean_squared_error


def update_weights(inputs, outputs, weights, lr):
    original_weights = deepcopy(weights)
    temp_weights = deepcopy(weights)
    updated_weights = deepcopy(weights)
    original_loss = feed_forward(inputs, outputs, original_weights)

    for i, layer in enumerate(original_weights):
        if len(layer.shape) > 1:
            row, col = layer.shape
            for r in range(row):
                for c in range(col):
                    temp_weights = deepcopy(weights)
                    temp_weights[i][r, c] += 0.0001
                    _loss_plus = feed_forward(inputs, outputs, temp_weights)
                    grad = (_loss_plus - original_loss) / (0.0001)
                    updated_weights[i][r, c] -= grad * lr
        else:
            row = layer.shape[0]
            for r in range(row):
                temp_weights = deepcopy(weights)
                temp_weights[i][r] += 0.0001
                _loss_plus = feed_forward(inputs, outputs, temp_weights)
                grad = (_loss_plus - original_loss) / (0.0001)
                updated_weights[i][r] -= grad * lr
    return updated_weights


W = [Tensor([[0]], dtype=ms.float32), Tensor([[0]], dtype=ms.float32)]
weight_value = []
for epx in range(1000):
    W = update_weights(x, y, W, 0.01)
    weight_value.append(W[0][0][0])

import matplotlib.pyplot as plt

plt.plot(weight_value)
plt.title('Weight value over increasing epochs')
plt.xlabel('Epochs')
plt.ylabel('Weight value')
plt.show()

W = [Tensor([[0]], dtype=ms.float32), Tensor([[0]], dtype=ms.float32)]
weight_value = []
for epx in range(1000):
    W = update_weights(x, y, W, 0.1)
    weight_value.append(W[0][0][0])

plt.plot(weight_value)
plt.title('Weight value over increasing epochs with learning rate of 0.1')
plt.xlabel('Epochs')
plt.ylabel('Weight value')
plt.show()

W = [Tensor([[0]], dtype=ms.float32), Tensor([[0]], dtype=ms.float32)]
weight_value = []
for epx in range(1000):
    W = update_weights(x, y, W, 1)
    weight_value.append(W[0][0][0])

plt.plot(weight_value)
plt.title('Weight value over increasing epochs with learning rate of 1')
plt.xlabel('Epochs')
plt.ylabel('Weight value')
plt.show()


def update_weights(inputs, outputs, weights, lr):
    original_weights = deepcopy(weights)
    temp_weights = deepcopy(weights)
    updated_weights = deepcopy(weights)
    original_loss = feed_forward(inputs, outputs, original_weights)

    for i, layer in enumerate(original_weights):
        if len(layer.shape) > 1:
            row, col = layer.shape
            for r in range(row):
                for c in range(col):
                    temp_weights = deepcopy(weights)
                    temp_weights[i][r, c] += 0.0001
                    _loss_plus = feed_forward(inputs, outputs, temp_weights)
                    grad = (_loss_plus - original_loss) / (0.0001)
                    updated_weights[i][r, c] -= grad * lr
                    if (i % 2 == 0):
                        if row == 1 and col == 1:
                            print('weight value:', np.round(original_weights[i][0].asnumpy(), 2),
                              'original loss:', np.round(original_loss.asnumpy(), 2),
                              'loss_plus:', np.round(_loss_plus.asnumpy(), 2),
                              'gradient:', np.round(grad.asnumpy(), 2),
                              'updated_weights:', np.round(updated_weights[i][r, c].asnumpy(), 2))
                        else:
                            print('weight value:', np.round(original_weights[i][r, c].asnumpy(), 2),
                                  'original loss:', np.round(original_loss.asnumpy(), 2),
                                  'loss_plus:', np.round(_loss_plus.asnumpy(), 2),
                                  'gradient:', np.round(grad.asnumpy(), 2),
                                  'updated_weights:', ms.ops.round(updated_weights[i][r, c].asnumpy(), 2))
        else:
            row = layer.shape[0]
            for r in range(row):
                temp_weights = deepcopy(weights)
                temp_weights[i][r] += 0.0001
                _loss_plus = feed_forward(inputs, outputs, temp_weights)
                grad = (_loss_plus - original_loss) / (0.0001)
                updated_weights[i][r] -= grad * lr
                if (i % 2 == 0):
                    print('weight value:', np.round(original_weights[i][r].asnumpy(), 2),
                          'original loss:', np.round(original_loss.asnumpy(), 2),
                          'loss_plus:', np.round(_loss_plus.asnumpy(), 2),
                          'gradient:', np.round(grad.asnumpy(), 2),
                          'updated_weights:', np.round(updated_weights[i][r].asnumpy(), 2))
    return updated_weights


W = [Tensor([[0]], dtype=ms.float32), Tensor([[0]], dtype=ms.float32)]
weight_value = []
for epx in range(10):
    W = update_weights(x, y, W, 0.01)
    weight_value.append(W[0][0][0])
print(W)

plt.plot(weight_value[:100])
plt.title('Weight value over increasing epochs when learning rate is 0.01')
plt.xlabel('Epochs')
plt.ylabel('Weight value')
plt.show()

W = [Tensor([[0]], dtype=ms.float32), Tensor([[0]], dtype=ms.float32)]
weight_value = []
for epx in range(10):
    W = update_weights(x, y, W, 0.1)
    weight_value.append(W[0][0][0])
print(W)

plt.plot(weight_value[:100])
plt.title('Weight value over increasing epochs when learning rate is 0.1')
plt.xlabel('Epochs')
plt.ylabel('Weight value')
plt.show()

W = [Tensor([[0]], dtype=ms.float32), Tensor([[0]], dtype=ms.float32)]
weight_value = []
for epx in range(10):
    W = update_weights(x, y, W, 1)
    weight_value.append(W[0][0][0])
print(W)

plt.plot(weight_value[:100])
plt.title('Weight value over increasing epochs when learning rate is 1')
plt.xlabel('Epochs')
plt.ylabel('Weight value')
plt.show()
