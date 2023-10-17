
import mindspore as ms
from mindspore import context, Tensor, nn
from copy import deepcopy
import matplotlib.pyplot as plt

context.set_context(device_target="GPU")

x = ms.Tensor([[1,1]], dtype=ms.float32)
y = ms.Tensor([[0]], dtype=ms.float32)
print(x)

def feed_forward(inputs, outputs, weights):     
    pre_hidden = ms.ops.dot( inputs, weights[0]) + weights[1]
    hidden = 1/(1+ms.ops.exp(-pre_hidden))
    out = ms.ops.dot(hidden, weights[2]) + weights[3]
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
                    grad = (_loss_plus - original_loss)/(0.0001)
                    updated_weights[i][r, c] -= grad*lr
        else:
            row = layer.shape[0]
            for r in range(row):
                temp_weights = deepcopy(weights)
                temp_weights[i][r] += 0.0001
                _loss_plus = feed_forward(inputs, outputs, temp_weights)
                grad = (_loss_plus - original_loss)/(0.0001)
                updated_weights[i][r] -= grad*lr
    return updated_weights, original_loss

W = [
    ms.Tensor([[-0.0053, 0.3793],
              [-0.5820, -0.5204],
              [-0.2723, 0.1896]], dtype=ms.float32).transpose(),
    ms.Tensor([-0.0140, 0.5607, -0.0628], dtype=ms.float32),
    ms.Tensor([[ 0.1528, -0.1745, -0.1135]], dtype=ms.float32).T,
    ms.Tensor([-0.5516], dtype=ms.float32)
]

losses = []
for epoch in range(100):
    W, loss = update_weights(x, y, W, 0.01)
    losses.append(loss)
plt.plot(losses)
plt.title('Loss over increasing number of epochs')
plt.show()

print('W: ', W)
pre_hidden = ms.ops.dot(x,W[0]) + W[1]
hidden = 1/(1+ms.ops.exp(-pre_hidden))
out = ms.ops.dot(hidden, W[2]) + W[3]
print('out: ', out)

