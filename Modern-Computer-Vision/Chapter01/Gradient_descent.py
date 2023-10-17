
import mindspore as ms
from copy import deepcopy

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


