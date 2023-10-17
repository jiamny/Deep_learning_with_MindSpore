import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor
import mindspore.context as context
import matplotlib.pyplot as plt

ms.set_seed(10)
context.set_context(device_target="GPU")

X = Tensor([[1,2],[3,4],[5,6],[7,8]], dtype=ms.float32)
Y = Tensor([[3],[7],[11],[15]], dtype=ms.float32)

class MyNeuralNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.input_to_hidden_layer = nn.Dense(2,8)
        self.hidden_layer_activation = nn.ReLU()
        self.hidden_to_output_layer = nn.Dense(8,1)

    def construct(self, x):
        x = self.input_to_hidden_layer(x)
        x = self.hidden_layer_activation(x)
        x = self.hidden_to_output_layer(x)
        return x

mynet = MyNeuralNet()
loss_func = nn.MSELoss()
print('X: ', X)
_Y = mynet(X)
print('_Y ms: ', _Y)
loss_value = loss_func(_Y,Y)
print('loss_value: ', loss_value)

net = MyNeuralNet()
loss_fn = nn.MSELoss()


def forward(X, Y):
    logits = net(X)
    loss = loss_fn(logits, Y)
    return loss, logits


opt = nn.SGD(net.trainable_params(), learning_rate = 0.001)
grad_fn = ms.value_and_grad(forward, None, opt.parameters, has_aux=False)

# train -----------------------------------------------------------------
loss_history = []
for _ in range(50):
    (loss, _), grads = grad_fn(X, Y)
    opt(grads)
    loss_history.append(float(loss.item()))


plt.plot(loss_history)
plt.title('Loss variation over increasing epochs')
plt.xlabel('epochs')
plt.ylabel('loss value')
plt.show()

