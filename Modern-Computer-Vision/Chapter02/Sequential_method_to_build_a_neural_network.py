import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor, load_checkpoint, load_param_into_net
import mindspore.context as context
import mindspore.dataset as ds
from matplotlib import pyplot as plt

ms.set_seed(10)
context.set_context(device_target="GPU")

X = Tensor([[1,2],[3,4],[5,6],[7,8]], dtype=ms.float32)
Y = Tensor([[3],[7],[11],[15]], dtype=ms.float32)

class MyDataset:
    def __init__(self,x,y):
        self.x = x.copy().asnumpy()
        self.y = y.copy().asnumpy()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]


dataset = MyDataset(X, Y)
dataloader = ds.GeneratorDataset(MyDataset(X, Y), column_names=["data", "label"],
                                 shuffle=True).batch(2, drop_remainder=False)

# ---------------------------------------------------------------------------------
# Sequential method to build a neural network
# ---------------------------------------------------------------------------------
model = nn.SequentialCell(
    nn.Dense(2, 8),
    nn.ReLU(),
    nn.Dense(8, 1)
)

print('model:\n', model)

loss_func = nn.MSELoss()

def forward(X, Y):
    logits = model(X)
    loss = loss_func(logits, Y)
    return loss, logits


opt = nn.SGD(model.trainable_params(), learning_rate = 0.001)
grad_fn = ms.value_and_grad(forward, None, opt.parameters, has_aux=False)

print('-------------------------- Train -----------------------')
import time
loss_history = []
start = time.time()
for _ in range(50):
    (loss, _), grads = grad_fn(X, Y)
    opt(grads)
    loss_history.append(loss.item())

end = time.time()
print('Training takes time: ', end - start)

plt.plot(loss_history)
plt.title('Loss variation over increasing epochs')
plt.xlabel('epochs')
plt.ylabel('loss value')
plt.show()

# Predictions
val = Tensor([[8,9],[10,11],[1.5,2.5]], dtype=ms.float32)
print('--------------------- Predictions ----------------------')
print('net(val):\n', model(val))
print('val.sum(-1):\n', val.sum(-1))
