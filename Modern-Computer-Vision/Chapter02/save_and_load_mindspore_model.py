import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor, load_checkpoint, load_param_into_net
import mindspore.context as context
import mindspore.dataset as ds
import numpy as np

ms.set_seed(10)
context.set_context(device_target="GPU")

X = Tensor([[1,2],[3,4],[5,6],[7,8]], dtype=ms.float32)
Y = Tensor([[3],[7],[11],[15]], dtype=ms.float32)
print(X.shape, ' ', len(X))
print(X[0])

class MyDataset:
    def __init__(self,x,y):
        self.x = x.copy().asnumpy()
        self.y = y.copy().asnumpy()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, ix):
        return self.x[ix], self.y[ix]


dataset = MyDataset(X, Y)
print("Access with dataset[0]: ", len(dataset[0]))

dataloader = ds.GeneratorDataset(dataset, column_names=["data", "label"],
                                 shuffle=True).batch(2, drop_remainder=False)
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

iterator = dataloader.create_tuple_iterator()
import time
loss_history = []
start = time.time()
for _ in range(50):
    epoch_loss = []
    for batch in iterator:
        x, y = batch
        (loss, _), grads = grad_fn(x, y)
        opt(grads)
        epoch_loss.append(float(loss.item()))
    epoch_loss = np.array(epoch_loss).mean()
    loss_history.append(epoch_loss)

end = time.time()
print('Training takes time: ', end - start)

# Saving model
save_path = 'mymodel.ckpt'
ms.save_checkpoint(model, save_path)

# Loading model
load_path = 'mymodel.ckpt'

# Save the model parameters to the parameter dictionary.
# The model parameters saved during the training are loaded.
param_dict = load_checkpoint(load_path)

# Redefine a Net neural network.
net = nn.SequentialCell(
    nn.Dense(2, 8),
    nn.ReLU(),
    nn.Dense(8, 1)
)

# Load parameters to the network.
load_param_into_net(net, param_dict)

# Predictions
val = Tensor([[8,9],[10,11],[1.5,2.5]], dtype=ms.float32)
print('--------------------- Predictions ----------------------')
print('net(val):\n', net(val))
print('val.sum(-1):\n', val.sum(-1))
