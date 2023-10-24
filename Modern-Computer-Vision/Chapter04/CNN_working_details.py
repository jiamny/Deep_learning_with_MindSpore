import mindspore as ms
from mindspore import nn
from mindspore import ops
import mindspore.dataset as ds

from mindspore import context, set_seed
set_seed(0)
context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=0)

import numpy as np
import matplotlib.pyplot as plt

class TensorDataset:
    def __init__(self, x, y):
        self.x = x.copy().asnumpy()
        self.y = y.copy().asnumpy()

    def __getitem__(self, ix):
        x, y = self.x[ix], self.y[ix]
        return x, y

    def __len__(self):
        return len(self.x)


X_train = ms.Tensor([[[[1,2,3,4],[2,3,4,5],[5,6,7,8],[1,3,4,5]]],
                  [[[-1,2,3,-4],[2,-3,4,5],[-5,6,-7,8],[-1,-3,-4,-5]]]], dtype=ms.float32)
X_train /= 8
y_train = ms.Tensor([0,1], dtype=ms.float32)
print(X_train[0])
print(len(X_train))

trn_dl = ds.GeneratorDataset(TensorDataset(X_train, y_train),
                                  column_names=["data", "target"],
                                  num_parallel_workers=1,
                                  shuffle=True,
                                  sampler=None).batch(1)
print(type(trn_dl))
data, target = (trn_dl.create_tuple_iterator()).__next__()
print(data.shape)
print(target)

model = nn.SequentialCell(
        nn.Conv2d(1, 1, kernel_size=3, has_bias=True),
        nn.MaxPool2d(kernel_size=2, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Dense(3*3, 1),
        nn.Sigmoid()
)

# ----------------------------------------------------------------
# summary of model
# ----------------------------------------------------------------
X = data
print(X.shape)
for blk in model:
    X = blk(X)
    print(blk.__class__.__name__,'output shape:\t',X.shape)


loss_fn = nn.BCELoss()
optimizer = nn.Adam(model.trainable_params(), learning_rate=1e-2)
EPOCH = 300

def train(model, dataset, loss_fn, optimizer):

    def forward_fn(data, label):
        prediction = model(data).flatten()
        loss = loss_fn(prediction, label)
        return loss, prediction

    grad_fn = ops.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

    def train_step(data, label):
        (loss, _), grads = grad_fn(data, label)
        # 获得损失 depend用来处理操作间的依赖关系
        loss = ops.depend(loss, optimizer(grads))
        return loss

    model.set_train()
    train_epoch_losses, train_epoch_accuracies = [], []

    for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
        # 批量训练获得损失值
        loss = train_step(data, label)
        train_epoch_losses.append(float(loss.item()))

    train_epoch_loss = np.array(train_epoch_losses).mean()
    return train_epoch_loss


for t in range(EPOCH):
    loss = train(model, trn_dl, loss_fn, optimizer)
    if (t+1) % 100 == 0:
        print(f"Epoch: {t+1} loss: {loss:>.4f}")

print('model(X_train[:1]: ', model(data))

cnn_w = None
cnn_b = None
lin_w = None
lin_b = None
for ix, par in enumerate(model.trainable_params()):
    print('ix: ', ix, ' ', par.__class__, ' ', par.asnumpy())
    if ix == 0: cnn_w = ms.Tensor(par.asnumpy(), dtype=ms.float32).squeeze()
    if ix == 1: cnn_b = ms.Tensor(par.asnumpy(), dtype=ms.float32).squeeze()
    if ix == 2: lin_w = ms.Tensor(par.asnumpy(), dtype=ms.float32).squeeze()
    if ix == 3: lin_b = ms.Tensor(par.asnumpy(), dtype=ms.float32).squeeze()

print(cnn_w.shape)

h_im, w_im = X_train.shape[2:]
h_conv, w_conv = cnn_w.shape
print('h_im: ', h_im, ' w_im: ', w_im)
print('h_conv: ', h_conv, ' w_conv: ', w_conv)

sumprod = ops.zeros((h_im - h_conv + 1, w_im - w_conv + 1), dtype=ms.float32)
print(sumprod.shape)

for i in range(h_im - h_conv + 1):
    for j in range(w_im - w_conv + 1):
        img_subset = X_train[0, 0, i:(i+3), j:(j+3)]
        model_filter = cnn_w.reshape((3,3))
        val = ops.sum(img_subset*model_filter) + cnn_b
        sumprod[i,j] = val

#sumprod = ops.clamp(sumprod, min=0)
pooling_layer_output, _ = ops.max(sumprod)
intermediate_output_value = pooling_layer_output * lin_w + lin_b
print('sigmoid(intermediate_output_value): ', ops.sigmoid(intermediate_output_value))

exit(0)