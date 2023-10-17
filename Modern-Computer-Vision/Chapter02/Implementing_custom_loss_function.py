import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor
import mindspore.context as context
import mindspore.dataset as ds

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

dataloader = ds.GeneratorDataset(MyDataset(X, Y), column_names=["data", "label"],
                                 shuffle=True).batch(2, drop_remainder=False)

# Iter the dataset and check if the data is created successful
for data in dataloader:
    print("RandomAccess dataset: ", data[0], ' ', data[1], ' ', type(data[0]))
print('-'*80)

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


def my_mean_squared_error(_y, y):
    loss = (_y-y)**2
    loss = loss.mean()
    return loss


loss_func = nn.MSELoss()
loss_value = loss_func(mynet(X),Y)
print('loss_value: ', loss_value.item())

print('mean squared error: ', my_mean_squared_error(mynet(X),Y))
input_to_hidden = mynet.input_to_hidden_layer(X)
print('input_to_hidden: ', input_to_hidden)
hidden_activation = mynet.hidden_layer_activation(input_to_hidden)
print('hidden_activation: ', hidden_activation)


