from resnetv2 import resnet18

from mindspore.compression.quant import QuantizationAwareTraining
from mindspore.nn.optim import Adam
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore import load_checkpoint
from mindspore import Model, context

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", save_graphs=False)

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-loc", "--check_point", default = 'ckpt',
   help="Model checkpoint file")

args = vars(ap.parse_args())

num_classes = 5
lr = 0.01
weight_decay = 1e-4

resnet = resnet18(num_classes)
print(resnet)

ls = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
opt = Adam(resnet.trainable_params(),lr, weight_decay)

quantizer = QuantizationAwareTraining(bn_fold=False)
quant = quantizer.quantize(resnet)

load_checkpoint(args['check_point'], net=quant) # loading the custom trained checkpoint

eval_data = create_dataset(training = False) # define the test dataset


model = Model(quant, loss_fn=ls, optimizer=opt, metrics={'acc'})

acc = model.eval(eval_data)

print('Accuracy of model is: ', acc['acc'] * 100)
