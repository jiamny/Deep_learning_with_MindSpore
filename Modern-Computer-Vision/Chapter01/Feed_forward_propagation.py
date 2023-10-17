


### Forward Propagation
import mindspore as ms

def feed_forward(inputs, outputs, weights):       
    pre_hidden = ms.ops.dot(inputs,weights[0])+ weights[1]
    hidden = 1/(1+ms.ops.exp(-pre_hidden))
    pred_out = ms.ops.dot(hidden, weights[2]) + weights[3]
    mean_squared_error = ms.ops.mean(ms.ops.square(pred_out - outputs))
    return mean_squared_error
