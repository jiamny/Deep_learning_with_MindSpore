# --------------------------------
# Saving the models
# --------------------------------

'''
Iteration policy
'''
from mindspore.train.callback import CheckpointConfig

# Save one CheckPoint file every 32 steps, and up to 10 CheckPoint files
config_ck = CheckpointConfig(save_checkpoint_steps=32, keep_checkpoint_max=10)

'''
Time policy
'''
from mindspore.train.callback import CheckpointConfig

# Save a CheckPoint file every 30 seconds and a CheckPoint file every 3 minutes
config_ck = CheckpointConfig(save_checkpoint_seconds=30, keep_checkpoint_per_n_minutes=3)

'''
Breakpoint renewal
'''
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig

# Configure the breakpoint continuation function to turn on
config_ck = CheckpointConfig(save_checkpoint_steps=32, keep_checkpoint_max=10, exception_save=True)

'''
If an exception occurs during training, the end-of-life CheckPoint is automatically saved, and if an exception 
occurs in the 10th step of the 10th epoch in the training, the saved end-of-life CheckPoint file is as follows.

resnet50-10_10_breakpoint.ckpt  # The end-of-life CheckPoint file name will be marked by '_breakpoint' to distinguish it from the normal process checkPoint.
'''

'''
save_checkpoint saving models

save_obj parameter
'''
from mindspore import save_checkpoint, Tensor
from mindspore import dtype as mstype

save_list = [{"name": "lr", "data": Tensor(0.01, mstype.float32)}, {"name": "train_epoch", "data": Tensor(20, mstype.int32)}]
save_checkpoint(save_list, "hyper_param.ckpt")

'''
integrated_save parameter
'''
save_checkpoint(net, "resnet50-2_32.ckpt", integrated_save=True)

'''
async_save parameter
'''
save_checkpoint(net, "resnet50-2_32.ckpt", async_save=True)

'''
append_dict parameter
'''
save_dict = {"epoch_num": 2, "lr": 0.01}
# In addition to the parameters in net, the information save_dict is also saved in the ckpt file
save_checkpoint(net, "resnet50-2_32.ckpt",append_dict=save_dict)

# -------------------------------------------
# Transfer Learning
# -------------------------------------------
from mindvision.classification.models import resnet50
from mindspore import load_checkpoint, load_param_into_net
from mindvision.dataset import DownLoad
# Download the pre-trained model for Resnet50
dl = DownLoad()
dl.download_url('https://download.mindspore.cn/vision/classification/resnet50_224.ckpt')
# Define a resnet50 network with a classification class of 2
resnet = resnet50(2)
# Model parameters are saved to the param_dict
param_dict = load_checkpoint("resnet50_224.ckpt")

# Get a list of parameter names for the fully connected layer
param_filter = [x.name for x in resnet.head.get_parameters()]

def filter_ckpt_parameter(origin_dict, param_filter):
    """Delete elements including param_filter parameter names in the origin_dict"""
    for key in list(origin_dict.keys()): # Get all parameter names for the model
        for name in param_filter: # Iterate over the parameter names in the model to be deleted
            if name in key:
                print("Delete parameter from checkpoint:", key)
                del origin_dict[key]
                break

# Delete the full connection layer
filter_ckpt_parameter(param_dict, param_filter)

# Prints the updated model parameters
load_param_into_net(resnet, param_dict)

# -------------------------------------------
# Model Export
# -------------------------------------------
'''
Export MindIR Model

If you want to perform inference on the Ascend AI processor, 
you can also generate the corresponding AIR format model file through the network definition and CheckPoint.
'''
import numpy as np
from mindspore import Tensor, export, load_checkpoint
from mindvision.classification.models import resnet50

resnet = resnet50(1000)
load_checkpoint("resnet50_224.ckpt", net=resnet)

input_np = np.random.uniform(0.0, 1.0, size=[1, 3, 224, 224]).astype(np.float32)

# Export the file resnet50_224.mindir to the current folder
export(resnet, Tensor(input_np), file_name='resnet50_224', file_format='MINDIR')

'''
If you wish to save the data preprocess operations into MindIR and use them to perform inference,
you can pass the Dataset object into export method:
'''
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as C
from mindspore import export, load_checkpoint
from mindvision.classification.models import resnet50
from mindvision.dataset import DownLoad

def create_dataset_for_renset(path):
    """Create a dataset"""
    data_set = ds.ImageFolderDataset(path)
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    data_set = data_set.map(operations=[C.Decode(), C.Resize(256), C.CenterCrop(224),
                                        C.Normalize(mean=mean, std=std), C.HWC2CHW()], input_columns="image")
    data_set = data_set.batch(1)
    return data_set

dataset_url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/notebook/datasets/beginner/DogCroissants.zip"
path = "./datasets"
# Download and extract the dataset
dl = DownLoad()
dl.download_and_extract_archive(url=dataset_url, download_path=path)
# Load the dataset
path = "./datasets/DogCroissants/val/"
de_dataset = create_dataset_for_renset(path)
# Define the network
resnet = resnet50()

# Load the preprocessing model parameters into the network
load_checkpoint("resnet50_224.ckpt", net=resnet)
# Export a MindIR file with preprocessing information
export(resnet, de_dataset, file_name='resnet50_224', file_format='MINDIR')

'''
Export AIR Model

If you want to perform inference on the Ascend AI processor, you can also generate 
the corresponding AIR format model file through the network definition and CheckPoint.
'''
import numpy as np
from mindspore import Tensor, export, load_checkpoint
from mindvision.classification.models import resnet50

resnet = resnet50()
# Load parameters into the network
load_checkpoint("resnet50_224.ckpt", net=resnet)
# Network input
input_np = np.random.uniform(0.0, 1.0, size=[1, 3, 224, 224]).astype(np.float32)
# Save the resnet50_224.air file to the current directory
export(resnet, Tensor(input_np), file_name='resnet50_224', file_format='AIR')

'''
Export ONNX Model

When you have a CheckPoint file, if you want to do inference on Ascend AI processor, 
GPU, or CPU, you need to generate ONNX models based on the network and CheckPoint. 
'''
import numpy as np
from mindspore import Tensor, export, load_checkpoint
from mindvision.classification.models import resnet50

resnet = resnet50()
load_checkpoint("resnet50_224.ckpt", net=resnet)

input_np = np.random.uniform(0.0, 1.0, size=[1, 3, 224, 224]).astype(np.float32)

# Save the resnet50_224.onnx file to the current directory
export(resnet, Tensor(input_np), file_name='resnet50_224', file_format='ONNX')
