from mindvision.dataset import Cifar10

'''
Loading a Dataset
'''
# Dataset root directory
data_dir = "../data/cifar10"

# Download, extract, and load the CIFAR-10 training dataset.
dataset = Cifar10(path=data_dir, split='train', batch_size=6, resize=32, download=False)
dataset = dataset.run()

'''
Iterating a Dataset
'''
data = next(dataset.create_dict_iterator())
print(f"Data type:{type(data['image'])}\nImage shape: {data['image'].shape}, Label: {data['label']}")

data = next(dataset.create_dict_iterator(output_numpy=True))
print(f"Data type:{type(data['image'])}\nImage shape: {data['image'].shape}, Label: {data['label']}")

'''
Data Processing and Augmentation
'''
import numpy as np
import matplotlib.pyplot as plt

import mindspore.dataset.vision.c_transforms as transforms

trans = [transforms.HWC2CHW()]  # convert shape of the input image from <H,W,C> to <C,H,W>
dataset = Cifar10(data_dir, batch_size=6, resize=32, repeat_num=1, shuffle=True, transform=trans)
data = dataset.run()
data = next(data.create_dict_iterator())

images = data["image"].asnumpy()
labels = data["label"].asnumpy()
print(f"Image shape: {images.shape}, Label: {labels}")

plt.figure()
for i in range(1, 7):
    plt.subplot(2, 3, i)
    image_trans = np.transpose(images[i - 1], (1, 2, 0))
    plt.title(f"{dataset.index2label[labels[i - 1]]}")
    plt.imshow(image_trans, interpolation="None")
plt.show()
plt.close()

'''
Data Augmentation
'''
import numpy as np
import matplotlib.pyplot as plt

import mindspore.dataset.vision.c_transforms as transforms

# Image augmentation
trans = [
    transforms.RandomCrop((32, 32), (4, 4, 4, 4)),  # Automatically crop the image.
    transforms.RandomHorizontalFlip(prob=0.5),  # Flip the image horizontally at random.
    transforms.HWC2CHW(),  # Convert (h, w, c) to (c, h, w).
]

dataset = Cifar10(data_dir, batch_size=6, resize=32, transform=trans)
data = dataset.run()
data = next(data.create_dict_iterator())
images = data["image"].asnumpy()
labels = data["label"].asnumpy()
print(f"Image shape: {images.shape}, Label: {labels}")

plt.figure()
for i in range(1, 7):
    plt.subplot(2, 3, i)
    image_trans = np.transpose(images[i - 1], (1, 2, 0))
    plt.title(f"{dataset.index2label[labels[i - 1]]}")
    plt.imshow(image_trans, interpolation="None")
plt.show()
plt.close()

exit(0)
