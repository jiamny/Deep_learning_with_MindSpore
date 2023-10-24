import imgaug
print(imgaug.__version__)

from imgaug.augmenters import geometric as iaa
from imgaug.augmenters import arithmetic as iar
from imgaug import augmenters as agt

import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../utils')
from fashion_mnist import _FashionMNIST

from mindspore import context, set_seed
set_seed(0)
context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=0)


data_folder = '/media/hhj/localssd/DL_data/fashion_MNIST'
tmnist = _FashionMNIST(data_folder, is_train=True)
tr_images, tr_targets = tmnist.get_data_and_label()

fmnist_classes = tmnist.classes()
print('fmnist_classes: ', fmnist_classes)

print(type(tr_images[0]), ' ', tr_images[0].dtype)

plt.imshow(tr_images[0], cmap='gray')
plt.title('Original image')
plt.show()

aug = iaa.Affine(scale=2)
plt.imshow(aug.augment_image(tr_images[0]))
plt.title('Scaled image')
plt.show()

aug = iaa.Affine(translate_px=10)
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray')
plt.title('Translated image by 10 pixels (right and bottom)')
plt.show()

aug = iaa.Affine(translate_px={'x':10,'y':2})
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray')
plt.title('Translation of 10 pixels \nacross columns and 2 pixels over rows')
plt.show()

aug = iaa.Affine(rotate=30)
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray')
plt.title('Rotation of image by 30 degrees')
plt.show()

aug = iaa.Affine(rotate=-30)
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray')
plt.title('Rotation of image by -30 degrees')
plt.show()

aug = iaa.Affine(shear=30)
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray')
plt.title('Shear of image by 30 degrees')
plt.show()

aug = iaa.Affine(shear=-30)
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray')
plt.title('Shear of image by -30 degrees')
plt.show()

plt.figure(figsize=(20,20))
plt.subplot(161)
plt.imshow(tr_images[0], cmap='gray')
plt.title('Original image')
plt.subplot(162)
aug = iaa.Affine(scale=2, fit_output=True)
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray')
plt.title('Scaled image')
plt.subplot(163)
aug = iaa.Affine(translate_px={'x':10,'y':2}, fit_output=True)
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray')
plt.title('Translation of 10 pixels across \ncolumns and 2 pixels over rows')
plt.subplot(164)
aug = iaa.Affine(rotate=30, fit_output=True)
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray')
plt.title('Rotation of image \nby 30 degrees')
plt.subplot(165)
aug = iaa.Affine(shear=30, fit_output=True)
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray')
plt.title('Shear of image \nby 30 degrees')
plt.show()

plt.figure(figsize=(20,20))
plt.subplot(161)
plt.imshow(tr_images[0], cmap='gray')
plt.title('Original image')
plt.subplot(162)
aug = iaa.Affine(scale=2, fit_output=True)
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray')
plt.title('Scaled image')
plt.subplot(163)
aug = iaa.Affine(translate_px={'x':10,'y':2}, fit_output=True, cval = 255)
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray')
plt.title('Translation of 10 pixels across \ncolumns and 2 pixels over rows')
plt.subplot(164)
aug = iaa.Affine(rotate=30, fit_output=True)
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray')
plt.title('Rotation of image \nby 30 degrees')
plt.subplot(165)
aug = iaa.Affine(shear=30, fit_output=True)
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray')
plt.title('Shear of image \nby 30 degrees')
aug = iaa.Affine(rotate=30, fit_output=True, cval=255)
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray')
plt.title('Rotation of image by 30 degrees')
plt.show()

plt.figure(figsize=(20,20))
plt.subplot(161)
aug = iaa.Affine(rotate=30, fit_output=True, cval=0, mode='constant')
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray')
plt.title('Rotation of image by \n30 degrees with constant mode')
plt.subplot(162)
aug = iaa.Affine(rotate=30, fit_output=True, cval=0, mode='edge')
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray')
plt.title('Rotation of image by 30 degrees \n with edge mode')
plt.subplot(163)
aug = iaa.Affine(rotate=30, fit_output=True, cval=0, mode='symmetric')
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray')
plt.title('Rotation of image by \n30 degrees with symmetric mode')
plt.subplot(164)
aug = iaa.Affine(rotate=30, fit_output=True, cval=0, mode='reflect')
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray')
plt.title('Rotation of image by 30 degrees \n with reflect mode')
plt.subplot(165)
aug = iaa.Affine(rotate=30, fit_output=True, cval=0, mode='wrap')
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray')
plt.title('Rotation of image by \n30 degrees with wrap mode')
plt.show()

plt.figure(figsize=(20,20))
plt.subplot(151)
aug = iaa.Affine(rotate=(-45,45), fit_output=True, cval=0, mode='constant')
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray')
plt.subplot(152)
aug = iaa.Affine(rotate=(-45,45), fit_output=True, cval=0, mode='constant')
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray')
plt.subplot(153)
aug = iaa.Affine(rotate=(-45,45), fit_output=True, cval=0, mode='constant')
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray')
plt.subplot(154)
aug = iaa.Affine(rotate=(-45,45), fit_output=True, cval=0, mode='constant')
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray')
plt.show()

aug = iar.Multiply(1)
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray')
plt.title('Pixels multiplied by 1.0')
plt.show()

aug = iar.Multiply(0.5)
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray',vmin = 0, vmax = 255)
plt.title('Pixels multiplied by 0.5')
plt.show()

aug = agt.contrast.LinearContrast(0.5)
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray',vmin = 0, vmax = 255)
plt.title('Pixel contrast by 0.5')
plt.show()

aug = iar.Dropout(p=0.2)
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray',vmin = 0, vmax = 255)
plt.title('Random 20% pixel dropout')
plt.show()

aug = iar.SaltAndPepper(0.2)
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray',vmin = 0, vmax = 255)
plt.title('Random 20% salt and pepper noise')
plt.show()

plt.figure(figsize=(10,10))
plt.subplot(121)
aug = iar.Dropout(p=0.2,)
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray',vmin = 0, vmax = 255)
plt.title('Random 20% pixel dropout')
plt.subplot(122)
aug = iar.SaltAndPepper(0.2,)
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray',vmin = 0, vmax = 255)
plt.title('Random 20% salt and pepper noise')
plt.show()

seq = agt.meta.Sequential([
    iar.Dropout(p=0.2,),
    iaa.Affine(rotate=(-30,30))], random_order= True)
plt.imshow(seq.augment_image(tr_images[0]), cmap='gray',vmin = 0, vmax = 255)
plt.title('Image augmented using a \nrandom orderof the two augmentations')
plt.show()

aug = agt.blur.GaussianBlur(sigma=1)
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray',vmin = 0, vmax = 255)
plt.title('Gaussian blurring of image\n with a sigma of 1')
plt.show()

exit(0)