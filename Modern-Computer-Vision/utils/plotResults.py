import numpy as np
import matplotlib.pyplot as plt

def displayImages(images, label):
    # 显示10张图片
    images = (images*255).asnumpy().astype(np.uint8)

    for i in range(images.shape[0]):
        image = images[i].reshape((28,28))

        if i >= 10:
            break
        plt.subplot(2, 5, i + 1)
        plt.imshow(image.squeeze(), cmap=plt.cm.gray)
        plt.title(label[i])
    plt.show()


def showLossAndAccuracy(train_losses, train_accuracies, val_losses, val_accuracies, EPOCH, tlt):
    epochs = np.arange(EPOCH)+1

    plt.figure(figsize=(10, 12))
    plt.subplot(211)
    plt.plot(epochs, train_losses, 'bo', label='Training loss')
    plt.plot(epochs, val_losses, 'r', label='Validation loss')
    plt.title('Training and validation loss with ' + tlt)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid('off')

    plt.subplot(212)
    plt.plot(epochs, train_accuracies, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy with ' + tlt)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    yl = [x for x in plt.gca().get_yticks()]
    plt.gca().set_yticks(yl)
    plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in yl])
    plt.legend()
    plt.grid('off')
    plt.show()


def showTrainLossAndAccuracy(train_losses, train_accuracies, EPOCH, tlt):
    epochs = np.arange(EPOCH)+1
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.title('Loss over epochs ' + tlt)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.legend()

    plt.subplot(122)
    plt.title('Accuracy over epochs ' + tlt)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    yl = [x for x in plt.gca().get_yticks()]
    plt.gca().set_yticks(yl)
    plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in yl])
    plt.legend()
    plt.show()

def showWeightDistribution(model):
    plt.figure(figsize=(16, 16))
    for ix, par in enumerate(model.trainable_params()):
        if(ix==0):
            plt.subplot(2, 2, ix + 1)
            plt.hist(par.asnumpy().flatten())
            plt.title('Weights conencting input to hidden layer')
        elif(ix ==1):
            plt.subplot(2, 2, ix + 1)
            plt.hist(par.asnumpy().flatten())
            plt.title('Biases of hidden layer')
        elif(ix==2):
            plt.subplot(2, 2, ix + 1)
            plt.hist(par.asnumpy().flatten())
            plt.title('Weights conencting hidden to output layer')
        elif(ix ==3):
            plt.subplot(2, 2, ix + 1)
            plt.hist(par.asnumpy().flatten())
            plt.title('Biases of output layer')
    plt.show()

