
import globals
import os

#-------------------------------------------------------------------
def DiffraNet():
    data_train_images = 'Datasets/DiffraNet/Dataset/Train/'
    data_test_images = 'Datasets/DiffraNet/Dataset/Test/'

    return data_train_images, data_test_images

#-------------------------------------------------------------------

def MNIST():
    data_train_images = 'Datasets/MNIST/Dataset/Train/'
    data_test_images = 'Datasets/MNIST/Dataset/Test/'

    return data_train_images, data_test_images

#-------------------------------------------------------------------
def AGIPD():
    data_train_images = 'Datasets/JAFFE/Dataset/Train/'
    data_test_images = 'Datasets/JAFFE/Dataset/Test/'

    return data_train_images, data_test_images

#-------------------------------------------------------------------
def printTrainingPath():
    print('Training-set path: %s\n' % globals.data_train_images, file = globals.file)

#-------------------------------------------------------------------
def printTestPath():
    print('Test-set path: %s\n' % globals.data_test_images, file = globals.file)

#-------------------------------------------------------------------
def printNumberOfClasses():
    globals.num_classes = len(os.listdir(globals.data_train_images))
    print('Number of Classes: %i\n' % globals.num_classes, file = globals.file)

    