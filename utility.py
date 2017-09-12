import os
import struct
import numpy as np
import math



def read_mnist(dataset = "training", path = "/home/anand/store/datasets/mnist"):
    """Reads the mnist binary files and returns the labels and images
       Code taken from https://gist.github.com/akesling/5358964
    Arguments:
      dataset: whether 'training' or 'testing'
      path: where the mnist binaries reside

    Returns:
      numpy array of labels, shape (60000, ) for training
      numpy array of images, shape (60000, 28, 28) for training
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    return lbl, img


def read_mnist_normalized(dataset = "training", path = "/home/anand/store/datasets/mnist"):
    labels, images = read_mnist(dataset, path)

    labels_expanded = []
    for l in labels:
        a = np.zeros((10, ))
        a[l] = 1
        labels_expanded.append(a)
        
    labels = np.array(labels_expanded)
    images = images.reshape(images.shape[0], 784,)
    images = images/255

    return labels, images

def batches(batch_size, features, labels):
    assert len(features) == len(labels)
    outout_batches = []
                    
    sample_size = len(features)
    for start_i in range(0, sample_size, batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        outout_batches.append(batch)
                            
    return outout_batches

