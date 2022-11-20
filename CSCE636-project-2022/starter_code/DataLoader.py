import os
import pickle
import numpy as np

"""This script implements the functions for reading data.
"""

def load_data(data_dir):
    """Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches
            are stored.

    Returns:
        x_train: An numpy array of shape [50000, 3072].
            (dtype=np.float32)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072].
            (dtype=np.float32)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int32)
    """

    ### YOUR CODE HERE
    y_train = []
    x_train = None
    for i in range(1, 6):
        with open(os.path.join(data_dir, f'data_batch_{i}'), 'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
            y_train.extend(data[b'labels'])
            if i != 1:
                x_train = np.vstack((x_train, data[b'data'].astype(np.uint8)))
            else:
                x_train = data[b'data'].astype(np.uint8)
    y_train = np.array(y_train)
    with open(os.path.join(data_dir, 'test_batch'), 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
        x_test = data[b'data'].astype(np.uint8)
        y_test = np.array(data[b'labels'])

    ### END CODE HERE

    return x_train, y_train, x_test, y_test


def load_testing_images(data_dir): # TODO: CHANGE AND PREPROCESS IT ACCORDING TO GIVEN TEST IMAGE.
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        x_test: An numpy array of shape [N, 3072].
            (dtype=np.float32)
    """

    ### YOUR CODE HERE
    # assuming this is the prediction.npy
    x_test = np.load(data_dir)

    ### END CODE HERE

    return x_test


def train_valid_split(x_train, y_train, train_ratio=0.8):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        train_ratio: A float number between 0 and 1.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    
    ### YOUR CODE HERE
    n,_ = x_train.shape
    split_index = int(train_ratio*n)
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]

    ### END CODE HERE

    return x_train_new, y_train_new, x_valid, y_valid

