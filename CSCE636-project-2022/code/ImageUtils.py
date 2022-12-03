import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms
from PIL import Image
# todo: study

"""This script implements the functions for data augmentation
and preprocessing.
"""

def parse_record(record, training):
    """Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32].
    """
    # Reshape from [depth * height * width] to [depth, height, width].
    depth_major = record.reshape((3, 32, 32))

    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])

    image = preprocess_image(image, training)

    # Convert from [height, width, depth] to [depth, height, width]
    #image = np.transpose(image, [2, 0, 1])

    #todo; taking hw2 preprocessing. can use new techniques here.

    return image


def preprocess_image(image, training):
    """Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [3, 32, 32].
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [3, 32, 32]. The processed image.
    """
    current_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255, np.array([63.0, 62.1, 66.7]) / 255)
    ])
    if training:
        current_transform = transforms.Compose([
        transforms.Pad(4,padding_mode='reflect'),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomCrop(32),
        current_transform
        ])

    #todo: recheck this
    #print(type(image),image.shape)
    #print(image)
    image = Image.fromarray(image)
    image = current_transform(image)

    return image


def visualize(image, save_name='test.png'):
    """Visualize a single test image.
    
    Args:
        image: An array of shape [3072]
        save_name: An file name to save your visualization.
    
    Returns:
        image: An array of shape [32, 32, 3].
    """
    image = image.reshape((3, 32, 32))
    image = np.transpose(image, [1, 2, 0])

    plt.imshow(image)

    plt.savefig(save_name)
    return image

# Other functions
### YOUR CODE HERE

### END CODE HERE