import numpy as np

"""This script implements the functions for reading data.
"""

def load_data(filename):
    """Load a given txt file.

    Args:
        filename: A string.

    Returns:
        raw_data: An array of shape [n_samples, 256].
        labels : An array of shape [n_samples,].
        
    """
    data= np.load(filename)
    x= data['x']
    y= data['y']
    return x, y

def train_valid_split(raw_data, labels, split_index):
	"""Split the original training data into a new training dataset
	and a validation dataset.
	n_samples = n_train_samples + n_valid_samples

	Args:
		raw_data: An array of shape [n_samples, 256].
        labels : An array of shape [n_samples,].
		split_index: An integer.

	"""
	return raw_data[:split_index], raw_data[split_index:], labels[:split_index], labels[split_index:]

def prepare_X(raw_X):
    """Extract features from raw_X as required.
    Args:
        raw_X: An array of shape [n_samples, 256].
    Returns:
        X: An array of shape [n_samples, n_features].
    """
    raw_image = raw_X.reshape((-1, 16, 16))
    nsamples, _, _ = raw_image.shape
    final = np.zeros((nsamples, 3))
    # created an output matrix with 3 columns, one for the bias term and the first and second column for the two features
    # Feature 1: Measure of Symmetry
	### YOUR CODE HERE
    for i,each in enumerate(raw_image):
        final[i][1] = - abs(each-np.fliplr(each)).sum()/256
    # updating the final matrix with the first feature, i.e measure of symmetry
    #I take 16x16 pixel matrix corresponding to each pixel, flip it , subtract it from original matrix,
    #compute absolute value of the subtraction and multiply with -1/256
	### END YOUR CODE
	# Feature 2: Measure of Intensity
	### YOUR CODE HERE
    for i,each in enumerate(raw_image):
        final[i][2] = each.sum()/256
    # updating the feature matrix with second feature , measure of intensity.
    # i take 16x16 pixel matrix corresponding to each image compute its sum and divide it by 256
	### END YOUR CODE

	# Feature 3: Bias Term. Always 1.
	### YOUR CODE HERE
    for i in range(nsamples):
        final[i][0] = 1
    # updating the feature matrix with bias term in the first column
	### END YOUR CODE
	# Stack features together in the following order.
	# [Feature 3, Feature 1, Feature 2]
	### YOUR CODE HERE
    # NO need for this code as i already built features in a stacked fashion in the above code.
	
	### END YOUR CODE
    return final

def prepare_y(raw_y):
    """
    Args:
        raw_y: An array of shape [n_samples,].
        
    Returns:
        y: An array of shape [n_samples,].
        idx:return idx for data label 1 and 2.
    """
    y = raw_y
    idx = np.where((raw_y==1) | (raw_y==2))
    y[np.where(raw_y==0)] = 0
    y[np.where(raw_y==1)] = 1
    y[np.where(raw_y==2)] = 2

    return y, idx




