import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans

"""
This script implements a kernel logistic regression model, a radial basis function network model
and a two-layer feed forward network.
"""

class Kernel_Layer(nn.Module):

    def __init__(self, sigma, hidden_dim=None):
        """
        Set hyper-parameters.
        Args:
            sigma: the sigma for Gaussian kernel (radial basis function)
            hidden_dim: the number of "kernel units", default is None, then the number of "kernel units"
                                       will be set to be the number of training samples
        """
        super(Kernel_Layer, self).__init__()
        self.sigma = sigma
        self.hidden_dim = hidden_dim
    
    def reset_parameters(self, X):
        """
        Set prototypes (stored training samples or "representatives" of training samples) of
        the kernel layer.
        """
        if self.hidden_dim is not None:
            X = self._k_means(X)
        self.prototypes = nn.Parameter(torch.tensor(X).float(), requires_grad=False)
    
    def _k_means(self, X):
        """
        K-means clustering
        
        Args:
            X: A Numpy array of shape [n_samples, n_features].
        
        Returns:
            centroids: A Numpy array of shape [self.hidden_dim, n_features].
        """
        ### YOUR CODE HERE

        kmeans_model = KMeans(init='random', n_clusters=self.hidden_dim)
        kmeans_model.fit(X)

        centroids = kmeans_model.cluster_centers_
        ### END YOUR CODE
        return centroids

    def kernel_function(self,x):
        """
        x: a torch tensor of shape [1,n_features]

        Returns:
        a torch tensor of shape [1,num_of_protoypes]
        """

        def _each_prototype_kernel(_row):
            a = _row-x
            value = np.dot(a,a)
            #print(value)
            return np.exp(-1*value/(2*self.sigma*self.sigma))

        my_function = lambda row:_each_prototype_kernel(row)

        #print(self.prototypes.shape)
        result = np.array(list(map(my_function, self.prototypes)))
            #np.apply(self.prototypes,1,my_function)
        #print(result.shape)
        return result.T

    def forward(self, x):
        """
        Compute Gaussian kernel (radial basis function) of the input sample batch
        and self.prototypes (stored training samples or "representatives" of training samples).

        Args:
            x: A torch tensor of shape [batch_size, n_features]

        Returns:
            A torch tensor of shape [batch_size, num_of_prototypes]
        """
        assert x.shape[1] == self.prototypes.shape[1]

        #my_function = lambda row: self.kernel_function(row)

        #print(x.shape)
        #print('above is the x')
        #result = torch.Tensor(list(map(my_function, x))).float()
        #print('------')
        #print(result.shape)
        _size = (x.shape[0], self.prototypes.shape[0], x.shape[1])
        # batch size , 1, nfeatures - batch size, m, features.   1, m, 256; n, m , 256
        x = x.unsqueeze(1).expand(_size)
        p = self.prototypes.unsqueeze(0).expand(_size)
        norms = (x - p).pow(2).sum(-1).pow(0.5)
        return torch.exp(-1 * norms / (2 * (self.sigma * self.sigma)))


        #np.apply_along_axis(self.kernel_function,1,x)
        # self.prototypes = [num_of_prototypes, n_features]. x[0] = [1,nfeatures]
        # each prototype  - x[0]
        ### YOUR CODE HERE
        # Basically you need to follow the equation of radial basis function
        # in the section 5 of note at http://people.tamu.edu/~sji/classes/nnkernel.pdf
        
        ### END YOUR CODE

class Kernel_LR(nn.Module):

    def __init__(self, sigma, hidden_dim):
        """
        Define network structure.

        Args:
            sigma: used in the kernel layer.
            hidden_dim: the number of prototypes in the kernel layer,
                                       in this model, hidden_dim has to be equal to the 
                                       number of training samples.
        """
        super(Kernel_LR, self).__init__()
        self.hidden_dim = hidden_dim

        ### YOUR CODE HERE
        # Use pytorch nn.Sequential object to build a network composed of a

        # kernel layer (Kernel_Layer object) and a linear layer (nn.Linear object)

        # keeping bias false in linear layer for now.
        _kernel_layer = Kernel_Layer(sigma=sigma)
        linear_layer  = nn.Linear(in_features=self.hidden_dim,out_features=1,bias=False)
        self.net = nn.Sequential(_kernel_layer,linear_layer)



        # Remember that kernel logistic regression model uses all training samples
        # in kernel layer, so set 'hidden_dim' argument to be None when creating
        # a Kernel_Layer object.

        # How should we set the "bias" argument of nn.Linear? 
        
        ### END YOUR CODE

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: A torch tensor of shape [batch_size, n_features]
        
        Returns:
            A torch tensor of shape [batch_size, 1]
        """
        return self.net(x)
    
    def reset_parameters(self, X):
        """
        Initialize the weights of the linear layer and the prototypes of the kernel layer.

        Args:
            X: A Numpy array of shape [n_samples, n_features], training data matrix.
        """
        assert X.shape[0] == self.hidden_dim
        for layer in self.net:
            if hasattr(layer, 'reset_parameters'):
                if isinstance(layer, Kernel_Layer):
                    layer.reset_parameters(X)
                else:
                    layer.reset_parameters()



class RBF(nn.Module):

    def __init__(self, sigma, hidden_dim):
        """
        Define network structure.

        Args:
            sigma: used in the kernel layer.
            hidden_dim: the number of prototypes in the kernel layer,
                                       in this model, hidden_dim is a user-specified hyper-parameter.
        """
        super(RBF, self).__init__()
        self.sigma = sigma
        self.hidden_dim = hidden_dim
        _kernel_layer = Kernel_Layer(sigma=self.sigma,hidden_dim=self.hidden_dim)
        linear_layer = nn.Linear(in_features=self.hidden_dim, out_features=1, bias=False)
        self.net = nn.Sequential(_kernel_layer, linear_layer)


        ### YOUR CODE HERE
        # Use pytorch nn.Sequential object to build a network composed of a
        # kernel layer (Kernel_Layer object) and a linear layer (nn.Linear object)
        # How should we set the "bias" argument of nn.Linear? 
        
        ### END CODE HERE

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: A torch tensor of shape [batch_size, n_features]
        
        Returns:
            A torch tensor of shape [batch_size, 1]
        """
        return self.net(x)
    
    def reset_parameters(self, X):
        """
        Initialize the weights of the linear layer and the prototypes of the kernel layer.

        Args:
            X: A Numpy array of shape [n_samples, n_features], training data matrix.
        """
        for layer in self.net:
            if hasattr(layer, 'reset_parameters'):
                if isinstance(layer, Kernel_Layer):
                    layer.reset_parameters(X)
                else:
                    layer.reset_parameters()



class FFN(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        """
        Define network structure.

        Args:
            input_dim: number of features of each input.
            hidden_dim: the number of hidden units in the hidden layer, a user-specified hyper-parameter.
        """
        super(FFN, self).__init__()
        ### YOUR CODE HERE
        # Use pytorch nn.Sequential object to build a network composed of
        # two linear layers (nn.Linear object)
        l1 = nn.Linear(in_features=input_dim, out_features=hidden_dim, bias=False)
        l2 = nn.Linear(in_features=hidden_dim, out_features=1, bias=False)
        self.net = nn.Sequential(l1,l2)
        
        ### END CODE HERE

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: A torch tensor of shape [batch_size, n_features]
        
        Returns:
            A torch tensor of shape [batch_size, 1]
        """
        return self.net(x)

    def reset_parameters(self):
        """
        Initialize the weights of the linear layers.
        """
        for layer in self.net:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()