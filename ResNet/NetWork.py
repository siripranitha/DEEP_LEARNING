import torch
from torch.functional import Tensor
import torch.nn as nn

""" This script defines the network.
"""

class ResNet(nn.Module):
    def __init__(self,
            resnet_version,
            resnet_size,
            num_classes,
            first_num_filters,
        ):
        """
        1. Define hyperparameters.
        Args:
            resnet_version: 1 or 2, If 2, use the bottleneck blocks.
            resnet_size: A positive integer (n).
            num_classes: A positive integer. Define the number of classes.
            first_num_filters: An integer. The number of filters to use for the
                first block layer of the model. This number is then doubled
                for each subsampling block layer.
        
        2. Classify a batch of input images.

        Architecture (first_num_filters = 16):
        layer_name      | start | stack1 | stack2 | stack3 | output      |
        output_map_size | 32x32 | 32X32  | 16x16  | 8x8    | 1x1         |
        #layers         | 1     | 2n/3n  | 2n/3n  | 2n/3n  | 1           |
        #filters        | 16    | 16(*4) | 32(*4) | 64(*4) | num_classes |

        n = #residual_blocks in each stack layer = self.resnet_size
        The standard_block has 2 layers each.
        The bottleneck_block has 3 layers each.
        
        Example of replacing:
        standard_block      conv3-16 + conv3-16
        bottleneck_block    conv1-16 + conv3-16 + conv1-64

        Args:
            inputs: A Tensor representing a batch of input images.
        
        Returns:
            A logits Tensor of shape [<batch_size>, self.num_classes].
        """
        super(ResNet, self).__init__()
        self.resnet_version = resnet_version
        self.resnet_size = resnet_size
        self.num_classes = num_classes
        self.first_num_filters = first_num_filters

        ### YOUR CODE HERE
        # define conv1
        self.conv1 = nn.Conv2d(in_channels=3,
                          out_channels=self.first_num_filters, # first num filters?
                          kernel_size=3,
                          stride=1,
                          padding=1,
                               bias=False,
                               padding_mode='zeros') # doubt about padding.


        
        ### YOUR CODE HERE

        # We do not include batch normalization or activation functions in V2
        # for the initial conv1 because the first block unit will perform these
        # for both the shortcut and non-shortcut paths as part of the first
        # block's projection.
        if self.resnet_version == 1:
            self.batch_norm_relu_start = batch_norm_relu_layer(
                num_features=self.first_num_filters, 
                eps=1e-5, 
                momentum=0.997,
            )
        if self.resnet_version == 1:
            block_fn = standard_block
        else:
            block_fn = bottleneck_block

        self.stack_layers = nn.ModuleList()
        for i in range(3):
            filters = self.first_num_filters * (2**i) # is there any problem here?
            strides = 1 if i == 0 else 2
            self.stack_layers.append(stack_layer(filters, block_fn, strides, self.resnet_size, self.first_num_filters))
        self.output_layer = output_layer(filters*4, self.resnet_version, self.num_classes)

    def start_layer(self,inputs):
        return self.conv1(inputs)
    
    def forward(self, inputs):
        outputs = self.start_layer(inputs)
        if self.resnet_version == 1:
            outputs = self.batch_norm_relu_start(outputs)
        for i in range(3):
            outputs = self.stack_layers[i](outputs)
        outputs = self.output_layer(outputs)
        return outputs

#############################################################################
# Blocks building the network
#############################################################################

class batch_norm_relu_layer(nn.Module):
    """ Perform batch normalization then relu.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.997) -> None:
        super(batch_norm_relu_layer, self).__init__()
        ### YOUR CODE HERE
        self.batch_norm_layer = nn.BatchNorm2d(num_features=num_features,
                                               eps=eps,
                                               momentum=momentum)
        self.relu = nn.ReLU(inplace=True)

        ### YOUR CODE HERE
    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        out = self.batch_norm_layer(inputs)
        out = self.relu(out)
        return out
        
        ### YOUR CODE HERE

class standard_block(nn.Module):
    """ Creates a standard residual block for ResNet.

    Args:
        filters: A positive integer. The number of filters for the first 
            convolution.
        projection_shortcut: The function to use for projection shortcuts
      		(typically a 1x1 convolution when downsampling the input).
		strides: A positive integer. The stride to use for the block. If
			greater than 1, this block will ultimately downsample the input.
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
            # first num filters, we will get when it is particularly first block in given stack,
    """
    def __init__(self, filters, projection_shortcut, strides, first_num_filters) -> None:
        super(standard_block, self).__init__()
        ### YOUR CODE HERE
        self.filters = filters
        self.projection_shortcut = projection_shortcut
        self.strides = strides
        self.first_num_filters = first_num_filters
        #standard_block      conv3-16 + conv3-16
        in_filters = self.filters
        if projection_shortcut:
            in_filters = self.filters//2
        #in_filters = self.filters if self.filters==self.first_num_filters or self else self.filters//2
        self.conv3_blk1 = nn.Conv2d(in_channels=in_filters,
                          out_channels=self.filters, # first num filters?
                          kernel_size=(3,3),
                          stride=strides, # replace strides here.
                          padding=(1,1))
        self.conv3_blk2 = nn.Conv2d(in_channels=self.filters,
                                    out_channels=self.filters,  # first num filters?
                                    kernel_size=(3, 3),
                                    stride=(1,1), # replace strides here
                                    padding=(1, 1))

        self.batch_norm_relu_layer_current = batch_norm_relu_layer(num_features=self.filters)


        if self.projection_shortcut is not None:
            self.identity_conv = self.projection_shortcut

        self.relu = nn.ReLU(inplace=True)
        ### YOUR CODE HERE

        ### YOUR CODE HERE
        
        ### YOUR CODE HERE

    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        identity_mapping  = inputs
        if self.projection_shortcut:
            identity_mapping = self.projection_shortcut(identity_mapping)
            identity_mapping = self.batch_norm_relu_layer_current(identity_mapping)

        out = self.conv3_blk1(inputs)
        out = self.batch_norm_relu_layer_current.forward(out)
        out = self.conv3_blk2(out)
        out = self.batch_norm_relu_layer_current.forward(out)

        out = out + identity_mapping

        out = self.relu(out)
        ### YOUR CODE HERE
        return out

        


class bottleneck_block(nn.Module):
    """ Creates a bottleneck block for ResNet.

    Args:
        filters: A positive integer. The number of filters for the first 
            convolution. NOTE: filters_out will be 4xfilters.
        projection_shortcut: The function to use for projection shortcuts
      		(typically a 1x1 convolution when downsampling the input).
		strides: A positive integer. The stride to use for the block. If
			greater than 1, this block will ultimately downsample the input.
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, projection_shortcut, strides, first_num_filters) -> None:
        super(bottleneck_block, self).__init__()

        ### YOUR CODE HERE
        self.filters = filters
        self.projection_shortcut = projection_shortcut
        self.strides = strides
        self.first_num_filters = first_num_filters
        #bottleneck_block    conv1-16 + conv3-16 + conv1-64

        in_filters = self.filters
        if self.projection_shortcut:
            in_filters = self.filters//2
            if self.filters==self.first_num_filters*4:
                in_filters = self.first_num_filters
            # update in_filters



        self.conv1_blk1 = nn.Conv2d(in_channels=in_filters,
                                    out_channels=self.filters//4,  # first num filters?
                                    kernel_size=(1, 1),
                                    stride=self.strides
                                    )
        self.conv3_blk2 = nn.Conv2d(in_channels=self.filters//4,
                                    out_channels=self.filters//4,  # first num filters?
                                    kernel_size=(3, 3),
                                    stride=(1,1),
                                    padding=(1, 1))
        self.conv1_blk3 = nn.Conv2d(in_channels=self.filters//4,
                                    out_channels=self.filters,  # first num filters?
                                    kernel_size=(1, 1),
                                    stride=(1,1))

        self.batch_norm_relu_layer_first = batch_norm_relu_layer(num_features=in_filters) # assumiing 4*filters here
        self.batch_norm_relu_layer_second = batch_norm_relu_layer(num_features=self.filters//4)
        self.relu = nn.ReLU(inplace=True)
        # check and calculate the count of filters once more.
        # Hint: Different from standard lib implementation, you need pay attention to
        # how to define in_channel of the first bn and conv of each block based on
        # Args given above.
        
        ### YOUR CODE HERE
    
    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        # The projection shortcut should come after the first batch norm and ReLU
		# since it performs a 1x1 convolution.

        if self.projection_shortcut:
            inputs1 = self.batch_norm_relu_layer_first.forward(inputs)
            identity_mapping = self.projection_shortcut(inputs1)
        else:
            identity_mapping = inputs

        out = self.batch_norm_relu_layer_first.forward(inputs)
        out = self.conv1_blk1(out)

        out = self.batch_norm_relu_layer_second.forward(out)
        out = self.conv3_blk2(out)

        out = self.batch_norm_relu_layer_second.forward(out)
        out = self.conv1_blk3(out)

        out = out + identity_mapping
        return out # is relu required?
        ### YOUR CODE HERE

class stack_layer(nn.Module):
    """ Creates one stack of standard blocks or bottleneck blocks.

    Args:
        filters: A positive integer. The number of filters for the first
			    convolution in a block.
		block_fn: 'standard_block' or 'bottleneck_block'.
		strides: A positive integer. The stride to use for the first block. If
				greater than 1, this layer will ultimately downsample the input.
        resnet_size: #residual_blocks in each stack layer
        first_num_filters: An integer. The number of filters to use for the
            first block layer of the model.
    """
    def __init__(self, filters, block_fn, strides, resnet_size, first_num_filters) -> None:
        super(stack_layer, self).__init__()

        self.filters = filters
        self.filters_out = filters * 4 if block_fn is bottleneck_block else filters # for projection shortcut code.
        self.first_num_filters = first_num_filters
        self.resnet_size = resnet_size
        self.strides = strides




        if block_fn is bottleneck_block:
            #bb_infliters
            projection_def = nn.Conv2d(in_channels=self.filters*2,
                                    out_channels=self.filters_out,
                                    kernel_size=(1,1),
                                    stride=strides
                                    )

        else:
            #_in_filters = self.filters if self.filters==self.first_num_filters else self.filters//2
            projection_def = nn.Conv2d(in_channels=filters//2,
                                       out_channels=self.filters,
                                       kernel_size=(1,1),
                                       stride=strides
                                       )
            #self.std_params = [_in_filters,self.filters_out]


        if filters==first_num_filters:
            if block_fn is standard_block:
                projection_def = None
            else:
                projection_def = nn.Conv2d(in_channels=self.first_num_filters,
                                           out_channels=self.filters_out,
                                           kernel_size=(1, 1),
                                           stride=strides
                                           )

        self.block_fn_blk0 = block_fn(filters=self.filters_out,
                                   projection_shortcut=projection_def,
                                   strides=self.strides,
                                   first_num_filters=self.first_num_filters)
        self.block_fn = block_fn(filters=self.filters_out,
                                   projection_shortcut=None,
                                   strides=(1,1),
                                   first_num_filters=self.first_num_filters)

        ### END CODE HERE
        # projection_shortcut = ?
        # Only the first block per stack_layer uses projection_shortcut and strides
        
        ### END CODE HERE
    
    def forward(self, inputs: Tensor) -> Tensor:
        ### END CODE HERE
        out = inputs

        for i in range(self.resnet_size):
            if i==0:
                #print('first of stack')
                #print(self.filters,self.filters_out)
                out = self.block_fn_blk0.forward(out)
            else:
                #print('remaining stack')
                out = self.block_fn.forward(out)

        return out

        
        ### END CODE HERE

class output_layer(nn.Module):
    """ Implement the output layer.

    Args:
        filters: A positive integer. The number of filters.
        resnet_version: 1 or 2, If 2, use the bottleneck blocks.
        num_classes: A positive integer. Define the number of classes.
    """
    def __init__(self, filters, resnet_version, num_classes) -> None:
        super(output_layer, self).__init__()
        # Only apply the BN and ReLU for model that does pre_activation in each
		# bottleneck block, e.g. resnet V2.
        #self.filters = filters
        self.resnet_version = resnet_version
        self.num_classes = num_classes
        if (resnet_version == 2):
            #self.filters=filters//4
            self.bn_relu = batch_norm_relu_layer(filters, eps=1e-5, momentum=0.997)

        self.global_pool =nn.AdaptiveAvgPool2d((1,1))#nn.AvgPool2d(kernel_size=8)
        if resnet_version==1:
            self.fc = nn.Linear(filters//4,10,bias=True)
        else:
            self.fc = nn.Linear(filters , 10, bias=True)

        self.softmax = nn.Softmax(dim=-1)

        # recheck this code, seems fishy
        
        ### END CODE HERE
        
        ### END CODE HERE
    
    def forward(self, inputs: Tensor) -> Tensor:
        ### END CODE HERE
        if self.resnet_version==2:
            out = self.bn_relu(inputs)
        else:
            out = inputs

        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        #print(out.shape,.shape)
        out = self.fc(out)
        out = self.softmax(out)
        return out
        
        ### END CODE HERE