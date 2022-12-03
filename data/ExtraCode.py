### YOUR CODE HERE
# import tensorflow as tf
# import torch
import torch
from torch.functional import Tensor
import torch.nn as nn

"""This script defines the network.
"""

# class MyNetwork(object):
#
#     def __init__(self, configs):
#         self.configs = configs
#
#     def __call__(self, inputs, training):
#     	'''
#     	Args:
#             inputs: A Tensor representing a batch of input images.
#             training: A boolean. Used by operations that work differently
#                 in training and testing phases such as batch normalization.
#         Return:
#             The output Tensor ohenf the network.
#     	'''
#         return self.build_network(inputs, training)
#
#     def build_network(self, inputs, training):
#         return inputs




""" This script defines the network.
"""


def resnet(depth, width, num_classes):
    assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
    n = (depth - 4) // 6
    widths = [int(v * width) for v in (16, 32, 64)]

    def gen_block_params(ni, no):
        return {
            'conv0': utils.conv_params(ni, no, 3),
            'conv1': utils.conv_params(no, no, 3),
            'bn0': utils.bnparams(ni),
            'bn1': utils.bnparams(no),
            'convdim': utils.conv_params(ni, no, 1) if ni != no else None,
        }

    def gen_group_params(ni, no, count):
        return {'block%d' % i: gen_block_params(ni if i == 0 else no, no)
                for i in range(count)}

    flat_params = utils.cast(utils.flatten({
        'conv0': utils.conv_params(3, 16, 3),
        'group0': gen_group_params(16, widths[0], n),
        'group1': gen_group_params(widths[0], widths[1], n),
        'group2': gen_group_params(widths[1], widths[2], n),
        'bn': utils.bnparams(widths[2]),
        'fc': utils.linear_params(widths[2], num_classes),
    }))

    utils.set_requires_grad_except_bn_(flat_params)

    def block(x, params, base, mode, stride):
        o1 = F.relu(utils.batch_norm(x, params, base + '.bn0', mode), inplace=True)
        y = F.conv2d(o1, params[base + '.conv0'], stride=stride, padding=1)
        o2 = F.relu(utils.batch_norm(y, params, base + '.bn1', mode), inplace=True)
        z = F.conv2d(o2, params[base + '.conv1'], stride=1, padding=1)
        if base + '.convdim' in params:
            return z + F.conv2d(o1, params[base + '.convdim'], stride=stride)
        else:
            return z + x

    def group(o, params, base, mode, stride):
        for i in range(n):
            o = block(o, params, '%s.block%d' % (base,i), mode, stride if i == 0 else 1)
        return o

    def f(input, params, mode):
        x = F.conv2d(input, params['conv0'], padding=1)
        g0 = group(x, params, 'group0', mode, 1) # STACK.
        g1 = group(g0, params, 'group1', mode, 2)
        g2 = group(g1, params, 'group2', mode, 2)
        o = F.relu(utils.batch_norm(g2, params, 'bn', mode))
        o = F.avg_pool2d(o, 8, 1, 0)
        o = o.view(o.size(0), -1)
        o = F.linear(o, params['fc.weight'], params['fc.bias'])
        return o

##################################################
#############################################################################
# Blocks building the network
#############################################################################

class batch_norm_relu_layer(nn.Module):
    """ Perform batch normalization then relu.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.997) -> None:
        super(batch_norm_relu_layer, self).__init__()
        ### YOUR CODE HERE
        self.batch_norm = nn.BatchNorm2d(num_features,
                                         eps=eps,
                                         momentum=momentum)
        self.relu = nn.ReLU()
        ### YOUR CODE HERE

    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        output = self.batch_norm(inputs)
        output = self.relu(output)
        return output
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
    """

    def __init__(self, filters, projection_shortcut, strides, first_num_filters) -> None:
        super(standard_block, self).__init__()
        ### YOUR CODE HERE
        # if self.projection_shortcut is not None:

        # filters_in = first_num_filters if strides == 1 else filters // 2

        if filters == first_num_filters:
            filters_in = first_num_filters
        elif projection_shortcut:
            filters_in = filters // 2
        else:
            filters_in = filters

        # print("filters_in: ", filters_in)
        ### YOUR CODE HERE
        self.conv1 = nn.Conv2d(in_channels=filters_in,
                               out_channels=filters,
                               kernel_size=3,
                               stride=strides,
                               padding=1)

        self.bn_relu = batch_norm_relu_layer(num_features=filters)

        self.conv2 = nn.Conv2d(in_channels=filters,
                               out_channels=filters,
                               kernel_size=3,
                               padding=1)  # here we are keeping stride 1

        self.bn = nn.BatchNorm2d(filters,
                                 eps=1e-5,
                                 momentum=0.997)

        if projection_shortcut is None:
            self.addition = nn.Identity()
        else:
            self.addition = projection_shortcut  # NOTE: projection_shortcut should handle stride

        ### YOUR CODE HERE
        self.relu = nn.ReLU()
        ### YOUR CODE HERE

    def forward(self, inputs: Tensor) -> Tensor:
        ### YOUR CODE HERE
        # referring the Figure 4(a) in paper [2], original format
        # print(f"inputs.shape={inputs.shape}")

        aux = self.addition(inputs)
        # print(f"aux.shape={aux.shape}")

        output = self.conv1(inputs)
        output = self.bn_relu(output)

        # print(f"output.shape={output.shape}")

        output = self.conv2(output)
        output = self.bn(output)

        output += aux
        output = self.relu(output)

        # print(f"output.shape={output.shape}")

        return output
        ### YOUR CODE HERE

# TRYING OUT HW2 RESNET MODEL WITH WIDE RESNET VARIATION TO CHECK ACCURACY.

class ResNet(nn.Module):
    def __init__(self,
                 resnet_size,
                 num_classes,
                 first_num_filters,
                 width,resnet_version=2
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
        output_map_size | 32x32 | 32X32  |16  | 8x8    | 1x1         |
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
        self.width = width

        ### YOUR CODE HERE
        # define conv1
        self.start_layer = nn.Conv2d(in_channels=3,
                                     out_channels=first_num_filters * width,
                                     kernel_size=3,
                                     padding=1)

        # NOTE: since, output feature map size is not decreasing here, we do NOT add stride here.
        #       but in paper[1], there are stride = 1 for conv1, that is the start layer
        #       but in paper[1], there are stride = 1 for conv1, that is the start layer
        ### YOUR CODE HERE

        # We do not include batch normalization or activation functions in V2
        # for the initial conv1 because the first block unit will perform these
        # for both the shortcut and non-shortcut paths as part of the first
        # block's projection.
        if self.resnet_version == 1:
            self.batch_norm_relu_start = batch_norm_relu_layer(
                num_features=self.first_num_filters * width,
                eps=1e-5,
                momentum=0.997,
            )
        if self.resnet_version == 1:
            block_fn = standard_block
        else:
            block_fn = bottleneck_block

        self.stack_layers = nn.ModuleList()
        for i in range(3):
            filters = self.first_num_filters * (2 ** i)
            strides = 1 if i == 0 else 2
            self.stack_layers.append(
                stack_layer(filters * width, block_fn, strides, self.resnet_size, self.first_num_filters * width))
        self.output_layer = output_layer(filters * 4 * width, self.resnet_version, self.num_classes)

    def forward(self, inputs):
        # print(inputs.shape)
        outputs = self.start_layer(inputs)
        if self.resnet_version == 1:
            outputs = self.batch_norm_relu_start(outputs)
        for i in range(3):
            # print(f"i={i}")
            outputs = self.stack_layers[i](outputs)
        outputs = self.output_layer(outputs)
        return outputs



class bottleneck_block(nn.Module):
    def __init__(self, filters, projection_shortcut, strides, first_num_filters) -> None:
        super(bottleneck_block, self).__init__()


        intermediate_filters = filters // 4

        if projection_shortcut is None:
            in_filters = filters
        else:
            if first_num_filters == filters // 4:
                in_filters = filters // 4
            else:
                in_filters = filters // 2

        self.bn_relu1 = batch_norm_relu_layer(in_filters)  # TODO: check
        self.conv1 = nn.Conv2d(in_channels=in_filters,  # TODO: check
                               out_channels=intermediate_filters,
                               kernel_size=1)  # 1d conv does not need padding

        self.bn_relu2 = batch_norm_relu_layer(intermediate_filters)
        self.conv2 = nn.Conv2d(in_channels=intermediate_filters,
                               out_channels=intermediate_filters,
                               kernel_size=3,
                               stride=strides,
                               padding=1)

        self.bn_relu3 = batch_norm_relu_layer(intermediate_filters)
        self.conv3 = nn.Conv2d(in_channels=intermediate_filters,
                               out_channels=filters,  # multiplying by 4 here!
                               kernel_size=1)  # 1d conv does not need padding

        if projection_shortcut is None:
            self.addition = nn.Identity()
        else:
            self.addition = projection_shortcut  # NOTE: do I need to add a brackets: () ??? verify at the end

    def forward(self, inputs: Tensor) -> Tensor:

        output = self.bn_relu1(inputs)

        aux = self.addition(output)  # projection shortcut or identity


        output = self.conv1(output)
        output = self.bn_relu2(output)
        output = self.conv2(output)
        output = self.bn_relu3(output)
        output = self.conv3(output)


        output += aux

        return output

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
        filters_out = filters * 4 if block_fn is bottleneck_block else filters

        if filters == first_num_filters:
            filters_in = first_num_filters
        else:
            filters_in = filters // 2

        if filters_out == first_num_filters * 4:
            projection_shortcut = nn.Conv2d(in_channels=filters_out // 4,
                                            out_channels=filters_out,
                                            kernel_size=(1,1),
                                            stride=strides)
        else:
            projection_shortcut = nn.Conv2d(in_channels=filters_out // 2,
                                            out_channels=filters_out,
                                            kernel_size=(1,1),
                                            stride=strides)

        blocks = [block_fn(filters_out, projection_shortcut, strides, first_num_filters)]  # TODO: recheck
        #filters, projection_shortcut, strides, first_num_filters
        for i in range(resnet_size - 1):
            blocks.append(block_fn(filters_out, None, 1, first_num_filters))

        self.layer = nn.Sequential(*blocks)


    def forward(self, inputs: Tensor) -> Tensor:
        output = self.layer(inputs)
        return output

