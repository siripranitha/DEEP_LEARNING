### YOUR CODE HERE
# import tensorflow as tf
# import torch
import torch
from torch.functional import Tensor
import torch.nn as nn

"""This script defines the network.
"""
class bn_relu_layer(nn.Module):
    def __init__(self, filters, eps=1e-5, momentum=0.997):

        super().__init__()
        self.layer =  nn.Sequential(nn.BatchNorm2d(filters,eps=eps,momentum=momentum),nn.ReLU(inplace=True))
    def forward(self,input):
        return self.layer(input)


class MyNetwork(nn.Module):
    def __init__(self, resnet_size,
                 num_classes,
                 width):
        super().__init__()


        self.resnet_size = resnet_size
        self.num_classes = num_classes
        self.first_num_filters = 16
        self.width = width

        #three groups
        each_group_start_filters = [int(v * width) for v in (16, 32, 64)]

        self.start_layer = nn.Conv2d(in_channels=3,
                                     out_channels=16, # width in first conv?
                                     kernel_size=(3,3),
                                     padding=1)

        self.stack_layers = nn.ModuleList()
        # batch norm, relu, conv2d(in out for first block, out, out for rem.)
        # bn, relu, add projection shortcut x if first block or x.
        for i in range(3):
            if i==0:
                self.stack_layers.append(group_layer(in_filters=16, out_filters=each_group_start_filters[0],
                                                     stride=1,resnet_size=resnet_size))
            else:
                self.stack_layers.append(group_layer(in_filters=each_group_start_filters[i-1], out_filters=each_group_start_filters[i],
                                                     stride=2,resnet_size=resnet_size))

        self.output_layer = output_layer(each_group_start_filters[2],  self.num_classes)

    def forward(self, inputs):
        outputs = self.start_layer(inputs)

        for i in range(3):
            #print('stack number :',i)
            outputs = self.stack_layers[i](outputs)
        outputs = self.output_layer(outputs)
        return outputs




class group_layer(nn.Module):

    def __init__(self, in_filters,out_filters, stride, resnet_size) -> None:

        super().__init__()
        blocks = []
        for i in range(resnet_size):
            if i==0:

                blocks.append(StdBlock(in_filters=in_filters,out_filters=out_filters,stride=stride))
            else:
                blocks.append(StdBlock(in_filters=out_filters,out_filters=out_filters,stride=1))
        self.blocks = blocks
        self.layer = nn.Sequential(*blocks)


    def forward(self, inputs: Tensor) -> Tensor:
        output = self.layer(inputs)
        # for i,each_block in enumerate(self.blocks):
        #     #print('block number :',i)
        #     output = each_block.forward(output)
        #     #print(output.shape)
        return output



class StdBlock(nn.Module):
    def __init__(self, in_filters, out_filters, stride):
        self.params = [in_filters,out_filters,stride]
        super().__init__()
        self.initial_layer = bn_relu_layer(in_filters)

        layers = []

        self.stride = stride

        layers.append(nn.Conv2d(in_channels=in_filters,  # TODO: check
                               out_channels=out_filters,
                               kernel_size=3,padding=1,stride=stride))
        layers.append(bn_relu_layer(out_filters))
        layers.append(nn.Conv2d(in_channels=out_filters,  # TODO: check
                                out_channels=out_filters,
                                kernel_size= 3, padding=1, stride=1))

        self.layers = layers
        self.projection_shortcut = nn.Conv2d(in_channels=in_filters,
                                             out_channels=out_filters,
                                             kernel_size=1, stride=stride) if in_filters != out_filters else None

    def forward(self,inputs):

        o1 = self.initial_layer(inputs)
        z = o1
        #print(o1.shape,inputs.shape)
        for i,each_layer in enumerate(self.layers):
            z = each_layer.forward(z)
            #print(i,z.shape)

        #print(inputs.shape, self.params,o1.shape,z.shape)
        if self.projection_shortcut:
            return z + self.projection_shortcut(o1)
        else:
            return z+inputs


class output_layer(nn.Module):

    def __init__(self, filters, num_classes) -> None:
        super(output_layer, self).__init__()
        # Only apply the BN and ReLU for model that does pre_activation in each
        # bottleneck block, e.g. resnet V2.
        # layers = []


        self.one = bn_relu_layer(filters)
        self.global_avg_pool = nn.AvgPool2d(8)
        self.dropout = nn.Dropout()
        self.lin = nn.Linear(filters, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        ### END CODE HERE

        output = self.one(inputs)
        # print(f"output.shape={output.shape}")
        output = self.global_avg_pool(output)
        # print(f"output.shape={output.shape}")

        # flatten
        output = output.view(output.size(0), -1)

        # print(f"output.shape={output.shape}")
        output = self.dropout(output)

        output = self.lin(output)
        # print(f"output.shape={output.shape}")
        # output = self.softmax(output)
        # print(f"output.shape={output.shape}")

        return output
        ### END CODE HERE

