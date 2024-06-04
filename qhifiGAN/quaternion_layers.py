##########################################################
# pytorch-qnn v1.0
# Titouan Parcollet
# LIA, Université d'Avignon et des Pays du Vaucluse
# ORKIS, Aix-en-provence
# October 2018
##########################################################

import numpy  as np
from   numpy.random import RandomState
import torch
from torch.nn.functional import normalize
from   torch.autograd import Variable
import torch.nn.functional  as F
import torch.nn as nn
from   torch.nn.parameter  import Parameter
from   torch.nn import Module
from   quaternion_ops import *
import math
from typing import Any, Optional, TypeVar
import sys

class QuaternionTransposeConv(Module):
    r"""Applies a Quaternion Transposed Convolution (or Deconvolution) to the incoming data.
    """

    # def __init__(self, in_channels, out_channels, kernel_size, stride,
    #              dilation=1, padding=0, output_padding=0, groups=1, bias=True, init_criterion='he',
    #              weight_init='quaternion', seed=None, operation='convolution2d', rotation=False,
    #              quaternion_format=False): I have changed operation='convolution2d' to operation only
    def __init__(self, in_channels, out_channels, kernel_size, stride, operation,
                 dilation=1, padding=0, output_padding=0, groups=1, bias=True, init_criterion='he',
                 weight_init='quaternion', seed=None, rotation=False,
                 quaternion_format=False):

        super(QuaternionTransposeConv, self).__init__()

        self.in_channels       = in_channels  // 4
        self.out_channels      = out_channels // 4
        self.stride            = stride
        self.padding           = padding
        self.output_padding    = output_padding
        self.groups            = groups
        self.dilation        = dilation
        self.init_criterion    = init_criterion
        self.weight_init       = weight_init
        self.seed              = seed if seed is not None else np.random.randint(0,1234)
        self.rng               = RandomState(self.seed)
        self.operation         = operation
        self.rotation          = rotation
        self.quaternion_format = quaternion_format
        self.winit             = {'quaternion': quaternion_init,
                                  'unitary'   : unitary_init,
                                  'random'    : random_init}[self.weight_init]


        (self.kernel_size, self.w_shape) = get_kernel_and_weight_shape( self.operation,
            self.out_channels, self.in_channels, kernel_size )

        self.r_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.i_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.j_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.k_weight  = Parameter(torch.Tensor(*self.w_shape))


        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        affect_init_conv(self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                    self.kernel_size, self.winit, self.rng, self.init_criterion)
        if self.bias is not None:
           self.bias.data.zero_()

    def forward(self, input):

        if self.rotation:
            return quaternion_transpose_conv_rotation(input, self.r_weight, self.i_weight,
                self.j_weight, self.k_weight, self.bias, self.stride, self.padding,
                self.output_padding, self.groups, self.dilation, self.quaternion_format)
        else:
            return quaternion_transpose_conv(input, self.r_weight, self.i_weight, self.j_weight,
                self.k_weight, self.bias, self.stride, self.padding, self.output_padding,
                self.groups, self.dilation)


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_channels='      + str(self.in_channels) \
            + ', out_channels='   + str(self.out_channels) \
            + ', bias='           + str(self.bias is not None) \
            + ', kernel_size='    + str(self.kernel_size) \
            + ', stride='         + str(self.stride) \
            + ', padding='        + str(self.padding) \
            + ', dilation='       + str(self.dilation) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init='    + str(self.weight_init) \
            + ', seed='           + str(self.seed) \
            + ', operation='      + str(self.operation) + ')'





class QuaternionTransposeConv2D(Module):
    r"""Applies a Quaternion Transposed Convolution (or Deconvolution) to the incoming data.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilation=1, padding=0, output_padding=0, groups=1, bias=True, init_criterion='he',
                 weight_init='quaternion', seed=None, operation='convolution2d', rotation=False,
                 quaternion_format=False): 
        
    # def __init__(self, in_channels, out_channels, kernel_size, stride, operation,
    #              dilation=1, padding=0, output_padding=0, groups=1, bias=True, init_criterion='he',
    #              weight_init='quaternion', seed=None, rotation=False,
    #              quaternion_format=False):

        super(QuaternionTransposeConv2D, self).__init__()

        self.in_channels       = in_channels  // 4
        self.out_channels      = out_channels // 4
        self.stride            = stride
        self.padding           = padding
        self.output_padding    = output_padding
        self.groups            = groups
        self.dilation        = dilation
        self.init_criterion    = init_criterion
        self.weight_init       = weight_init
        self.seed              = seed if seed is not None else np.random.randint(0,1234)
        self.rng               = RandomState(self.seed)
        self.operation         = operation
        self.rotation          = rotation
        self.quaternion_format = quaternion_format
        self.winit             = {'quaternion': quaternion_init,
                                  'unitary'   : unitary_init,
                                  'random'    : random_init}[self.weight_init]


        (self.kernel_size, self.w_shape) = get_kernel_and_weight_shape( self.operation,
            self.out_channels, self.in_channels, kernel_size )

        self.r_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.i_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.j_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.k_weight  = Parameter(torch.Tensor(*self.w_shape))


        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        affect_init_conv(self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                    self.kernel_size, self.winit, self.rng, self.init_criterion)
        if self.bias is not None:
           self.bias.data.zero_()

    def forward(self, input):

        if self.rotation:
            return quaternion_transpose_conv_rotation(input, self.r_weight, self.i_weight,
                self.j_weight, self.k_weight, self.bias, self.stride, self.padding,
                self.output_padding, self.groups, self.dilation, self.quaternion_format)
        else:
            return quaternion_transpose_conv(input, self.r_weight, self.i_weight, self.j_weight,
                self.k_weight, self.bias, self.stride, self.padding, self.output_padding,
                self.groups, self.dilation)


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_channels='      + str(self.in_channels) \
            + ', out_channels='   + str(self.out_channels) \
            + ', bias='           + str(self.bias is not None) \
            + ', kernel_size='    + str(self.kernel_size) \
            + ', stride='         + str(self.stride) \
            + ', padding='        + str(self.padding) \
            + ', dilation='       + str(self.dilation) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init='    + str(self.weight_init) \
            + ', seed='           + str(self.seed) \
            + ', operation='      + str(self.operation) + ')'



class QuaternionConv(Module):
    r"""Applies a Quaternion Convolution to the incoming data. conv1d 
    """

    # def __init__(self, in_channels, out_channels, kernel_size, stride,operation,
    #              dilation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
    #              weight_init='quaternion', seed=None, rotation=False, quaternion_format=True, scale=False):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='quaternion', seed=None, operation='convolution1d', rotation=False, quaternion_format=True, scale=False):

        super(QuaternionConv, self).__init__()

        self.in_channels       = in_channels  // 4
        self.out_channels      = out_channels // 4
        self.stride            = stride
        self.padding           = padding
        self.groups            = groups
        self.dilation        = dilation
        self.init_criterion    = init_criterion
        self.weight_init       = weight_init
        self.seed              = seed if seed is not None else np.random.randint(0,1234)
        self.rng               = RandomState(self.seed)
        self.operation         = operation
        self.rotation          = rotation
        self.quaternion_format = quaternion_format
        self.winit             =    {'quaternion': quaternion_init,
                                     'unitary'   : unitary_init,
                                     'random'    : random_init}[self.weight_init]
        self.scale             = scale


        (self.kernel_size, self.w_shape) = get_kernel_and_weight_shape( self.operation,
            self.in_channels, self.out_channels, kernel_size )

        self.r_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.i_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.j_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.k_weight  = Parameter(torch.Tensor(*self.w_shape))

        if self.scale:
            self.scale_param  = Parameter(torch.Tensor(self.r_weight.shape))
        else:
            self.scale_param  = None

        if self.rotation:
            self.zero_kernel = Parameter(torch.zeros(self.r_weight.shape), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        affect_init_conv(self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                    self.kernel_size, self.winit, self.rng, self.init_criterion)
        if self.scale_param is not None:
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        if self.bias is not None:
           self.bias.data.zero_()

    def forward(self, input):


        if self.rotation:
            return quaternion_conv_rotation(input, self.zero_kernel, self.r_weight, self.i_weight, self.j_weight,
                self.k_weight, self.bias, self.stride, self.padding, self.groups, self.dilation,
                self.quaternion_format, self.scale_param)
        else:
            return quaternion_conv(input, self.r_weight, self.i_weight, self.j_weight,
                self.k_weight, self.bias, self.stride, self.padding, self.groups, self.dilation)


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_channels='      + str(self.in_channels) \
            + ', out_channels='   + str(self.out_channels) \
            + ', bias='           + str(self.bias is not None) \
            + ', kernel_size='    + str(self.kernel_size) \
            + ', stride='         + str(self.stride) \
            + ', padding='        + str(self.padding) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init='    + str(self.weight_init) \
            + ', seed='           + str(self.seed) \
            + ', rotation='       + str(self.rotation) \
            + ', q_format='       + str(self.quaternion_format) \
            + ', operation='      + str(self.operation) + ')'




class QuaternionConv2d(Module):
    r"""Applies a Quaternion Convolution to the incoming data.
    """

    # def __init__(self, in_channels, out_channels, kernel_size, stride,operation,
    #              dilation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
    #              weight_init='quaternion', seed=None, rotation=False, quaternion_format=True, scale=False):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='quaternion', seed=None, operation='convolution2d', rotation=False, quaternion_format=True, scale=False):

        super(QuaternionConv2d, self).__init__()

        self.in_channels       = in_channels  // 4
        self.out_channels      = out_channels // 4
        self.stride            = stride
        self.padding           = padding
        self.groups            = groups
        self.dilation        = dilation
        self.init_criterion    = init_criterion
        self.weight_init       = weight_init
        self.seed              = seed if seed is not None else np.random.randint(0,1234)
        self.rng               = RandomState(self.seed)
        self.operation         = operation
        self.rotation          = rotation
        self.quaternion_format = quaternion_format
        self.winit             =    {'quaternion': quaternion_init,
                                     'unitary'   : unitary_init,
                                     'random'    : random_init}[self.weight_init]
        self.scale             = scale


        (self.kernel_size, self.w_shape) = get_kernel_and_weight_shape( self.operation,
            self.in_channels, self.out_channels, kernel_size )

        self.r_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.i_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.j_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.k_weight  = Parameter(torch.Tensor(*self.w_shape))

        if self.scale:
            self.scale_param  = Parameter(torch.Tensor(self.r_weight.shape))
        else:
            self.scale_param  = None

        if self.rotation:
            self.zero_kernel = Parameter(torch.zeros(self.r_weight.shape), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        affect_init_conv(self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                    self.kernel_size, self.winit, self.rng, self.init_criterion)
        if self.scale_param is not None:
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        if self.bias is not None:
           self.bias.data.zero_()

    def forward(self, input):


        if self.rotation:
            return quaternion_conv_rotation(input, self.zero_kernel, self.r_weight, self.i_weight, self.j_weight,
                self.k_weight, self.bias, self.stride, self.padding, self.groups, self.dilation,
                self.quaternion_format, self.scale_param)
        else:
            return quaternion_conv(input, self.r_weight, self.i_weight, self.j_weight,
                self.k_weight, self.bias, self.stride, self.padding, self.groups, self.dilation)


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_channels='      + str(self.in_channels) \
            + ', out_channels='   + str(self.out_channels) \
            + ', bias='           + str(self.bias is not None) \
            + ', kernel_size='    + str(self.kernel_size) \
            + ', stride='         + str(self.stride) \
            + ', padding='        + str(self.padding) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init='    + str(self.weight_init) \
            + ', seed='           + str(self.seed) \
            + ', rotation='       + str(self.rotation) \
            + ', q_format='       + str(self.quaternion_format) \
            + ', operation='      + str(self.operation) + ')'






class QuaternionConv2dUni(Module):
    r"""Applies a Quaternion Convolution to the incoming data.
    """

    # def __init__(self, in_channels, out_channels, kernel_size, stride,operation,
    #              dilation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
    #              weight_init='quaternion', seed=None, rotation=False, quaternion_format=True, scale=False):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilation=1, padding=0, groups=1, bias=True, init_criterion='glorot',
                 weight_init='unitary', seed=None, operation='convolution2d', rotation=False, quaternion_format=True, scale=False):

        super(QuaternionConv2dUni, self).__init__()

        self.in_channels       = in_channels  // 4
        self.out_channels      = out_channels // 4
        self.stride            = stride
        self.padding           = padding
        self.groups            = groups
        self.dilation        = dilation
        self.init_criterion    = init_criterion
        self.weight_init       = weight_init
        self.seed              = seed if seed is not None else np.random.randint(0,1234)
        self.rng               = RandomState(self.seed)
        self.operation         = operation
        self.rotation          = rotation
        self.quaternion_format = quaternion_format
        self.winit             =    {'quaternion': quaternion_init,
                                     'unitary'   : unitary_init,
                                     'random'    : random_init}[self.weight_init]
        self.scale             = scale


        (self.kernel_size, self.w_shape) = get_kernel_and_weight_shape( self.operation,
            self.in_channels, self.out_channels, kernel_size )

        self.r_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.i_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.j_weight  = Parameter(torch.Tensor(*self.w_shape))
        self.k_weight  = Parameter(torch.Tensor(*self.w_shape))

        if self.scale:
            self.scale_param  = Parameter(torch.Tensor(self.r_weight.shape))
        else:
            self.scale_param  = None

        if self.rotation:
            self.zero_kernel = Parameter(torch.zeros(self.r_weight.shape), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        affect_init_conv(self.r_weight, self.i_weight, self.j_weight, self.k_weight,
                    self.kernel_size, self.winit, self.rng, self.init_criterion)
        if self.scale_param is not None:
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        if self.bias is not None:
           self.bias.data.zero_()

    def forward(self, input):


        if self.rotation:
            return quaternion_conv_rotation(input, self.zero_kernel, self.r_weight, self.i_weight, self.j_weight,
                self.k_weight, self.bias, self.stride, self.padding, self.groups, self.dilation,
                self.quaternion_format, self.scale_param)
        else:
            return quaternion_conv(input, self.r_weight, self.i_weight, self.j_weight,
                self.k_weight, self.bias, self.stride, self.padding, self.groups, self.dilation)


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_channels='      + str(self.in_channels) \
            + ', out_channels='   + str(self.out_channels) \
            + ', bias='           + str(self.bias is not None) \
            + ', kernel_size='    + str(self.kernel_size) \
            + ', stride='         + str(self.stride) \
            + ', padding='        + str(self.padding) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init='    + str(self.weight_init) \
            + ', seed='           + str(self.seed) \
            + ', rotation='       + str(self.rotation) \
            + ', q_format='       + str(self.quaternion_format) \
            + ', operation='      + str(self.operation) + ')'



# class QuaternionMaxPool(Module):
#     def __init__(self) -> None:
#         super().__init__()







######################




class QuaternionLinearAutograd(Module):
    r"""Applies a quaternion linear transformation to the incoming data. A custom
    Autograd function is call to drastically reduce the VRAM consumption. Nonetheless, computing
    time is also slower compared to QuaternionLinear().
    """

    def __init__(self, in_features, out_features, bias=True,
                 init_criterion='glorot', weight_init='quaternion',
                 seed=None, rotation=False, quaternion_format=True, scale=False):

        super(QuaternionLinearAutograd, self).__init__()
        self.in_features       = in_features//4
        self.out_features      = out_features//4
        self.rotation          = rotation
        self.quaternion_format = quaternion_format
        self.r_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.i_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.j_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.k_weight = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.scale    = scale

        if self.scale:
            self.scale_param  = Parameter(torch.Tensor(self.in_features, self.out_features))
        else:
            self.scale_param  = None

        if self.rotation:
            self.zero_kernel  = Parameter(torch.zeros(self.r_weight.shape), requires_grad=False)

        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features*4))
        else:
            self.register_parameter('bias', None)
        self.init_criterion = init_criterion
        self.weight_init = weight_init 
        self.seed = seed if seed is not None else np.random.randint(0,1234)
        self.rng = RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self):
        winit = {'quaternion': quaternion_init, 'unitary': unitary_init, 'random': random_init}[self.weight_init]
        if self.scale_param is not None:
            torch.nn.init.xavier_uniform_(self.scale_param.data)
        if self.bias is not None:
            self.bias.data.fill_(0)
        affect_init(self.r_weight, self.i_weight, self.j_weight, self.k_weight, winit,
                    self.rng, self.init_criterion)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        if self.rotation:
            return quaternion_linear_rotation(input, self.zero_kernel, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias, self.quaternion_format, self.scale_param)
        else:
            return quaternion_linear(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init=' + str(self.weight_init) \
            + ', rotation='       + str(self.rotation) \
            + ', seed=' + str(self.seed) + ')'

class QuaternionLinear(Module):
    r"""Applies a quaternion linear transformation to the incoming data.
    """

    def __init__(self, in_features, out_features, bias=True,
                 init_criterion='he', weight_init='quaternion',
                 seed=None):

        super(QuaternionLinear, self).__init__()
        self.in_features  = in_features//4
        self.out_features = out_features//4
        self.r_weight     = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.i_weight     = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.j_weight     = Parameter(torch.Tensor(self.in_features, self.out_features))
        self.k_weight     = Parameter(torch.Tensor(self.in_features, self.out_features))

        if bias:
            self.bias     = Parameter(torch.Tensor(self.out_features*4))
        else:
            self.register_parameter('bias', None)

        self.init_criterion = init_criterion
        self.weight_init    = weight_init
        self.seed           = seed if seed is not None else np.random.randint(0,1234)
        self.rng            = RandomState(self.seed)
        self.reset_parameters()

    def reset_parameters(self):
        winit = {'quaternion': quaternion_init,
                 'unitary': unitary_init}[self.weight_init]
        if self.bias is not None:
            self.bias.data.fill_(0)
        affect_init(self.r_weight, self.i_weight, self.j_weight, self.k_weight, winit,
                    self.rng, self.init_criterion)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        if input.dim() == 3:
            T, N, C = input.size()
            input  = input.view(T * N, C)
            output = QuaternionLinearFunction.apply(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias)
            output = output.view(T, N, output.size(1))
        elif input.dim() == 2:
            output = QuaternionLinearFunction.apply(input, self.r_weight, self.i_weight, self.j_weight, self.k_weight, self.bias)
        else:
            raise NotImplementedError

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) \
            + ', init_criterion=' + str(self.init_criterion) \
            + ', weight_init=' + str(self.weight_init) \
            + ', seed=' + str(self.seed) + ')'




def moving_average_update(statistic, curr_value, momentum):
    
    new_value = (1 - momentum) * statistic + momentum * curr_value
    
    return  new_value.data

class QuaternionBatchNorm2d(Module):
    r"""Applies a 2D Quaternion Batch Normalization to the incoming data.
        """

    def __init__(self, num_features, gamma_init=1., beta_param=True, momentum=0.1):
        super(QuaternionBatchNorm2d, self).__init__()
        self.num_features = num_features // 4
        self.gamma_init = gamma_init
        self.beta_param = beta_param
        self.gamma = Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
        self.beta = Parameter(torch.zeros(1, self.num_features * 4, 1, 1), requires_grad=self.beta_param)
        self.eps = torch.tensor(1e-5)
        
        self.register_buffer('moving_var', torch.ones(1) )
        self.register_buffer('moving_mean', torch.zeros(4))
        self.momentum = momentum

    def reset_parameters(self):
        self.gamma = Parameter(torch.full([1, self.num_features, 1, 1], self.gamma_init))
        self.beta = Parameter(torch.zeros(1, self.num_features * 4, 1, 1), requires_grad=self.beta_param)

    def forward(self, input):
        # print(self.training)
        if self.training:
            quat_components = torch.chunk(input, 4, dim=1)
    
            r, i, j, k = quat_components[0], quat_components[1], quat_components[2], quat_components[3]
            
            mu_r = torch.mean(r)
            mu_i = torch.mean(i)
            mu_j = torch.mean(j)
            mu_k = torch.mean(k)
            mu = torch.stack([mu_r,mu_i, mu_j, mu_k], dim=0)
            # print('mu shape', mu.shape)
            
            delta_r, delta_i, delta_j, delta_k = r - mu_r, i - mu_i, j - mu_j, k - mu_k
    
            quat_variance = torch.mean(delta_r**2 + delta_i**2 + delta_j**2 + delta_k**2)
            var = quat_variance
            denominator = torch.sqrt(quat_variance + self.eps)
            
            # Normalize
            r_normalized = delta_r / denominator
            i_normalized = delta_i / denominator
            j_normalized = delta_j / denominator
            k_normalized = delta_k / denominator
    
            beta_components = torch.chunk(self.beta, 4, dim=1)
    
            # Multiply gamma (stretch scale) and add beta (shift scale)
            new_r = (self.gamma * r_normalized) + beta_components[0]
            new_i = (self.gamma * i_normalized) + beta_components[1]
            new_j = (self.gamma * j_normalized) + beta_components[2]
            new_k = (self.gamma * k_normalized) + beta_components[3]
    
            new_input = torch.cat((new_r, new_i, new_j, new_k), dim=1)
        

            # with torch.no_grad():
            self.moving_mean.copy_(moving_average_update(self.moving_mean.data, mu.data, self.momentum))
            self.moving_var.copy_(moving_average_update(self.moving_var.data, var.data, self.momentum))
                
            # print(var, self.moving_var)

            
            return new_input
        
        else:
            with torch.no_grad():
                # print(input.shape, self.moving_mean.shape)
                r,i,j,k = torch.chunk(input, 4, dim=1)
                quaternions = [r,i,j,k]
                output = []
                denominator = torch.sqrt(self.moving_var + self.eps)
                beta_components = torch.chunk(self.beta, 4, dim=1)
                # print(torch.tensor(quaternions).shape)
                # print(quaternions[0].shape, self.moving_mean.shape, self.moving_var.shape, torch.squeeze(self.beta).shape)
                for q in range(4):
                    new_quat = self.gamma * ( (quaternions[q] - self.moving_mean[q]) / denominator ) + beta_components[q]
                    output.append(new_quat)
                output = torch.cat(output, dim=1)

                return  output 

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'num_features=' + str(self.num_features) \
               + ', gamma=' + str(self.gamma.shape) \
               + ', beta=' + str(self.beta.shape) \
               + ', eps=' + str(self.eps.shape) + ')'
               
       



##############################################################


class QSpectralNorm:
    # Invariant before and after each forward call:
    #   u = normalize(W @ v)
    # NB: At initialization, this invariant is not enforced

    _version: int = 1
    # At version 1:
    #   made  `W` not a buffer,
    #   added `v` as a buffer, and
    #   made eval mode use `W = u @ W_orig @ v` rather than the stored `W`.
    name: str
    dim: int
    n_power_iterations: int
    eps: float

    def __init__(self, name: str = 'weight', n_power_iterations: int = 1, dim: int = 0, eps: float = 1e-12) -> None:
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError('Expected n_power_iterations to be positive, but '
                             'got n_power_iterations={}'.format(n_power_iterations))
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(self.dim,
                                            *[d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    def compute_weight(self, module: Module, do_power_iteration: bool) -> torch.Tensor:
        # NB: If `do_power_iteration` is set, the `u` and `v` vectors are
        #     updated in power iteration **in-place**. This is very important
        #     because in `DataParallel` forward, the vectors (being buffers) are
        #     broadcast from the parallelized module to each module replica,
        #     which is a new module object created on the fly. And each replica
        #     runs its own spectral norm power iteration. So simply assigning
        #     the updated vectors to the module this function runs on will cause
        #     the update to be lost forever. And the next time the parallelized
        #     module is replicated, the same randomly initialized vectors are
        #     broadcast and used!
        #
        #     Therefore, to make the change propagate back, we rely on two
        #     important behaviors (also enforced via tests):
        #       1. `DataParallel` doesn't clone storage if the broadcast tensor
        #          is already on correct device; and it makes sure that the
        #          parallelized module is already on `device[0]`.
        #       2. If the out tensor in `out=` kwarg has correct shape, it will
        #          just fill in the values.
        #     Therefore, since the same power iteration is performed on all
        #     devices, simply updating the tensors in-place will make sure that
        #     the module replica on `device[0]` will update the _u vector on the
        #     parallized module (by shared storage).
        #
        #    However, after we update `u` and `v` in-place, we need to **clone**
        #    them before using them to normalize the weight. This is to support
        #    backproping through two forward passes, e.g., the common pattern in
        #    GAN training: loss = D(real) - D(fake). Otherwise, engine will
        #    complain that variables needed to do backward for the first forward
        #    (i.e., the `u` and `v` vectors) are changed in the second forward.
        # weight = getattr(module, self.name + '_orig')
        # weight = getattr(module, self.name)
        weight_r = getattr(module, 'r_weight' + '_orig')
        weight_i = getattr(module, 'i_weight' + '_orig')
        weight_j = getattr(module, 'j_weight' + '_orig')
        weight_k = getattr(module, 'k_weight' + '_orig')
        
        cat_kernels_4_r = torch.cat([weight_r, -weight_i, -weight_j, -weight_k], dim=1)
        cat_kernels_4_i = torch.cat([weight_i,  weight_r, -weight_k, weight_j], dim=1)
        cat_kernels_4_j = torch.cat([weight_j,  weight_k, weight_r, -weight_i], dim=1)
        cat_kernels_4_k = torch.cat([weight_k,  -weight_j, weight_i, weight_r], dim=1)
        weight = torch.cat([cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], dim=0)
        
        u = getattr(module, self.name + '_u')
        v = getattr(module, self.name + '_v')
        weight_mat = self.reshape_weight_to_matrix(weight)
        # print(module.r_weight)
        #print(u)
        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = normalize(torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v)
                    u = normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)

        sigma = torch.dot(u, torch.mv(weight_mat, v))
        # weight = weight / sigma
        weight_r = weight_r / sigma
        weight_i = weight_i / sigma
        weight_j = weight_j / sigma
        weight_k = weight_k / sigma
        # return weight
        
        return weight_r, weight_i, weight_j, weight_k

    # def remove(self, module: Module) -> None:
    #     with torch.no_grad():
    #         weight = self.compute_weight(module, do_power_iteration=False)
    #     delattr(module, self.name)
    #     delattr(module, self.name + '_u')
    #     delattr(module, self.name + '_v')
    #     delattr(module, self.name + '_orig')
    #     module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))

    def __call__(self, module: Module, inputs: Any) -> None:
        # setattr(module, self.name, self.compute_weight(module, do_power_iteration=module.training))
        weight_r, weight_i, weight_j, weight_k = self.compute_weight(module, do_power_iteration=module.training)
        setattr(module, 'r_weight', weight_r)
        setattr(module, 'i_weight', weight_i)
        setattr(module, 'j_weight', weight_j)
        setattr(module, 'k_weight', weight_k)
        # print('r after:', weight_r)

    def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
        # Tries to returns a vector `v` s.t. `u = normalize(W @ v)`
        # (the invariant at top of this class) and `u @ W @ v = sigma`.
        # This uses pinverse in case W^T W is not invertible.
        v = torch.chain_matmul(weight_mat.t().mm(weight_mat).pinverse(), weight_mat.t(), u.unsqueeze(1)).squeeze(1)
        return v.mul_(target_sigma / torch.dot(u, torch.mv(weight_mat, v)))

    @staticmethod
    def apply(module: Module, name: str, n_power_iterations: int, dim: int, eps: float) -> 'QSpectralNorm':
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, QSpectralNorm) and hook.name == name:
                raise RuntimeError("Cannot register two spectral_norm hooks on "
                                   "the same parameter {}".format(name))

        fn = QSpectralNorm(name, n_power_iterations, dim, eps)
        # weight = module._parameters[name]
        weight_r = module._parameters['r_weight']
        weight_i = module._parameters['i_weight']
        weight_j = module._parameters['j_weight']
        weight_k = module._parameters['k_weight']
        # weight = getattr(module, name)
        cat_kernels_4_r = torch.cat([weight_r, -weight_i, -weight_j, -weight_k], dim=1)
        cat_kernels_4_i = torch.cat([weight_i,  weight_r, -weight_k, weight_j], dim=1)
        cat_kernels_4_j = torch.cat([weight_j,  weight_k, weight_r, -weight_i], dim=1)
        cat_kernels_4_k = torch.cat([weight_k,  -weight_j, weight_i, weight_r], dim=1)
        weight = torch.cat([cat_kernels_4_r, cat_kernels_4_i, cat_kernels_4_j, cat_kernels_4_k], dim=0)
        # print(weight)
        # if isinstance(weight, torch.nn.parameter.UninitializedParameter):
        #     raise ValueError(
        #         'The module passed to `SpectralNorm` can\'t have uninitialized parameters. '
        #         'Make sure to run the dummy forward before applying spectral normalization')
        

        with torch.no_grad():
            # weight_mat = fn.reshape_weight_to_matrix(weight)
            weight_mat = fn.reshape_weight_to_matrix(weight)
            h, w = weight_mat.size()
            # randomly initialize `u` and `v`
            u = normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
            v = normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)

        #delattr(module, fn.name)
        delattr(module, 'r_weight')
        delattr(module, 'i_weight')
        delattr(module, 'j_weight')
        delattr(module, 'k_weight')
        
        #module.register_parameter(fn.name + "_orig", weight)
        module.register_parameter('r_weight' + "_orig", weight_r)
        module.register_parameter('i_weight' + "_orig", weight_i)
        module.register_parameter('j_weight' + "_orig", weight_j)
        module.register_parameter('k_weight' + "_orig", weight_k)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        # setattr(module, fn.name, weight.data)
        setattr(module, 'r_weight', weight_r.data)
        setattr(module, 'i_weight', weight_i.data)
        setattr(module, 'j_weight', weight_j.data)
        setattr(module, 'k_weight', weight_k.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)
        # print(module.r_weight)
        module.register_forward_pre_hook(fn)
        module._register_state_dict_hook(QSpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(QSpectralNormLoadStateDictPreHook(fn))
        return fn


# This is a top level class because Py2 pickle doesn't like inner class nor an
# instancemethod.
    
class QSpectralNormLoadStateDictPreHook:
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, fn) -> None:
        self.fn = fn

    # For state_dict with version None, (assuming that it has gone through at
    # least one training forward), we have
    #
    #    u = normalize(W_orig @ v)
    #    W = W_orig / sigma, where sigma = u @ W_orig @ v
    #
    # To compute `v`, we solve `W_orig @ x = u`, and let
    #    v = x / (u @ W_orig @ x) * (W / W_orig).
    def __call__(self, state_dict, prefix, local_metadata, strict,
                 missing_keys, unexpected_keys, error_msgs) -> None:
        fn = self.fn
        version = local_metadata.get('Qspectral_norm', {}).get(fn.name + '.version', None)
        print('QSpectralNormLoadStateDictPreHook')
        print(version)
        if version is None or version < 1:
            weight_key = prefix + fn.name
            if version is None and all(weight_key + s in state_dict for s in ('_orig', '_u', '_v')) and \
                    weight_key not in state_dict:
                # Detect if it is the updated state dict and just missing metadata.
                # This could happen if the users are crafting a state dict themselves,
                # so we just pretend that this is the newest.
                return
            has_missing_keys = False
            for suffix in ('_orig', '', '_u'):
                key = weight_key + suffix
                if key not in state_dict:
                    has_missing_keys = True
                    if strict:
                        missing_keys.append(key)
            if has_missing_keys:
                return
            with torch.no_grad():
                weight_orig = state_dict[weight_key + '_orig']
                weight = state_dict.pop(weight_key)
                sigma = (weight_orig / weight).mean()
                weight_mat = fn.reshape_weight_to_matrix(weight_orig)
                u = state_dict[weight_key + '_u']
                v = fn._solve_v_and_rescale(weight_mat, u, sigma)
                state_dict[weight_key + '_v'] = v


# This is a top level class because Py2 pickle doesn't like inner class nor an
# instancemethod.
class QSpectralNormStateDictHook:
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, fn) -> None:
        self.fn = fn

    def __call__(self, module, state_dict, prefix, local_metadata) -> None:
        print('QSpectralNormStateDictHook')
        if 'Qspectral_norm' not in local_metadata:
            local_metadata['Qspectral_norm'] = {}
        key = self.fn.name + '.version'
        if key in local_metadata['Qspectral_norm']:
            raise RuntimeError("Unexpected key in metadata['Qspectral_norm']: {}".format(key))
        local_metadata['Qspectral_norm'][key] = self.fn._version


T_module = TypeVar('T_module', bound=Module)

def Qspectral_norm(module: T_module,
                  name: str = 'weight',
                  n_power_iterations: int = 1,
                  eps: float = 1e-12,
                  dim: Optional[int] = None) -> T_module:
    r"""Applies spectral normalization to a parameter in the given module.
    .. math::
        \mathbf{W}_{SN} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})},
        \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}
    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.
    See `Spectral Normalization for Generative Adversarial Networks`_ .
    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957
    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``
    Returns:
        The original module with the spectral norm hook
    Example::
        >>> m = spectral_norm(nn.Linear(20, 40))
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_u.size()
        torch.Size([40])
    """
    if dim is None:
        if isinstance(module, (torch.nn.ConvTranspose1d,
                               torch.nn.ConvTranspose2d,
                               torch.nn.ConvTranspose3d)):
            dim = 1
        else:
            dim = 0
    QSpectralNorm.apply(module, name, n_power_iterations, dim, eps)
    return module


def remove_spectral_norm(module: T_module, name: str = 'weight') -> T_module:
    r"""Removes the spectral normalization reparameterization from a module.
    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter
    Example:
        >>> m = spectral_norm(nn.Linear(40, 10))
        >>> remove_spectral_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, QSpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            break
    else:
        raise ValueError("Qspectral_norm of '{}' not found in {}".format(
            name, module))

    for k, hook in module._state_dict_hooks.items():
        if isinstance(hook, QSpectralNormStateDictHook) and hook.fn.name == name:
            del module._state_dict_hooks[k]
            break

    for k, hook in module._load_state_dict_pre_hooks.items():
        if isinstance(hook, QSpectralNormLoadStateDictPreHook) and hook.fn.name == name:
            del module._load_state_dict_pre_hooks[k]
            break

    return module



from torch.nn.parameter import Parameter, UninitializedParameter
from torch import _weight_norm, norm_except_dim
from typing import Any, TypeVar
import warnings


class QWeightNorm:
    name: str
    dim: int

    def __init__(self, name: str, dim: int) -> None:
        if dim is None:
            dim = -1
        self.name = name
        self.dim = dim

    # TODO Make return type more specific
    def compute_weight(self,module: Module, name:str) -> Any:
        _g = getattr(module, name+'_g')
        _v = getattr(module, name + '_v')
        return _weight_norm(_v, _g, self.dim)
    
    @staticmethod
    def apply(module, name: str, dim: int) -> 'QWeightNorm':
        warnings.warn("torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.")

        for hook in module._forward_pre_hooks.values():
            if isinstance(hook, QWeightNorm) and hook.name == name:
                raise RuntimeError(f"Cannot register two weight_norm hooks on the same parameter {name}")

        if dim is None:
            dim = -1

        fn = QWeightNorm(name, dim)

        # weight = getattr(module, name)

        weight_r = getattr(module, 'r_weight')
        weight_i = getattr(module, 'i_weight')
        weight_j = getattr(module, 'j_weight')
        weight_k = getattr(module, 'k_weight')
        
        if isinstance(weight_r, UninitializedParameter) and isinstance(weight_i, UninitializedParameter) and isinstance(weight_j, UninitializedParameter) and isinstance(weight_k, UninitializedParameter):
            raise ValueError(
                'The module passed to `WeightNorm` can\'t have uninitialized parameters. '
                'Make sure to run the dummy forward before applying weight normalization')
        
        # remove w from parameter list
        # del module._parameters[name]
        del module._parameters['r_weight']
        del module._parameters['i_weight']
        del module._parameters['j_weight']
        del module._parameters['k_weight']


        # add g and v as new parameters and express w as g/||v|| * v
        module.register_parameter('r_weight_g', Parameter(norm_except_dim(weight_r, 2, dim).data))
        module.register_parameter('r_weight_v', Parameter(weight_r.data))
        module.register_parameter('i_weight_g', Parameter(norm_except_dim(weight_i, 2, dim).data))
        module.register_parameter('i_weight_v', Parameter(weight_i.data))
        module.register_parameter('j_weight_g', Parameter(norm_except_dim(weight_j, 2, dim).data))
        module.register_parameter('j_weight_v', Parameter(weight_j.data))
        module.register_parameter('k_weight_g', Parameter(norm_except_dim(weight_k, 2, dim).data))
        module.register_parameter('k_weight_v', Parameter(weight_k.data))

        setattr(module, 'r_weight', fn.compute_weight(module,'r_weight'))

        setattr(module,'i_weight', fn.compute_weight(module,'i_weight'))

        setattr(module,'j_weight', fn.compute_weight(module,'j_weight'))

        setattr(module, 'k_weight', fn.compute_weight(module,'k_weight'))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module: Module) -> None:
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_g']
        del module._parameters[self.name + '_v']
        setattr(module, self.name, Parameter(weight.data))

    def __call__(self, module: Module, inputs: Any) -> None:
        setattr(module, 'r_weight', self.compute_weight(module,'r_weight'))
        setattr(module, 'i_weight', self.compute_weight(module,'i_weight'))
        setattr(module, 'j_weight', self.compute_weight(module,'j_weight'))
        setattr(module, 'k_weight', self.compute_weight(module,'k_weight'))


# T_module = TypeVar('T_module', bound=Module)

def Qweight_norm(module: T_module, name: str = 'weight', dim: int = 0) -> T_module:
    r"""Applies weight normalization to a parameter in the given module.

    .. math::
         \mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}

    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. This replaces the parameter specified
    by :attr:`name` (e.g. ``'weight'``) with two parameters: one specifying the magnitude
    (e.g. ``'weight_g'``) and one specifying the direction (e.g. ``'weight_v'``).
    Weight normalization is implemented via a hook that recomputes the weight
    tensor from the magnitude and direction before every :meth:`~Module.forward`
    call.

    By default, with ``dim=0``, the norm is computed independently per output
    channel/plane. To compute a norm over the entire weight tensor, use
    ``dim=None``.

    See https://arxiv.org/abs/1602.07868

    .. warning::

        This function is deprecated.  Use :func:`torch.nn.utils.parametrizations.weight_norm`
        which uses the modern parametrization API.  The new ``weight_norm`` is compatible
        with ``state_dict`` generated from old ``weight_norm``.

        Migration guide:

        * The magnitude (``weight_g``) and direction (``weight_v``) are now expressed
          as ``parametrizations.weight.original0`` and ``parametrizations.weight.original1``
          respectively.  If this is bothering you, please comment on
          https://github.com/pytorch/pytorch/issues/102999

        * To remove the weight normalization reparametrization, use
          :func:`torch.nn.utils.parametrize.remove_parametrizations`.

        * The weight is no longer recomputed once at module forward; instead, it will
          be recomputed on every access.  To restore the old behavior, use
          :func:`torch.nn.utils.parametrize.cached` before invoking the module
          in question.

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter
        dim (int, optional): dimension over which to compute the norm

    Returns:
        The original module with the weight norm hook

    Example::

        >>> m = weight_norm(nn.Linear(20, 40), name='weight')
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_g.size()
        torch.Size([40, 1])
        >>> m.weight_v.size()
        torch.Size([40, 20])

    """
    QWeightNorm.apply(module, name, dim)
    return module



def remove_Qweight_norm(module: T_module, name: str = 'weight') -> T_module:
    r"""Removes the weight normalization reparameterization from a module.

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter

    Example:
        >>> m = weight_norm(nn.Linear(20, 40))
        >>> remove_weight_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, QWeightNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError(f"weight_norm of '{name}' not found in {module}")
