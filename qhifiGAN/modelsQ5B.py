import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import remove_weight_norm, spectral_norm
from torch.nn.utils.parametrizations import weight_norm
from utils import init_weights, get_padding
from torchinfo import summary
# from torchinfo import summary

LRELU_SLOPE = 0.1

from quaternion_layers import QuaternionConv2d, QuaternionConv,QuaternionTransposeConv2D,QuaternionTransposeConv,Qspectral_norm,Qweight_norm

class QResBlock1(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(QResBlock1, self).__init__()
        self.h = h
        self.convs1 = nn.ModuleList([
            Qweight_norm(QuaternionConv(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0]))),
                            #    nn.batchnorm(channels)
            Qweight_norm(QuaternionConv(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))),
            Qweight_norm(QuaternionConv(channels, channels, kernel_size, 1, dilation=dilation[2],
                               padding=get_padding(kernel_size, dilation[2])))
        ])
        # self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            Qweight_norm(QuaternionConv(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
            Qweight_norm(QuaternionConv(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1))),
           Qweight_norm(QuaternionConv(channels, channels, kernel_size, 1, dilation=1,
                               padding=get_padding(kernel_size, 1)))
        ])
        # self.convs2.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    # def remove_weight_norm(self):
    #     for l in self.convs1:
    #         remove_weight_norm(l)
    #     for l in self.convs2:
    #         remove_weight_norm(l)


# model  = QResBlock1(h="mkb",channels=4, kernel_size=3)
# print(model( torch.rand(1, 4,16000) ))

# print(summary(model, (1,4,16000)))

class QResBlock2(torch.nn.Module):
    def __init__(self, h, channels, kernel_size=3, dilation=(1, 3)):
        super(QResBlock2, self).__init__()
        self.h = h
        self.convs = nn.ModuleList([
            QuaternionConv(channels, channels, kernel_size, 1, dilation=dilation[0],
                               padding=get_padding(kernel_size, dilation[0])),
            QuaternionConv(channels, channels, kernel_size, 1, dilation=dilation[1],
                               padding=get_padding(kernel_size, dilation[1]))
        ])
        # self.convs.apply(init_weights)

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    # def remove_weight_norm(self):
    #     for l in self.convs:
    #         remove_weight_norm(l)


# model  = QResBlock2(h="mkb",channels=4, kernel_size=3)
# print(model( torch.rand(1, 4,16000) ))



class QGenerator(torch.nn.Module):
    def __init__(self, h):
        super(QGenerator, self).__init__()
        self.h = h
        self.num_kernels = len(h.resblock_kernel_sizes)
        self.num_upsamples = len(h.upsample_rates)
        self.conv_pre = QuaternionConv(320, h.upsample_initial_channel, 7, 1, padding=3)

        resblock = QResBlock1 if h.resblock == '1' else QResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(h.upsample_rates, h.upsample_kernel_sizes)):
            self.ups.append(
                QuaternionTransposeConv(in_channels=h.upsample_initial_channel//(2**i),out_channels= h.upsample_initial_channel//(2**(i+1)),
                               kernel_size= k, stride=u, padding=(k-u)//2, operation='convolution1d'))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = h.upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(h.resblock_kernel_sizes, h.resblock_dilation_sizes)):
                self.resblocks.append(resblock(h, ch, k, d))
        self.conv_post = nn.Sequential(
            weight_norm(nn.Conv1d(in_channels=ch, out_channels=ch//2, kernel_size=7, stride=1, padding=3, dilation=1)),
             weight_norm(nn.Conv1d(in_channels=ch//2, out_channels=1, kernel_size=5, stride=1, padding=2, dilation=1))
        )
         # adapter from Quaternion to Real
        # self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x):
        # print("inside generator",x.shape)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    # def remove_weight_norm(self):
    #     print('Removing weight norm...')
    #     for l in self.ups:
    #         remove_weight_norm(l)
    #     for l in self.resblocks:
    #         l.remove_weight_norm()
    #     remove_weight_norm(self.conv_pre)
    #     remove_weight_norm(self.conv_post)

# h_dic ={
#     "resblock": "1",
#     "num_gpus": 0,
#     "batch_size": 16,
#     "learning_rate": 0.0002,
#     "adam_b1": 0.8,
#     "adam_b2": 0.99,
#     "lr_decay": 0.999,
#     "seed": 1234,

#     "upsample_rates": [8,8,2,2],
#     "upsample_kernel_sizes": [16,16,4,4],
#     "upsample_initial_channel": 512,
#     'resblock_kernel_sizes': [3,7,11],
#     "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],

#     "segment_size": 8192,
#     "num_mels": 80,
#     "num_freq": 1025,
#     "n_fft": 1024,
#     "hop_size": 256,
#     "win_size": 1024,

#     "sampling_rate": 22050,

#     "fmin": 0,
#     "fmax": 8000,
#     "fmax_for_loss": None,

#     "num_workers": 4,

#     "dist_config": {
#         "dist_backend": "nccl",
#         "dist_url": "tcp://localhost:54321",
#         "world_size": 1
#     }
# }

# h_dic = {
#     "resblock": "1",
#     "batch_size": 16,
#     "learning_rate": 0.0002,
#     "adam_b1": 0.8,
#     "adam_b2": 0.99,
#     "lr_decay": 0.999,
#     "seed": 1234,

#     "upsample_rates": [8,8,2,2],
#     "upsample_kernel_sizes": [16,16,4,4],
#     "upsample_initial_channel": 128,
#     "resblock_kernel_sizes": [3,7,11],
#     "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],

#     "segment_size": 8192,
#     "num_mels": 80,
#     "num_freq": 1025,
#     "n_fft": 1024,
#     "hop_size": 256,
#     "win_size": 1024,

#     "sampling_rate": 22050,

#     "fmin": 0,
#     "fmax": 8000,
#     "fmax_for_loss": 0,
#     "num_workers": 4
# }

# h_dic = {
#     "resblock": "2",
#     "batch_size": 16,
#     "learning_rate": 0.0002,
#     "adam_b1": 0.8,
#     "adam_b2": 0.99,
#     "lr_decay": 0.999,
#     "seed": 1234,

#     "upsample_rates": [8,8,4],
#     "upsample_kernel_sizes": [16,16,8],
#     "upsample_initial_channel": 256,
#     "resblock_kernel_sizes": [3,5,7],
#     "resblock_dilation_sizes": [[1,2], [2,6], [3,12]],

#     "segment_size": 8192,
#     "num_mels": 80,
#     "num_freq": 1025,
#     "n_fft": 1024,
#     "hop_size": 256,
#     "win_size": 1024,

#     "sampling_rate": 22050,

#     "fmin": 0,
#     "fmax": 8000,
#     "fmax_for_loss": 0,

#     "num_workers": 4
# }


# class Configuration:
#     pass

# h = Configuration()

# for key, value in h_dic.items():
#     setattr(h, key, value)

# print(h.resblock_kernel_sizes)


# model  = QGenerator(h =h)
# # print(model( torch.rand(1, 320,16000) ))
# print(summary(model, (1,320,16000)))

class QDiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(QDiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm

        self.h2r = norm_f(Conv2d(1, 4, (3, 1), (1,1), padding=(1, 0))) # adapter from  Real to Quaternion 


        self.convs = nn.ModuleList([
           Qspectral_norm( QuaternionConv2d(4, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            Qspectral_norm(QuaternionConv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            Qspectral_norm(QuaternionConv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            Qspectral_norm(QuaternionConv2d(512, 1048, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            Qspectral_norm(QuaternionConv2d(1048, 1048, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1048, 1, (3, 1), 1, padding=(1, 0))) # adapter from Quaternion to Real

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        x = self.h2r(x)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

# model = QDiscriminatorP(period=2)
# print(model(torch.rand(1,1,16000)))


class QMultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(QMultiPeriodDiscriminator, self).__init__()

        self.discriminators = nn.ModuleList([
            QDiscriminatorP(2),
            QDiscriminatorP(3),
            QDiscriminatorP(5),
            QDiscriminatorP(7),
            QDiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

# model = QMultiPeriodDiscriminator()
# y=torch.rand(1,1,16000)
# y_hat = torch.rand(1,1,16000)
# print(model(y,y_hat))

# print(summary(model,((1,1,16000),(1,1,16000))))    
    

    

class QDiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(QDiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.h2r = norm_f(Conv1d(1, 4, 3, 1, padding=1)) # adapter from  Real to Quaternion 
        self.convs = nn.ModuleList(
            [
            Qspectral_norm(QuaternionConv(4, 128, 15, 1, padding=7)),
            Qspectral_norm(QuaternionConv(128, 128, 41, 2, groups=1, padding=20)),
            Qspectral_norm(QuaternionConv(128, 256, 41, 2, groups=1, padding=20)),
            Qspectral_norm(QuaternionConv(256, 512, 41, 4, groups=1, padding=20)),
            Qspectral_norm(QuaternionConv(512, 1024, 41, 4, groups=1, padding=20)),
            # Qspectral_norm(QuaternionConv(1024, 1024, 41, 1, groups=1, padding=20)),
            Qspectral_norm(QuaternionConv(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))  # Quaternion to Real adapter

    def forward(self, x):
        fmap = []
        x = self.h2r(x)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

# model = QDiscriminatorS()
# print(model(torch.rand(1,1,16000)))

class QMultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(QMultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            QDiscriminatorS(use_spectral_norm=True),
            QDiscriminatorS(),
            QDiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
    

# model =QMultiScaleDiscriminator()
# y=torch.rand(1,1,16000)
# y_hat = torch.rand(1,1,16000)
# print(model(y,y_hat))
# print("bhai ye kya hain bhai ?")
# print(summary(model,((1,1,16000),(1,1,16000))))

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses





# print(summary(m, (32,1,40,32)))