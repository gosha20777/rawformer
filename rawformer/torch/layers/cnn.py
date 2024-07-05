from itertools import repeat
import torch
from torch import nn
from rawformer.torch.select import extract_name_kwargs

def calc_conv1d_output_size(input_size, kernel_size, padding, stride):
    return (input_size + 2 * padding - kernel_size) // stride + 1

def calc_conv_transpose1d_output_size(
    input_size, kernel_size, padding, stride
):
    return (input_size - 1) * stride - 2 * padding + kernel_size

def calc_conv_output_size(input_size, kernel_size, padding, stride):
    if isinstance(kernel_size, int):
        kernel_size = repeat(kernel_size, len(input_size))

    if isinstance(stride, int):
        stride = repeat(stride, len(input_size))

    if isinstance(padding, int):
        padding = repeat(padding, len(input_size))

    return tuple(
        calc_conv1d_output_size(sz, ks, p, s)
            for (sz, ks, p, s) in zip(input_size, kernel_size, padding, stride)
    )

def calc_conv_transpose_output_size(input_size, kernel_size, padding, stride):
    if isinstance(kernel_size, int):
        kernel_size = repeat(kernel_size, len(input_size))

    if isinstance(stride, int):
        stride = repeat(stride, len(input_size))

    if isinstance(padding, int):
        padding = repeat(padding, len(input_size))

    return tuple(
        calc_conv_transpose1d_output_size(sz, ks, p, s)
            for (sz, ks, p, s) in zip(input_size, kernel_size, padding, stride)
    )



class CDown(nn.Module):
    def __init__(self, features):
        super().__init__()
        in_features = features
        out_features = 4 * features

        self.conv = nn.Conv2d(
            in_features, out_features, kernel_size = 2, stride = 2
        )

        self.pixel_unshuffle = nn.PixelUnshuffle(downscale_factor = 2)
        self.out_conv = nn.Conv2d(
            out_features, in_features, kernel_size = 3
        )

    def forward(self, x):
        x_c = self.conv(x)
        x_s = self.pixel_unshuffle(x)
        x = x_c + x_s
        x = self.out_conv(x)
        return x
    

class CUp(nn.Module):
    def __init__(self, features):
        super().__init__()
        in_features = features
        out_features = features // 4

        self.conv = nn.ConvTranspose2d(
            in_features, out_features, kernel_size = 2, stride = 2,
        )

        self.pixel_shuffle = nn.PixelShuffle(upscale_factor = 2)
        self.out_conv = nn.Conv2d(
            out_features, in_features, kernel_size = 3
        )

    def forward(self, x):
        x_c = self.conv(x)
        x_s = self.pixel_shuffle(x)
        x = x_c + x_s
        x = self.out_conv(x)
        return x


def get_downsample_x2_conv2_layer(features, **kwargs):
    return (
        nn.Conv2d(features, features, kernel_size = 2, stride = 2, **kwargs),
        features
    )

def get_downsample_x2_conv3_layer(features, **kwargs):
    return (
        nn.Conv2d(
            features, features, kernel_size = 3, stride = 2, padding = 1,
            **kwargs
        ),
        features
    )

def get_downsample_x2_pixelshuffle_layer(features, **kwargs):
    out_features = 4 * features
    return (nn.PixelUnshuffle(downscale_factor = 2, **kwargs), out_features)

def get_downsample_x2_pixelshuffle_conv_layer(features, **kwargs):
    out_features = features * 4

    layer = nn.Sequential(
        nn.PixelUnshuffle(downscale_factor = 2, **kwargs),
        nn.Conv2d(
            out_features, out_features, kernel_size = 3, padding = 1
        ),
    )

    return (layer, out_features)

def get_cdown_layer(features, **kwargs):
    out_features = features

    layer = CDown(features)

    return (layer, out_features)

def get_upsample_x2_deconv2_layer(features, **kwargs):
    return (
        nn.ConvTranspose2d(
            features, features, kernel_size = 2, stride = 2, **kwargs
        ),
        features
    )

def get_upsample_x2_upconv_layer(features, **kwargs):
    layer = nn.Sequential(
        nn.Upsample(scale_factor = 2, **kwargs),
        nn.Conv2d(features, features, kernel_size = 3, padding = 1),
    )

    return (layer, features)

def get_upsample_x2_pixelshuffle_conv_layer(features, **kwargs):
    out_features = features // 4

    layer = nn.Sequential(
        nn.PixelShuffle(upscale_factor = 2, **kwargs),
        nn.Conv2d(out_features, out_features, kernel_size = 3, padding = 1),
    )

    return (layer, out_features)


def get_cup_layer(features, **kwargs):
    out_features = features

    layer = CUp(features)

    return (layer, out_features)


def get_downsample_x2_layer(layer, features):
    name, kwargs = extract_name_kwargs(layer)

    if name == 'conv':
        return get_downsample_x2_conv2_layer(features, **kwargs)

    if name == 'conv3':
        return get_downsample_x2_conv3_layer(features, **kwargs)

    if name == 'avgpool':
        return (nn.AvgPool2d(kernel_size = 2, stride = 2, **kwargs), features)

    if name == 'maxpool':
        return (nn.MaxPool2d(kernel_size = 2, stride = 2, **kwargs), features)

    if name == 'pixel-unshuffle':
        return get_downsample_x2_pixelshuffle_layer(features, **kwargs)

    if name == 'pixel-unshuffle-conv':
        return get_downsample_x2_pixelshuffle_conv_layer(features, **kwargs)
    
    if name == 'cdown':
        return get_cdown_layer(features, **kwargs)

    raise ValueError("Unknown Downsample Layer: '%s'" % name)

def get_upsample_x2_layer(layer, features):
    name, kwargs = extract_name_kwargs(layer)

    if name == 'deconv':
        return get_upsample_x2_deconv2_layer(features, **kwargs)

    if name == 'upsample':
        return (nn.Upsample(scale_factor = 2, **kwargs), features)

    if name == 'upsample-conv':
        return get_upsample_x2_upconv_layer(features, **kwargs)

    if name == 'pixel-shuffle':
        return (nn.PixelShuffle(upscale_factor = 2, **kwargs), features // 4)

    if name == 'pixel-shuffle-conv':
        return get_upsample_x2_pixelshuffle_conv_layer(features, **kwargs)

    if name == 'cup':
        return get_cup_layer(features, **kwargs)

    raise ValueError("Unknown Upsample Layer: '%s'" % name)

