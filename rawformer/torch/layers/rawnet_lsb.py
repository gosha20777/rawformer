# pylint: disable=too-many-arguments
# pylint: disable=too-many-instance-attributes

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from rawformer.torch.select import get_activ_layer, get_norm_layer
from .cnn  import get_downsample_x2_layer, get_upsample_x2_layer


def get_demod_scale(mod_scale, weights, eps = 1e-6):
    # Ref: https://arxiv.org/pdf/1912.04958.pdf
    #
    # demod_scale[alpha] = 1 / sqrt(sigma[alpha]^2 + eps)
    #
    # sigma[alpha]^2
    #   = sum_{beta i} (mod_scale[alpha]  * weights[alpha, beta, i])^2
    #   = sum_{beta} (mod_scale[alpha])^2 * sum_i (weights[alpha, beta, i])^2
    #

    # mod_scale : (N, C_in)
    # weights   : (C_out, C_in, h, w)

    # w_sq : (C_out, C_in)
    w_sq = torch.sum(weights.square(), dim = (2, 3))

    # w_sq : (C_out, C_in) -> (1, C_in, C_out)
    w_sq = torch.swapaxes(w_sq, 0, 1).unsqueeze(0)

    # mod_scale_sq : (N, C_in, 1)
    mod_scale_sq = mod_scale.square().unsqueeze(2)

    # sigma : (N, C_out)
    sigma_sq = torch.sum(mod_scale_sq * w_sq, dim = 1)

    # result : (N, C_out)
    return 1 / torch.sqrt(sigma_sq + eps)


class SA(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(SA, self).__init__()
        self.num_heads = num_heads
        #self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature = 1 / torch.sqrt(torch.tensor(dim, dtype=torch.float32))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # transposed self-attention with attention map of shape (C×C)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class ModSA(nn.Module):
    def __init__(self, dim, num_heads, bias,
                 eps = 1e-6, demod = True):
        super(ModSA, self).__init__()
        self.num_heads = num_heads
        #self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperature = 1 / torch.sqrt(torch.tensor(dim, dtype=torch.float32))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self._eps   = eps
        self._demod = demod

    def forward(self, x, s):
        b, c, h, w = x.shape
        x = x * s.unsqueeze(2).unsqueeze(3)

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # transposed self-attention with attention map of shape (C×C)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)

        if self._demod:
            # s_demod : (N, C_out)
            s_demod = get_demod_scale(s, self.project_out.weight)
            out_demod = out * s_demod.unsqueeze(2).unsqueeze(3)
            return out_demod

        return out


class FFN(nn.Module):
    """
    Feed-forward Network with Depth-wise Convolution
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.pointwise1 = torch.nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.depthwise = torch.nn.Conv2d(hidden_features,hidden_features, kernel_size=3,stride=1,padding=1,dilation=1,groups=hidden_features)
        self.pointwise2 = torch.nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.act_layer = act_layer() if act_layer is not None else nn.Identity()
    def forward(self, x):
        x = self.pointwise1(x)
        x = self.depthwise(x)
        x = self.act_layer(x)
        x = self.pointwise2(x)
        return x


class StyleBlock(nn.Module):

    def __init__(
        self, mod_features, style_features, rezero = True, bias = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.affine_mod = nn.Linear(mod_features, style_features, bias = bias)
        self.rezero     = rezero

        if rezero:
            self.re_alpha = nn.Parameter(torch.zeros((1, )))
        else:
            self.re_alpha = 1

    def forward(self, mod):
        # mod : (N, mod_features)
        # s   : (N, style_features)
        s = 1 + self.re_alpha * self.affine_mod(mod)

        return s

    def extra_repr(self):
        return 're_alpha = %e' % (self.re_alpha, )


class RawFormerBlock(nn.Module):
    def __init__(
        self, in_features, out_features, activ, norm, mid_features = None,
        num_heads = 1, **kwargs
    ):
        super().__init__(**kwargs)

        if mid_features is None:
            mid_features = out_features


        self.norm1 = get_norm_layer(norm, in_features)
        self.attn = SA(in_features, num_heads, bias=True)
        self.dw = torch.nn.Conv2d(in_features, in_features, kernel_size=3,stride=1,padding=1,dilation=1,groups=in_features)
        self.pw = torch.nn.Conv2d(in_features*2, in_features, kernel_size=1)
        self.norm2 = get_norm_layer(norm, in_features)
        self.ffn = FFN(in_features, mid_features, in_features)
        self.out_conv = nn.Conv2d(
                in_features, out_features, kernel_size = 3, padding = 1
            )
        self.act = get_activ_layer(activ)

    def forward(self, x):
        #x = x + self.attn(self.norm1(x))
        #x = x + self.ffn(self.norm2(x))
        x_norm = self.norm1(x)
        x_atten = self.attn(x_norm)
        x_dw = self.act(self.dw(x_norm))
        x_atten_dw = self.pw(torch.cat([x_atten, x_dw], dim=1))
        x = x + x_atten_dw
        x = x + self.ffn(self.norm2(x))
        x = self.act(self.out_conv(x))
        return x
    

class ModRawFormerBlock(nn.Module):
    def __init__(
        self, in_features, out_features, activ, norm, mod_features,
        mid_features = None,
        demod        = True,
        style_rezero = True,
        style_bias   = True,
        num_heads=1,
        **kwargs
    ):
        super().__init__(**kwargs)

        if mid_features is None:
            mid_features = out_features

        self.style_block = StyleBlock(
            mod_features, in_features, style_rezero, style_bias
        )
        self.norm1 = get_norm_layer(norm, in_features)
        self.attn = ModSA(in_features, num_heads, bias=True, demod=demod)
        self.dw = torch.nn.Conv2d(in_features, in_features, kernel_size=3,stride=1,padding=1,dilation=1,groups=in_features)
        self.pw = torch.nn.Conv2d(in_features*2, in_features, kernel_size=1)
        self.norm2 = get_norm_layer(norm, in_features)
        self.ffn = FFN(in_features, mid_features, in_features)
        self.out_conv = nn.Conv2d(
                in_features, out_features, kernel_size = 3, padding = 1
            )
        self.act = get_activ_layer(activ)

    def forward(self, x, mod):
        # x   : (N, C_in, H_in, W_in)
        # mod : (N, mod_features)

        # mod_scale : (N, C_out)
        mod_scale = self.style_block(mod)

        x_norm = self.norm1(x)
        x_atten = self.attn(x_norm, mod_scale)
        x_dw = self.act(self.dw(x_norm))
        x_atten_dw = self.pw(torch.cat([x_atten, x_dw], dim=1))
        x = x + x_atten_dw
        x = x + self.ffn(self.norm2(x))
        x = self.act(self.out_conv(x))
        # result : (N, C_out, H_out, W_out)
        return x


class DBlock(nn.Module):
    def __init__(
        self, features, activ, norm, downsample, input_shape, num_heads = 1, **kwargs
    ):
        super().__init__(**kwargs)

        self.downsample, output_features = \
            get_downsample_x2_layer(downsample, features)

        (C, H, W)  = input_shape
        self.block = RawFormerBlock(C, features, activ, norm, num_heads=num_heads)

        self._output_shape = (output_features, H//2, W//2)

    @property
    def output_shape(self):
        return self._output_shape

    def forward(self, x):
        r = self.block(x)
        y = self.downsample(r)
        return (y, r)


class UBlock(nn.Module):
    def __init__(
        self, input_shape, output_features, skip_features, mod_features,
        activ, norm, upsample,
        rezero       = True,
        demod        = True,
        style_rezero = True,
        style_bias   = True,
        num_heads = 1,
        **kwargs
    ):
        super().__init__(**kwargs)

        (input_features, H, W) = input_shape
        self.upsample, input_features = get_upsample_x2_layer(
            upsample, input_features
        )

        self.block = ModRawFormerBlock(
            skip_features + input_features, output_features, activ,
            norm         = norm,
            mod_features = mod_features,
            mid_features = max(input_features, input_shape[0]),
            demod        = demod,
            style_rezero = style_rezero,
            style_bias   = style_bias,
            num_heads    = num_heads,
        )

        self._output_shape = (output_features, 2 * H, 2 * W)

        if rezero:
            self.re_alpha = nn.Parameter(torch.zeros((1, )))
        else:
            self.re_alpha = 1

    @property
    def output_shape(self):
        return self._output_shape

    def forward(self, x, r, mod):
        # x   : (N, C, H_in, W_in)
        # r   : (N, C, H_out, W_out)
        # mod : (N, mod_features)

        # x : (N, C_up, H_out, W_out)
        x = self.re_alpha * self.upsample(x)

        # y : (N, C + C_up, H_out, W_out)
        y = torch.cat([x, r], dim = 1)

        # result : (N, C_out, H_out, W_out)
        return self.block(y, mod)

    def extra_repr(self):
        return 're_alpha = %e' % (self.re_alpha, )


class RawNetBlock(nn.Module):
    def __init__(
        self, features, activ, norm, image_shape, downsample, upsample,
        mod_features,
        rezero       = True,
        demod        = True,
        style_rezero = True,
        style_bias   = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.conv = DBlock(
            features, activ, norm, downsample, image_shape
        )

        self.inner_shape  = self.conv.output_shape
        self.inner_module = None

        self.deconv = UBlock(
            input_shape     = self.inner_shape,
            output_features = image_shape[0],
            skip_features   = self.inner_shape[0],
            mod_features    = mod_features,
            activ           = activ,
            norm            = norm,
            upsample        = upsample,
            rezero          = rezero,
            demod           = demod,
            style_rezero    = style_rezero,
            style_bias      = style_bias,
        )

    def get_inner_shape(self):
        return self.inner_shape

    def set_inner_module(self, module):
        self.inner_module = module

    def get_inner_module(self):
        return self.inner_module

    def forward(self, x):
        # x : (N, C, H, W)

        # y : (N, C_inner, H_inner, W_inner)
        # r : (N, C_inner, H, W)
        (y, r) = self.conv(x)

        # y   : (N, C_inner, H_inner, W_inner)
        # mod : (N, mod_features)
        y, mod = self.inner_module(y)

        # y : (N, C, H, W)
        y = self.deconv(y, r, mod)

        return (y, mod)

class RawNet(nn.Module):

    def __init__(
        self, features_list, activ, norm, image_shape, downsample, upsample,
        mod_features,
        rezero       = True,
        demod        = True,
        style_rezero = True,
        style_bias   = True,
        return_mod   = False,
        **kwargs
    ):
        # pylint: disable = too-many-locals
        super().__init__(**kwargs)

        self.features_list = features_list
        self.image_shape   = image_shape
        self.return_mod    = return_mod

        self._construct_input_layer(activ)
        self._construct_output_layer()

        unet_layers = []
        curr_image_shape = (features_list[0], *image_shape[1:])

        for features in features_list:
            layer = RawNetBlock(
                features, activ, norm, curr_image_shape, downsample, upsample,
                mod_features, rezero, demod, style_rezero, style_bias
            )
            curr_image_shape = layer.get_inner_shape()
            unet_layers.append(layer)

        for idx in range(len(unet_layers)-1):
            unet_layers[idx].set_inner_module(unet_layers[idx+1])

        self.modnet = unet_layers[0]

    def _construct_input_layer(self, activ):
        self.layer_input = nn.Sequential(
            nn.Conv2d(
                self.image_shape[0], self.features_list[0],
                kernel_size = 3, padding = 1
            ),
            get_activ_layer(activ),
        )

    def _construct_output_layer(self):
        self.layer_output = nn.Conv2d(
            self.features_list[0], self.image_shape[0], kernel_size = 1
        )

    def get_innermost_block(self):
        result = self.modnet

        for _ in range(len(self.features_list)-1):
            result = result.get_inner_module()

        return result

    def set_bottleneck(self, module):
        self.get_innermost_block().set_inner_module(module)

    def get_bottleneck(self):
        return self.get_innermost_block().get_inner_module()

    def get_inner_shape(self):
        return self.get_innermost_block().get_inner_shape()

    def forward(self, x):
        # x : (N, C, H, W)

        y = self.layer_input(x)
        y, mod = self.modnet(y)
        y = self.layer_output(y)

        if self.return_mod:
            return (y, mod)

        return y

