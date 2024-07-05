

GEN_PRESETS = {
    'resnet9' : {
        'model'      : 'resnet_9blocks',
        'model_args' : None,
    },
    'unet' : {
        'model'      : 'unet_256',
        'model_args' : None,
    },
    'uvcgan' : {
        'model' : 'vit-unet',
        'model_args' : {
            'features'           : 384,
            'n_heads'            : 6,
            'n_blocks'           : 12,
            'ffn_features'       : 1536,
            'embed_features'     : 384,
            'activ'              : 'gelu',
            'norm'               : 'layer',
            'unet_features_list' : [48, 96, 192, 384],
            'unet_activ'         : 'leakyrelu',
            'unet_norm'          : 'instance',
            'unet_downsample'    : 'conv',
            'unet_upsample'      : 'upsample-conv',
            'rezero'             : True,
            'activ_output'       : 'sigmoid',
        },
    },
    'uvcgan2' : {
        'model' : 'vit-modnet',
        'model_args' : {
            'features'             : 384,
            'n_heads'              : 6,
            'n_blocks'             : 12,
            'ffn_features'         : 1536,
            'embed_features'       : 384,
            'activ'                : 'gelu',
            'norm'                 : 'layer',
            'modnet_features_list' : [48, 96, 192, 384],
            'modnet_activ'         : 'leakyrelu',
            'modnet_norm'          : None,
            'modnet_downsample'    : 'conv',
            'modnet_upsample'      : 'upsample-conv',
            'modnet_rezero'        : False,
            'modnet_demod'         : True,
            'rezero'               : True,
            'activ_output'         : 'sigmoid',
            'style_rezero'         : True,
            'style_bias'           : True,
            'n_ext'                : 1,
        },
    },
    'rawformer' : {
        'model' : 'vit-rawnet',
        'model_args' : {
            'features'             : 192, # 192
            'n_heads'              : 6, # 6
            'n_blocks'             : 12, # 12
            'ffn_features'         : 768, # 768
            'embed_features'       : 192, # 192
            'activ'                : 'gelu',
            'norm'                 : 'layer',
            'modnet_features_list' : [24, 48, 96, 192],
            'modnet_activ'         : 'leakyrelu',
            'modnet_norm'          : None,
            'modnet_downsample'    : 'conv', # 'conv'
            'modnet_upsample'      : 'upsample-conv', # 'upsample-conv'
            'modnet_rezero'        : False,
            'modnet_demod'         : True,
            'rezero'               : True,
            'activ_output'         : 'sigmoid',
            'style_rezero'         : True,
            'style_bias'           : True,
            'n_ext'                : 1,
            'rawnet_num_heads'     : 1,
            'rawnet_token_size'    : 4,
        },
    },
}

BH_PRESETS = {
    'bn'  : 'batch-norm-2d',
    'bsd' : 'batch-stdev',
}

