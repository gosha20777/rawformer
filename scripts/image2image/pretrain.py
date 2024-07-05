import argparse
import os

from rawformer import ROOT_OUTDIR, train
from rawformer.presets import GEN_PRESETS
from rawformer.utils.parsers import add_preset_name_parser, add_batch_size_parser

def parse_cmdargs():
    parser = argparse.ArgumentParser(
        description = 'Pretrain Image2Image generators'
    )
    add_preset_name_parser(parser, 'gen', GEN_PRESETS, 'rawformer')
    add_batch_size_parser(parser, default = 32)
    return parser.parse_args()

cmdargs   = parse_cmdargs()
args_dict = {
    'batch_size' : cmdargs.batch_size,
    'data' : {
        'datasets' : [
            {
                'dataset' : {
                    'name'   : 'cyclegan',
                    'domain' : domain,
                    'path'   : 'image2image',
                },
                'shape'           : (3, 256, 256),
                'transform_train' : [
                    { 'name' : 'resize',          'size'    : 286, },
                    { 'name' : 'random-rotation', 'degrees' : 10,  },
                    { 'name' : 'random-crop',     'size'    : 256, },
                    'random-flip-horizontal',
                    {
                        'name' : 'color-jitter',
                        'brightness' : 0.2,
                        'contrast'   : 0.2,
                        'saturation' : 0.2,
                        'hue'        : 0.2,
                    },
                ],
                'transform_test' : None,
            } for domain in [ 'a', 'b' ]
        ],
        'merge_type' : 'unpaired',
        'workers'    : 1,
    },
    'epochs'        : 5, # 2500 epochs
    'discriminator' : None,
    'generator' : {
        **GEN_PRESETS[cmdargs.gen],
        'optimizer'  : {
            'name'  : 'AdamW',
            'lr'    : cmdargs.batch_size * 5e-3 / 512,
            'betas' : (0.9, 0.99),
            'weight_decay' : 0.05,
        },
        'weight_init' : {
            'name'      : 'normal',
            'init_gain' : 0.02,
        }
    },
    'model'      : 'autoencoder',
    'model_args' : {
        'joint'   : True,
        'masking' : {
            'name'       : 'image-patch-random',
            'patch_size' : (32, 32),
            'fraction'   : 0.4,
        },
    },
    'scheduler' : {
        'name'    : 'CosineAnnealingWarmRestarts',
        'T_0'     : 500,
        'T_mult'  : 1,
        'eta_min' : cmdargs.batch_size * 5e-8 / 512,
    },
    'loss'             : 'pixwise',
    'gradient_penalty' : None,
    'steps_per_epoch'  : 32 * 1024 // cmdargs.batch_size,
# args
    'label'      : f'pretrain-{cmdargs.gen}',
    'outdir'     : os.path.join(ROOT_OUTDIR, 'image2image'),
    'log_level'  : 'DEBUG',
    'checkpoint' : 100,
}

train(args_dict)
