import os

CONFIG_NAME = 'config.json'
ROOT_DATA   = os.environ.get('RAWFORMER_DATA',   'train-data')
ROOT_OUTDIR = os.environ.get('RAWFORMER_OUTDIR', 'outdir')

SPLIT_TRAIN = 'train'
SPLIT_VAL   = 'val'
SPLIT_TEST  = 'test'

MERGE_PAIRED   = 'paired'
MERGE_UNPAIRED = 'unpaired'
MERGE_NONE     = 'none'

MODEL_STATE_TRAIN = 'train'
MODEL_STATE_EVAL  = 'eval'
