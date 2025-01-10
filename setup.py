#!/usr/bin/env python

import setuptools

setuptools.setup(
    name             = 'Rawformer',
    version          = '0.1.1',
    author           = 'Georgy Perevozchikov',
    author_email     = 'gosha20777@live.ru',
    classifiers      = [
        'Programming Language :: Python :: 3 :: Only',
    ],
    description      = "Unpaired Raw-to-Raw Translation for Learnable Camera ISPs",
    packages         = setuptools.find_packages(
        include = [ 'rawformer', 'rawformer.*' ]
    ),
    install_requires = [ 'numpy', 'pandas', 'tqdm', 'Pillow' ],
)

