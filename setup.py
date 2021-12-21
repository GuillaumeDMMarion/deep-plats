#!/usr/bin/env python3

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()

setup(
    name='deep-plats',
    version='0.1',
    license="MIT",
    author='Guillaume Marion',
    description='Deep piecewise linear analysis of time series',
    packages=['deepplats'],
    keywords=[
        "deep learning",
        "neural network",
        "time series",
        "forecasting",
        "piecewise",
        "linear",
    ],
    url='http://github.com/GuillaumeDMMarion/deep-plats',
    install_requires=install_requires,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
