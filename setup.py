#!/usr/bin/env python

import setuptools

VER = "0.0.1"

reqs = ["numpy",
        "scipy",
        "plotly",
        "pyaml",
        "h5py",
        # "larcv",
        # "torch",
        "timm==0.4.5",
        # "MinkowskiEngine",
        ]

setuptools.setup(
    name="LArDRIP",
    version=VER,
    author="Daniel D. and others",
    author_email="dougl215@slac.stanford.edu",
    description="The LArTPC Dead Region Inference Project",
    url="https://github.com/DanielMDouglas/LArDRIP",
    packages=setuptools.find_packages(),
    install_requires=reqs,
    classifiers=[
        "Development Status :: 1 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Physics"
    ],
    python_requires='>=3.2',
)
