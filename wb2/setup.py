"""
setup.py

Setup file used to stage all necessary dependencies for running our PC and WeatherBench2 evaluation jobs on Google Cloud Dataflow. 
"""

from setuptools import setup, find_packages

setup(
    name='pc',
    version='0.1.0',
    packages=find_packages(include=['pc']),
    install_requires=[
        'xarray>=2024.11.0',
        'xarray-beam',
        'zarr',
        'apache-beam[gcp]>=2.31.0',
        'gcsfs',
        'weatherbench2[gcp] @ git+https://github.com/google-research/weatherbench2.git@main#egg=weatherbench2',
        'isodisreg @ git+https://github.com/evwalz/isodisreg.git#egg=isodisreg'
    ]
)