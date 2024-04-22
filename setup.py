from setuptools import setup, find_packages

setup(
    name='pyssem',
    version='0.1',
    packages=find_packages(),
    package_data={
        'pyssem': ['utils/launch/data/*.csv'],
    },
    # Include any other package data below
)