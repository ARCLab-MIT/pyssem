from setuptools import setup, find_packages
import setuptools_scm

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='pyssem',
    use_scm_version={
        "local_scheme": "no-local-version" # STOPS LOCAL HASH BEING ADDED
                     },
    setup_requires=['setuptools_scm'],
    author='Indigo Brownhall',
    author_email='indigo.brownhall.20@ucl.ac.uk',
    url='https://github.com/ARCLab-MIT/pyssem',
    packages=find_packages(),
    package_data={
        'pyssem': ['utils/launch/data/*.csv'],
    },
    long_description=long_description,
    long_description_content_type='text/markdown', 
    install_requires=[
        "numpy~=1.21",
        "pandas~=2.0",
        "scipy~=1.10",
        "setuptools~=68.0.0", 
        "sympy~=1.11",
        "tqdm~=4.65"
    ],
    extras_require={
        'dev': ['pytest', 'check-manifest'],
        'test': ['coverage'],
    },
    python_requires='>=3.6',  # Define which Python versions are supported
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License', 
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
