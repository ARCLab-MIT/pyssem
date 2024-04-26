from setuptools import setup, find_packages

setup(
    name='pyssem',
    version='0.1',
    packages=find_packages(),
    package_data={
        'pyssem': ['utils/launch/data/*.csv'],
    },
    install_requires=[
        "numpy==1.24.3",
        "pandas==2.0.3",
        "scipy==1.10.1",
        "setuptools==68.0.0",
        "sympy==1.11.1",
        "tqdm==4.65.0"
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
    ],
)
