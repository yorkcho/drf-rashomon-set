from setuptools import setup

setup(
    name = 'pydrf',
    version = '0.5',
    description = 'A package for performing deep rule forest',
    packages = ['pydrf'],
    install_requires = ['numpy', 'scikit-learn', 'scipy', 'joblib']
)