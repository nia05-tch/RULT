from setuptools import setup, find_packages

setup(
    name="rult",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "torch>=1.9.0",
        "lightgbm>=3.3.0",
        "h5py>=3.1.0",
    ],
    python_requires=">=3.8",
)