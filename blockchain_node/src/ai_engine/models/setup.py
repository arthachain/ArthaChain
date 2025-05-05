from setuptools import setup, find_packages

setup(
    name="blockchain_ai_models",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.13.0",
        "numpy>=1.22.0",
    ],
    author="Blockchain Team",
    description="AI models for blockchain neural processing",
) 