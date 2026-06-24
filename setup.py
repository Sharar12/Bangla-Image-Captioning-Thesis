from setuptools import setup, find_packages

setup(
    name="bangla-image-captioning",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "transformers>=4.40.0",
        "nltk>=3.8.1",
        "gradio>=4.20.0",
        "fastapi>=0.110.0",
    ],
    author="Sharar Hossain",
    description="Bangla Image Captioning Research Framework v2.0.0",
)
