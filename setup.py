from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="lora_adapters",
    version="0.1.3",
    author="Kian Sierra McGettigan",
    author_email="kiansierra90@gmail.com",
    description="PyTorch implementation of low-rank adaptation (LoRA), a parameter-efficient approach to adapt a large pre-trained deep learning model which obtains performance on-par with full fine-tuning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages("src"),
    url="https://github.com/kiansierra/lora-adapters",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
