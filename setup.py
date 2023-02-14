import os

from setuptools import find_packages, setup

if __name__ == "__main__":
    from numpy import get_include


    # main setup
    setup(
        name="minari",
        version="0.2.2",
        description="Datasets for offline deep reinforcement learning",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        url="https://github.com/Farama-Foundation/Minari",
        author="Will Dudley",
        author_email="will2346@live.co.uk",
        license="MIT License",
        classifiers=[
            "Development Status :: 2 - Pre-Alpha",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: Implementation :: CPython",
        ],
        install_requires=[
            "numpy>=1.18.0",
            "h5py==3.7.0",
            "structlog==22.1.0",
            "tensorboardX==2.4",
            "typing_extensions==4.4.0",
            "google-cloud-storage==2.5.0",
            "protobuf==3.20.1",
            "gymnasium @ git+https://github.com/WillDudley/Gymnasium.git@spec_stack",
        ],
        packages=find_packages(exclude=["tests*"]),
        python_requires=">=3.7.0",
        # package_data={"minari": ["*.pyx", "*.pxd", "*.h", "*.pyi", "py.typed"]},
    )
