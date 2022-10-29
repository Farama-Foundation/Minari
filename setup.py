import os

from setuptools import Extension, find_packages, setup

os.environ["CFLAGS"] = "-std=c++11"

if __name__ == "__main__":
    from Cython.Build import cythonize
    from numpy import get_include

    # setup Cython build
    ext = Extension(
        "kabuki.dataset",
        sources=["kabuki/dataset.pyx"],
        include_dirs=[get_include(), "kabuki/cpp/include"],
        language="c++",
        extra_compile_args=["-std=c++11", "-O3", "-ffast-math"],
        extra_link_args=["-std=c++11"],
    )

    ext_modules = cythonize(
        [ext], compiler_directives={"linetrace": True, "binding": True}
    )

    # main setup
    setup(
        name="kabuki",
        version="0.0.1",
        description="An offline deep reinforcement learning library",
        long_description=open("README.md").read(),
        long_description_content_type="text/markdown",
        url="https://github.com/takuseno/d3rlpy",
        author="Takuma Seno",
        author_email="takuma.seno@gmail.com",
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
            "google-cloud-storage==4.4.0",
            "protobuf==3.20.1",
        ],
        packages=find_packages(exclude=["tests*"]),
        python_requires=">=3.7.0",
        package_data={"d3rlpy": ["*.pyx", "*.pxd", "*.h", "*.pyi", "py.typed"]},
        ext_modules=ext_modules,
    )
