import os

from setuptools import Extension, find_packages, setup

os.environ["CFLAGS"] = "-std=c++11"

if __name__ == "__main__":
    from Cython.Build import cythonize
    from numpy import get_include

    # setup Cython build
    ext = Extension(
        "minari.dataset",
        sources=["minari/dataset.pyx"],
        include_dirs=[get_include(), "minari/cpp/include"],
        language="c++",
        extra_compile_args=["-std=c++11", "-O3", "-ffast-math"],
        extra_link_args=["-std=c++11"],
    )

    ext_modules = cythonize(
        [ext], compiler_directives={"linetrace": True, "binding": True}
    )

    # main setup
    setup(
        name="minari",
        version="0.1.0",
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
            "gymnasium>=0.26"
        ],
        packages=find_packages(exclude=["tests*"]),
        python_requires=">=3.7.0",
        package_data={"minari": ["*.pyx", "*.pxd", "*.h", "*.pyi", "py.typed"]},
        ext_modules=ext_modules,
    )
