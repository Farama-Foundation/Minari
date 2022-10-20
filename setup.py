"""Sets up the Kabuki module."""

from setuptools import find_packages, setup


def get_description():
    """Gets the description from the readme."""
    with open("README.md") as fh:
        long_description = ""
        header_count = 0
        for line in fh:
            if line.startswith("##"):
                header_count += 1
            if header_count < 2:
                long_description += line
            else:
                break
    return header_count, long_description


def get_version():
    """Gets the kabuki version."""
    path = "kabuki/__init__.py"
    with open(path) as file:
        lines = file.readlines()

    for line in lines:
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("bad version data in __init__.py")


extras = {"robotics": ["gymnasium-robotics==1.0.1"]}

extras["all"] = (
    extras["atari"]
    + extras["classic"]
    + extras["butterfly"]
    + extras["mpe"]
    + extras["sisl"]
    + extras["other"]
    + extras["tests"]
)

version = get_version()
header_count, long_description = get_description()

setup(
    name="Kabuki",
    version=version,
    author="Farama Foundation",
    author_email="contact@farama.org",
    description="Gymnasium for offline reinforcement learning",
    url="https://kabuki.farama.org/",
    license_files=("LICENSE.txt",),
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=["Reinforcement Learning", "game", "RL", "AI", "gymnasium"],
    python_requires=">=3.7, <3.11",
    packages=["kabuki"] + ["kabuki." + pkg for pkg in find_packages("kabuki")],
    include_package_data=True,
    install_requires=["numpy>=1.18.0", "gymnasium>=0.26.0"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    extras_require=extras,
)
