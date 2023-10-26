# Minari documentation

This folder contains the documentation for Minari.

For more information about how to contribute to the documentation go to our [CONTRIBUTING.md](https://github.com/Farama-Foundation/Celshast/blob/main/CONTRIBUTING.md)

## Build the Documentation

Install the required packages and Minari:

```
git clone https://github.com/Farama-Foundation/Minari.git
cd Minari
pip install -e .
pip install -r docs/requirements.txt
```

To build the documentation once:

```
cd docs
make dirhtml
```

To rebuild the documentation automatically every time a change is made:

```
cd docs
sphinx-autobuild -b dirhtml . _build
```
