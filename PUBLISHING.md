1. bump version in `setup.py`
2. `python3 setup.py sdist bdist_wheel`
3. `auditwheel repair dist/minari-*-cp39-cp39-linux_x86_64.whl --plat manylinux_2_34_x86_64`
4. `mv wheelhouse/* dist`
5. `rm dist/minari-*-cp39-cp39-linux_x86_64.whl`
6. `twine upload --repository pypi dist/*`
7. `rm -r build dist wheelhouse minari.egg.info`

On Ubuntu 22.04, you can follow https://levelup.gitconnected.com/how-to-deploy-a-cython-package-to-pypi-8217a6581f09 but using `auditwheel repair dist/minari-0.2.0-cp39-cp39-linux_x86_64.whl --plat manylinux_2_34_x86_64`.

Ideally we'd use a Docker container to avoid OS x toolchain compatibility issues, see https://github.com/pypa/auditwheel/issues/291#issuecomment-791936357
