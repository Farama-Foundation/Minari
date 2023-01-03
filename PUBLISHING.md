On Ubuntu 22.04, you can follow https://levelup.gitconnected.com/how-to-deploy-a-cython-package-to-pypi-8217a6581f09 but using `auditwheel repair dist/minari-0.2.0-cp39-cp39-linux_x86_64.whl --plat manylinux_2_34_x86_64`.

Ideally we'd use a Docker container to avoid OS x toolchain compatibility issues, see https://github.com/pypa/auditwheel/issues/291#issuecomment-791936357
