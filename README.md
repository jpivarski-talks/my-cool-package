# my-cool-package

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![GitHub Discussion][github-discussions-badge]][github-discussions-link]

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/jpivarski-talks/my-cool-package/workflows/CI/badge.svg
[actions-link]:             https://github.com/jpivarski-talks/my-cool-package/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/my-cool-package
[conda-link]:               https://github.com/conda-forge/my-cool-package-feedstock
[github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/jpivarski-talks/my-cool-package/discussions
[pypi-link]:                https://pypi.org/project/my-cool-package/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/my-cool-package
[pypi-version]:             https://img.shields.io/pypi/v/my-cool-package
[rtd-badge]:                https://readthedocs.org/projects/my-cool-package/badge/?version=latest
[rtd-link]:                 https://my-cool-package.readthedocs.io/en/latest/?badge=latest

<!-- prettier-ignore-end -->

## How I made it

Using [scientific-python/cookie](https://github.com/scientific-python/cookie)!

```console

% cookiecutter gh:scientific-python/cookie
You've downloaded /home/jpivarski/.cookiecutters/cookie before. Is it okay to
delete and re-download it? [y/n] (y): y
  [1/9] The name of your project (package): my-cool-package
  [2/9] The name of your (GitHub?) org (org): jpivarski-talks
  [3/9] The url to your GitHub or GitLab repository
(https://github.com/jpivarski-talks/my-cool-package):
  [4/9] Your name (My Name): Jim Pivarski
  [5/9] Your email (me@email.com): jpivarski@gmail.com
  [6/9] A short description of your project (A great package.): It's so cool!
  [7/9] Select a license
    1 - BSD
    2 - Apache
    3 - MIT
    Choose from [1/2/3] (1): 1
  [8/9] Choose a build backend
    1 - Hatchling                      - Pure Python (recommended)
    2 - Flit-core                      - Pure Python (minimal)
    3 - PDM-backend                    - Pure Python
    4 - Whey                           - Pure Python
    5 - Poetry                         - Pure Python
    6 - Setuptools with pyproject.toml - Pure Python
    7 - Setuptools with setup.py       - Pure Python
    8 - Setuptools and pybind11        - Compiled C++
    9 - Scikit-build-core              - Compiled C++ (recommended)
    10 - Meson-python                  - Compiled C++ (also good)
    11 - Maturin                       - Compiled Rust (recommended)
    Choose from [1/2/3/4/5/6/7/8/9/10/11] (1): 1
  [9/9] Use version control for versioning [y/n] (y):
```
