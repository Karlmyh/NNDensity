from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name='NNDE',

    version='0.0.1',

    packages=find_packages(),

    description='Adaptive Weighted Nearest Neighbor Density Estimation',

    long_description=long_description,

    long_description_content_type="text/markdown",

    url='https://github.com/Karlmyh/NNDE',

    author="Karlmyh",

    author_email="yma@ruc.edu.cn",

    python_requires='>=3'
)
