from setuptools import Extension, setup

setup(
    ext_modules = [Extension("_kd_tree", ["_kd_tree.pyx"])]
)