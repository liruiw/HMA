from setuptools import find_packages, setup

setup(
    name="hma",
    version="1.0",
    packages=find_packages(include=["hma", "hma.*"]),
)
