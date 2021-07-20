import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

with open("requirements.txt", "r") as fh:
    REQUIREMENTS = [line.strip().split(";")[0] for line in fh]

# This call to setup() does all the work
setup(
    name="betsi-gui",
    version="1.0.20",
    description="BET Surface Identification - a program that fully implements the rouquerol criteria",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/fairen-group/betsi-gui",
    author="James Rampersad, Johannes W.M. Osterrieth & Nakul Rampal",
    author_email="nr472@cam.ac.uk",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    python_requires=">=3.7.3",
    packages=find_packages(),
    include_package_data=True,
    install_requires= REQUIREMENTS,
    entry_points={
        "console_scripts": [
            "betsi=betsi.__main__:main",
        ]
    },
)