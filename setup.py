import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="betsi",
    version="1.0.0",
    description="BET Surface Identification - a program that fully implements the rouquerol criteria",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/fairen-group/betsi-gui",
    author="James Rampersad & Johannes W.M. Osterrieth",
    author_email="nr472@cam.ac.uk",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["betsi"],
    include_package_data=True,
    install_requires=["pyqt", "matplotlib", "scipy", "numpy", "pathlib", "pandas", "seaborn", "statsmodels"],
    entry_points={
        "console_scripts": [
            "betsi=betsi.gui:runbetsi",
        ]
    },
)